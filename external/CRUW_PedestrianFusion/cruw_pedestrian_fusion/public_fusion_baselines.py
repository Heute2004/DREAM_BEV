from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import math
import subprocess
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import constant_, xavier_uniform_

from .public_fusion_evidence import (
    ExternalFusionOutput,
    RadarEvidence,
    VisualBearingEvidence,
    VisualEvidence,
    visual_evidence_to_bev_ray_map,
)
from .allscene import CRUWBEVGeometry


def recenter_logits_on_radar_candidate(
    logits: Tensor,
    base_xy_m: Tensor,
    candidate_xy_m: Tensor,
    geometry: CRUWBEVGeometry,
) -> Tensor:
    """Translate a frozen radar map so its maximum follows a radar-owned candidate."""

    if tuple(base_xy_m.shape) != tuple(candidate_xy_m.shape) or base_xy_m.shape[-1] != 2:
        raise ValueError("base and candidate positions must be aligned [batch,2]")
    output = logits.new_full(logits.shape, float(logits.detach().amin()))
    for index in range(logits.shape[0]):
        dx = int(torch.round((candidate_xy_m[index, 0] - base_xy_m[index, 0]) / geometry.resolution_m))
        dy = int(torch.round((candidate_xy_m[index, 1] - base_xy_m[index, 1]) / geometry.resolution_m))
        source_x0 = max(0, -dx)
        source_x1 = min(geometry.x_bins, geometry.x_bins - dx)
        source_y0 = max(0, -dy)
        source_y1 = min(geometry.y_bins, geometry.y_bins - dy)
        target_x0 = source_x0 + dx
        target_x1 = source_x1 + dx
        target_y0 = source_y0 + dy
        target_y1 = source_y1 + dy
        if source_x1 > source_x0 and source_y1 > source_y0:
            output[index, :, target_x0:target_x1, target_y0:target_y1] = logits[
                index, :, source_x0:source_x1, source_y0:source_y1
            ]
    return output


@dataclass(frozen=True)
class PublicBaselineSpec:
    display_name: str
    repository_url: str
    commit: str
    repository_dir: str
    required_files: tuple[str, ...]


PUBLIC_BASELINE_SPECS = {
    "rcrosnet": PublicBaselineSpec(
        display_name="RC-ROSNet",
        repository_url="https://github.com/Zhuanglong2/RC-ROSNet.git",
        commit="8ff57d49efa90934dca2032adf7f6625b5ad9248",
        repository_dir="RC-ROSNet",
        required_files=("mvrss/models/RCRODNet.py",),
    ),
    "crn": PublicBaselineSpec(
        display_name="CRN",
        repository_url="https://github.com/youngskkim/CRN.git",
        commit="5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d",
        repository_dir="CRN",
        required_files=(
            "layers/fuser/multimodal_feature_aggregation.py",
            "layers/modules/multimodal_deformable_cross_attention.py",
        ),
    ),
    "rcbevdet": PublicBaselineSpec(
        display_name="RCBEVDet",
        repository_url="https://github.com/VDIGPKU/RCBEVDet.git",
        commit="15c83ccdd5a8cfd3b7c0390eacd8664cf842d513",
        repository_dir="RCBEVDet",
        required_files=(
            "official_source/rcbevdet-master/mmdet3d/models/detectors/bevdet_rc.py",
            "official_source/rcbevdet-master/mmdet3d/models/model_utils/ops/modules/ms_deform_attn.py",
            "official_source/rcbevdet-master/mmdet3d/models/model_utils/ops/functions/ms_deform_attn_func.py",
        ),
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_public_source(name: str, repository_root: Path) -> dict[str, object]:
    if name not in PUBLIC_BASELINE_SPECS:
        raise ValueError(f"unknown public baseline: {name}")
    spec = PUBLIC_BASELINE_SPECS[name]
    repository_root = Path(repository_root)
    if not repository_root.is_dir() or not (repository_root / ".git").exists():
        raise ValueError(f"official source repository is missing: {repository_root}")
    required = [repository_root / relative for relative in spec.required_files]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"official source files are missing: {missing}")
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if commit != spec.commit:
        raise ValueError(
            f"official source commit mismatch for {name}: expected {spec.commit}, got {commit}"
        )
    return {
        "official_model": spec.display_name,
        "repository_url": spec.repository_url,
        "commit": commit,
        "source_files": {
            str(path.relative_to(repository_root)).replace("\\", "/"): _sha256(path)
            for path in required
        },
    }


def _prepend_sys_path(path: Path) -> None:
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def _residual_head(channels: int) -> nn.Sequential:
    head = nn.Sequential(
        nn.Conv2d(channels + 1, 128, 3, padding=1, bias=False),
        nn.GroupNorm(8, 128),
        nn.SiLU(inplace=True),
        nn.Conv2d(128, 1, 1),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class _OfficialOutputMixin:
    official_name: str

    def _make_output(self, radar: RadarEvidence, fused_feature: Tensor) -> ExternalFusionOutput:
        radar_low = F.interpolate(
            radar.logits,
            size=fused_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        delta = self.output_head(torch.cat((fused_feature, radar_low), dim=1))
        delta = F.interpolate(
            delta,
            size=radar.logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return ExternalFusionOutput(
            fused_logits=radar.logits + delta,
            radar_logits=radar.logits,
            delta_logits=delta,
            metadata={
                "official_model": self.official_name,
                "radar_anchor": "frozen_cruw_expert",
                "official_fusion_core_modified": False,
                "task_io_adapted": True,
                "range_preserving_constraint": False,
                "depth_input": False,
                "future_frame_input": False,
            },
        )


class RCROSNetOfficialFusionBaseline(nn.Module, _OfficialOutputMixin):
    official_name = "RC-ROSNet"

    def __init__(
        self,
        repository_root: Path | None = None,
        *,
        official_core: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.source_metadata: dict[str, object] = {"source": "injected_test_core"}
        if official_core is None:
            if repository_root is None:
                raise ValueError("repository_root is required for RC-ROSNet")
            repository_root = Path(repository_root)
            self.source_metadata = validate_public_source("rcrosnet", repository_root)
            _prepend_sys_path(repository_root)
            module = importlib.import_module("mvrss.models.RCRODNet")
            module_file = Path(module.__file__).resolve()
            if repository_root.resolve() not in module_file.parents:
                raise RuntimeError(f"RC-ROSNet imported from unexpected path: {module_file}")
            official_core = module.FusionModule()
        self.official_core = official_core
        self.radar_input = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.SiLU(inplace=True),
        )
        self.visual_input = nn.Conv2d(128, 3, 1)
        self.output_head = _residual_head(256)

    def forward(self, radar: RadarEvidence, visual: VisualEvidence) -> ExternalFusionOutput:
        radar.validate()
        visual_map = visual_evidence_to_bev_ray_map(
            visual, spatial_size=radar.bev_feature.shape[-2:]
        )
        fused = self.official_core(
            self.radar_input(radar.bev_feature), self.visual_input(visual_map)
        )
        return self._make_output(radar, fused)


def _ms_deform_attn_core_pytorch(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch, _, heads, dimensions = value.shape
    _, queries, _, levels, points, _ = sampling_locations.shape
    split_sizes = [int(height * width) for height, width in value_spatial_shapes.tolist()]
    value_list = value.split(split_sizes, dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled_values = []
    for level, (height, width) in enumerate(value_spatial_shapes.tolist()):
        level_value = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * heads, dimensions, int(height), int(width))
        )
        level_grid = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled_values.append(
            F.grid_sample(
                level_value,
                level_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads, 1, queries, levels * points
    )
    output = (
        torch.stack(sampled_values, dim=-2).flatten(-2) * weights
    ).sum(-1).view(batch, heads * dimensions, queries)
    return output.transpose(1, 2).contiguous()


class _PurePytorchMSDAFunction:
    @staticmethod
    def apply(
        value: Tensor,
        value_spatial_shapes: Tensor,
        value_level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        im2col_step: int,
    ) -> Tensor:
        del value_level_start_index, im2col_step
        return _ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )


def _register_crn_positional_encoding(dependency_root: Path) -> types.ModuleType:
    # Load only the official registry module. Importing the whole legacy mmdet
    # package would eagerly require unrelated compiled MMCV operators.
    module_name = "_dream_if_crn_mmdet_positional_encoding"
    cached = sys.modules.get(module_name)
    if isinstance(cached, types.ModuleType):
        return cached
    source = (
        Path(dependency_root)
        / "mmdet"
        / "models"
        / "utils"
        / "positional_encoding.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"CRN positional encoding source missing: {source}")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load CRN positional encoding source: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_crn_core(repository_root: Path, dependency_root: Path) -> nn.Module:
    _prepend_sys_path(dependency_root)
    _prepend_sys_path(repository_root)
    _register_crn_positional_encoding(dependency_root)
    compatibility_name = "layers.modules.multi_scale_deformable_attn_function"
    compatibility_module = types.ModuleType(compatibility_name)
    compatibility_module.MultiScaleDeformableAttnFunction_fp32 = _PurePytorchMSDAFunction
    compatibility_module.MultiScaleDeformableAttnFunction_fp16 = _PurePytorchMSDAFunction
    sys.modules[compatibility_name] = compatibility_module
    module = importlib.import_module("layers.fuser.multimodal_feature_aggregation")
    module_file = Path(module.__file__).resolve()
    if repository_root.resolve() not in module_file.parents:
        raise RuntimeError(f"CRN imported from unexpected path: {module_file}")
    return module.MFAFuser(
        num_sweeps=1,
        img_dims=128,
        pts_dims=128,
        embed_dims=256,
        num_layers=6,
        num_heads=4,
        bev_shape=(50, 59),
    )


class CRNOfficialFusionBaseline(nn.Module, _OfficialOutputMixin):
    official_name = "CRN"

    def __init__(
        self,
        repository_root: Path | None = None,
        *,
        dependency_root: Path | None = None,
        official_core: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.source_metadata: dict[str, object] = {"source": "injected_test_core"}
        if official_core is None:
            if repository_root is None or dependency_root is None:
                raise ValueError("repository_root and dependency_root are required for CRN")
            repository_root = Path(repository_root)
            self.source_metadata = validate_public_source("crn", repository_root)
            self.source_metadata["operator_backend"] = "differentiable_pytorch_grid_sample"
            official_core = _load_crn_core(repository_root, Path(dependency_root))
        self.official_core = official_core
        self.output_head = _residual_head(256)

    def forward(self, radar: RadarEvidence, visual: VisualEvidence) -> ExternalFusionOutput:
        radar.validate()
        visual_map = visual_evidence_to_bev_ray_map(
            visual, spatial_size=radar.bev_feature.shape[-2:]
        )
        features = torch.cat((visual_map, radar.bev_feature), dim=1).unsqueeze(1)
        output = self.official_core(features)
        fused = output[0] if isinstance(output, tuple) else output
        return self._make_output(radar, fused)


def _load_ast_symbols(
    source_file: Path,
    names: set[str],
    namespace: dict[str, Any],
) -> dict[str, Any]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    found = {node.name for node in selected}
    if found != names:
        raise RuntimeError(f"missing official symbols in {source_file}: {sorted(names - found)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_file), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _RCBEVDetCAMFCore(nn.Module):
    def __init__(self, repository_root: Path, dependency_root: Path) -> None:
        super().__init__()
        _prepend_sys_path(dependency_root)
        from einops import rearrange
        from mmcv.cnn import normal_init
        from mmcv.runner import BaseModule

        code_root = repository_root / "official_source" / "rcbevdet-master"
        attention_file = code_root / "mmdet3d/models/model_utils/ops/modules/ms_deform_attn.py"
        fuser_file = code_root / "mmdet3d/models/detectors/bevdet_rc.py"
        attention_namespace: dict[str, Any] = {
            "torch": torch,
            "nn": nn,
            "F": F,
            "warnings": warnings,
            "math": math,
            "xavier_uniform_": xavier_uniform_,
            "constant_": constant_,
            "BaseModule": BaseModule,
            "rearrange": rearrange,
            "MSDeformAttnFunction": _PurePytorchMSDAFunction,
        }
        symbols = _load_ast_symbols(
            attention_file,
            {"_is_power_of_2", "LearnedPositionalEncoding3D", "MSDeformAttn"},
            attention_namespace,
        )
        fuser_namespace: dict[str, Any] = {
            "torch": torch,
            "nn": nn,
            "BaseModule": BaseModule,
            "normal_init": normal_init,
        }
        fuser_symbols = _load_ast_symbols(fuser_file, {"RadarConvFuser"}, fuser_namespace)
        attention_class = symbols["MSDeformAttn"]
        position_class = symbols["LearnedPositionalEncoding3D"]
        fuser_class = fuser_symbols["RadarConvFuser"]
        self.attention_radar_to_image = attention_class(
            d_model=256, n_levels=1, n_heads=8, n_points=8
        )
        self.attention_image_to_radar = attention_class(
            d_model=256, n_levels=1, n_heads=8, n_points=8
        )
        self.position_radar = position_class(
            num_feats=128, row_num_embed=50, col_num_embed=59
        )
        self.position_image = position_class(
            num_feats=128, row_num_embed=50, col_num_embed=59
        )
        self.fuser = fuser_class(
            in_channels=(256, 256), out_channels=256, deconv_blocks=3
        )

    @staticmethod
    def _reference_points(
        batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        y, x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, device=device, dtype=dtype),
            torch.linspace(0.5, width - 0.5, width, device=device, dtype=dtype),
            indexing="ij",
        )
        points = torch.stack((x.reshape(-1) / width, y.reshape(-1) / height), dim=-1)
        return points[None, :, None].repeat(batch, 1, 1, 1)

    def forward(self, radar: Tensor, visual: Tensor) -> Tensor:
        batch, channels, height, width = radar.shape
        radar_tokens = radar.flatten(2).transpose(1, 2)
        visual_tokens = visual.flatten(2).transpose(1, 2)
        mask = torch.zeros(batch, height, width, device=radar.device, dtype=torch.bool)
        radar_pos = self.position_radar(mask).to(radar.dtype)
        visual_pos = self.position_image(mask).to(radar.dtype)
        reference = self._reference_points(batch, height, width, radar.device, radar.dtype)
        spatial_shapes = torch.tensor([[height, width]], device=radar.device, dtype=torch.long)
        level_start = torch.zeros(1, device=radar.device, dtype=torch.long)
        radar_aligned = self.attention_radar_to_image(
            query=radar_tokens + radar_pos,
            reference_points=reference,
            input_flatten=visual_tokens + visual_pos,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start,
            input_padding_mask=None,
        )
        visual_aligned = self.attention_image_to_radar(
            query=visual_tokens + visual_pos,
            reference_points=reference,
            input_flatten=radar_tokens + radar_pos,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start,
            input_padding_mask=None,
        )
        radar_aligned = radar_aligned.transpose(1, 2).reshape(batch, channels, height, width)
        visual_aligned = visual_aligned.transpose(1, 2).reshape(batch, channels, height, width)
        return self.fuser(radar_aligned, visual_aligned)


class RCBEVDetOfficialFusionBaseline(nn.Module, _OfficialOutputMixin):
    official_name = "RCBEVDet"

    def __init__(
        self,
        repository_root: Path | None = None,
        *,
        dependency_root: Path | None = None,
        official_core: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.source_metadata: dict[str, object] = {"source": "injected_test_core"}
        if official_core is None:
            if repository_root is None or dependency_root is None:
                raise ValueError(
                    "repository_root and dependency_root are required for RCBEVDet"
                )
            repository_root = Path(repository_root)
            self.source_metadata = validate_public_source("rcbevdet", repository_root)
            self.source_metadata["operator_backend"] = (
                "official_repository_pytorch_grid_sample_fallback"
            )
            official_core = _RCBEVDetCAMFCore(repository_root, Path(dependency_root))
        self.official_core = official_core
        self.radar_input = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.SiLU(inplace=True),
        )
        self.visual_input = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False),
            nn.GroupNorm(16, 256),
            nn.SiLU(inplace=True),
        )
        self.output_head = _residual_head(256)

    def forward(self, radar: RadarEvidence, visual: VisualEvidence) -> ExternalFusionOutput:
        radar.validate()
        visual_map = visual_evidence_to_bev_ray_map(
            visual, spatial_size=radar.bev_feature.shape[-2:]
        )
        fused = self.official_core(
            self.radar_input(radar.bev_feature), self.visual_input(visual_map)
        )
        return self._make_output(radar, fused)


def build_public_official_baseline(name: str, public_source_root: Path) -> nn.Module:
    public_source_root = Path(public_source_root)
    dependency_root = public_source_root / "python_deps" / "mmcv117"
    if name == "rcrosnet":
        return RCROSNetOfficialFusionBaseline(
            public_source_root / PUBLIC_BASELINE_SPECS[name].repository_dir
        )
    if name == "crn":
        return CRNOfficialFusionBaseline(
            public_source_root / PUBLIC_BASELINE_SPECS[name].repository_dir,
            dependency_root=dependency_root,
        )
    if name == "rcbevdet":
        return RCBEVDetOfficialFusionBaseline(
            public_source_root / PUBLIC_BASELINE_SPECS[name].repository_dir,
            dependency_root=dependency_root,
        )
    raise ValueError(f"unknown public baseline: {name}")
