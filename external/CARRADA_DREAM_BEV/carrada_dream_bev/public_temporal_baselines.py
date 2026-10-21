"""Source-pinned temporal polar radar baselines under the shared BEV contract.

The models retain their public feature extractors and polar output heads.  This
module only handles the fixed CARRADA comparison contract: two RA-derived
channels, causal eight-frame history, a current-frame RA readout, and the
parameter-free polar-to-BEV sampler used by every public polar baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from .canonical_dream_bev import BEVDecoder, CARRADAPolarToBEV
from .data import CARRADABEVGeometry


PublicTemporalVariant = Literal[
    "t_rodnet",
    "t_rodnet_compact",
    "trodnet_core",
    "erodnet_core",
    "e_rodnet",
    "private_erodnet_core",
]

T_RODNET_REPOSITORY = "https://github.com/Zhuanglong2/T-RODNet.git"
T_RODNET_COMMIT = "39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9"
T_RODNET_SOURCE = ("Nets/T_RODNet.py", "9822dff4bb282266bf4975a068d62b3da412ed384078cbbd977d238f7a57641d")
T_RODNET_REFERENCE_CONFIGURATION = {"embed_dim": 64, "win_size": 7}
T_RODNET_COMPACT_CONFIGURATION = {
    "embed_dim": 64,
    "win_size": 4,
    "inactive_modules": [
        "swin_unet.layers.0.downsamplelast",
        "swin_unet.layers.1.downsample",
        "swin_unet.layers.2.downsamplelast",
    ],
}

E_RODNET_REPOSITORY = "https://github.com/lupeng-xm/E-RODNet.git"
E_RODNET_COMMIT = "da4be8011090ad565a85cc39f849ad9f0688ef14"
E_RODNET_SOURCE = (
    "nets/E_RODNet/E_RODNet.py",
    "76919b6edfdab1db73a70df940967b0c4659eb5df0e84c8b30d5538030ada937",
)
E_RODNET_CONFIGURATION = {"SFF_channels": (4, 32), "n_class": 1}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_path(root: Path, relative_path: str, expected_hash: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned public source: {path}")
    actual_hash = _sha256(path)
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(
            f"public source hash mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
        )
    return path


def _load_standalone_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load public source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prune_t_rodnet_inactive_branches(backbone: nn.Module) -> nn.Module:
    """Remove only branches verified unreachable by the frozen eight-frame graph."""

    if getattr(backbone, "embed_dim", None) != T_RODNET_COMPACT_CONFIGURATION["embed_dim"]:
        raise ValueError("compact T-RODNet requires embed_dim=64")
    try:
        layers = backbone.swin_unet.layers
    except AttributeError as error:
        raise ValueError("compact T-RODNet requires the public Swin encoder layers") from error
    if len(layers) != 3:
        raise ValueError("compact T-RODNet requires three public Swin encoder layers")
    for layer, attribute in (
        (layers[0], "downsamplelast"),
        (layers[1], "downsample"),
        (layers[2], "downsamplelast"),
    ):
        if not hasattr(layer, attribute):
            raise ValueError(f"compact T-RODNet missing expected {attribute} branch")
        setattr(layer, attribute, nn.Identity())
    return backbone


def _load_t_rodnet(
    public_root: Path,
    configuration: dict[str, int] | None = None,
    compact: bool = False,
) -> tuple[nn.Module, dict[str, object]]:
    source_path = _checked_path(public_root, *T_RODNET_SOURCE)
    module = _load_standalone_module(source_path, "_pinned_t_rodnet")
    runtime_configuration = dict(configuration or T_RODNET_REFERENCE_CONFIGURATION)
    backbone = module.T_RODNet(num_classes=1, FLOPs=False, **runtime_configuration)
    if compact:
        expected = {
            "embed_dim": int(T_RODNET_COMPACT_CONFIGURATION["embed_dim"]),
            "win_size": int(T_RODNET_COMPACT_CONFIGURATION["win_size"]),
        }
        if runtime_configuration != expected:
            raise ValueError(f"compact T-RODNet is pinned to {expected}")
        before = sum(parameter.numel() for parameter in backbone.parameters())
        backbone = prune_t_rodnet_inactive_branches(backbone)
        if sum(parameter.numel() for parameter in backbone.parameters()) >= before:
            raise RuntimeError("compact T-RODNet did not remove inactive parameters")
    metadata: dict[str, object] = {
        "repository": T_RODNET_REPOSITORY,
        "commit": T_RODNET_COMMIT,
        "source_file": T_RODNET_SOURCE[0],
        "source_sha256": _sha256(source_path),
        "source_modified": False,
        "task_input_adaptation": "none",
        "source_trainer_configuration": runtime_configuration,
        "execution": "causal_fixed_8_frame_window",
    }
    if compact:
        metadata["execution_graph_adaptation"] = (
            "replaces only three branches verified unreachable under the causal "
            "eight-frame RA+RA-MTI contract; the pinned public source remains unmodified"
        )
        metadata["inactive_modules"] = T_RODNET_COMPACT_CONFIGURATION["inactive_modules"]
    return backbone, metadata


def _load_e_rodnet(public_root: Path) -> tuple[nn.Module, dict[str, object]]:
    source_path = _checked_path(public_root, *E_RODNET_SOURCE)
    module = _load_standalone_module(source_path, "_pinned_e_rodnet")
    try:
        backbone = module.E_RODNet(**E_RODNET_CONFIGURATION)
    except AttributeError as error:
        raise ImportError("pinned E-RODNet source does not expose E_RODNet") from error
    return backbone, {
        "repository": E_RODNET_REPOSITORY,
        "commit": E_RODNET_COMMIT,
        "source_file": E_RODNET_SOURCE[0],
        "source_sha256": _sha256(source_path),
        "source_modified": False,
        "task_input_adaptation": (
            "causal left zero padding from the observed eight frames to E-RODNet's native "
            "sixteen-frame tensor, followed by parameter-free repetition into four chirp slots"
        ),
        "source_trainer_configuration": E_RODNET_CONFIGURATION,
        "execution": "causal_eight_observed_frame_window_with_unobserved_history_zero_padding",
    }


def load_public_temporal_backbone(
    variant: PublicTemporalVariant,
    public_root: str | Path,
    t_rodnet_configuration: dict[str, int] | None = None,
) -> tuple[nn.Module, dict[str, object]]:
    root = Path(public_root)
    if variant in {"t_rodnet", "t_rodnet_compact"}:
        return _load_t_rodnet(
            root,
            configuration=t_rodnet_configuration,
            compact=variant == "t_rodnet_compact",
        )
    if variant == "e_rodnet":
        return _load_e_rodnet(root)
    raise ValueError(f"unsupported public temporal variant: {variant}")


def prepare_public_temporal_input(radar: torch.Tensor) -> torch.Tensor:
    """Convert a causal shared `[B,T,C,R,A]` tensor to public `[B,C,T,R,A]`."""

    if radar.ndim != 5 or radar.shape[1] < 1 or radar.shape[2:] != (2, 128, 128):
        raise ValueError("radar must be [batch,time,2,128,128] under a causal contract")
    return radar.permute(0, 2, 1, 3, 4).contiguous()


def prepare_e_rodnet_input(radar: torch.Tensor) -> torch.Tensor:
    """Adapt the shared eight observed frames to E-RODNet without adding evidence."""

    observed = prepare_public_temporal_input(radar)
    padded = functional.pad(observed, (0, 0, 0, 0, 8, 0))
    return padded.unsqueeze(3).expand(-1, -1, -1, 4, -1, -1).contiguous()


def trodnet_core_source_metadata(public_root: str | Path) -> dict[str, object]:
    """Record the public source motif without instantiating the full U-Net."""

    source_path = _checked_path(Path(public_root), *T_RODNET_SOURCE)
    return {
        "repository": T_RODNET_REPOSITORY,
        "commit": T_RODNET_COMMIT,
        "source_file": T_RODNET_SOURCE[0],
        "source_sha256": _sha256(source_path),
        "source_modified": False,
        "model_form": "compact core adaptation",
        "task_input_adaptation": "none; causal eight-frame RA+RA-MTI",
        "retained_public_motif": "3D polar encoding followed by temporal window attention",
        "omitted_public_components": "full Swin U-Net encoder-decoder and task-specific multi-class head",
    }


def erodnet_core_source_metadata(public_root: str | Path) -> dict[str, object]:
    """Record the E-RODNet feature-mixing motif retained by the compact core."""

    source_path = _checked_path(Path(public_root), *E_RODNET_SOURCE)
    return {
        "repository": E_RODNET_REPOSITORY,
        "commit": E_RODNET_COMMIT,
        "source_file": E_RODNET_SOURCE[0],
        "source_sha256": _sha256(source_path),
        "source_modified": False,
        "model_form": "compact core adaptation",
        "task_input_adaptation": "none; causal eight-frame RA+RA-MTI",
        "retained_public_motif": "3D MetaFormer-style token mixing",
        "omitted_public_components": "chirp SFF, full GFF decoder, and task-specific multi-class head",
    }


def select_current_polar_logits(output: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Normalize public current or temporal one-class RA outputs to logits."""

    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("public temporal backbone returned an empty output stack")
        output = output[-1]
    if output.ndim == 5:
        if output.shape[1] != 1 or output.shape[-2:] != (128, 128):
            raise ValueError(f"expected [B,1,T,128,128], got {tuple(output.shape)}")
        output = output[:, :, -1]
    if output.ndim != 4 or output.shape[1:] != (1, 128, 128):
        raise ValueError(f"expected [B,1,128,128], got {tuple(output.shape)}")
    return output


@dataclass(frozen=True)
class PublicTemporalOutput:
    logits: torch.Tensor
    polar_logits: torch.Tensor


class _MetaFormer3DBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.token_mixer = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 5, 5),
            padding=(1, 2, 2),
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, channels * 2, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
        )
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.token_mixer(self.norm1(value))
        return value + self.mlp(self.norm2(value))


class ERODNetCoreBEV(nn.Module):
    """Compact E-RODNet-style 3D MetaFormer core under the shared BEV contract."""

    def __init__(
        self,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CARRADABEVGeometry | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or CARRADABEVGeometry()
        self.stem = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList([_MetaFormer3DBlock(64) for _ in range(3)])
        self.reduce = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
        )
        self.polar_to_bev = CARRADAPolarToBEV(range_grid, angle_grid, self.geometry)
        self.decoder = BEVDecoder(48, self.geometry)
        support = (self.polar_to_bev.sampling_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
        self.register_buffer("valid_bev_mask", support, persistent=False)

    def forward(self, radar: torch.Tensor) -> PublicTemporalOutput:
        if radar.ndim != 5 or radar.shape[1:] != (8, 2, 128, 128):
            raise ValueError("radar must be [batch,8,2,128,128] under the frozen causal contract")
        feature = self.stem(prepare_public_temporal_input(radar))
        for block in self.blocks:
            feature = block(feature)
        polar_feature = self.reduce(feature[:, :, -1])
        polar_logits = polar_feature.mean(dim=1, keepdim=True)
        shared_feature = self.decoder.encode_shared(self.polar_to_bev(polar_feature))
        logits = self.decoder.decode_shared(shared_feature)
        logits = torch.where(self.valid_bev_mask, logits, torch.full_like(logits, -20.0))
        return PublicTemporalOutput(logits=logits, polar_logits=polar_logits)


def _private_erodnet_types(private_root: Path) -> tuple[type[nn.Module], dict[str, object]]:
    """Load the actual private ERODNetCoreBEV class from a declared source root."""

    root = private_root.expanduser().resolve()
    source = root / "radar_bev" / "models" / "backbones.py"
    if not source.is_file():
        raise FileNotFoundError(f"private DREAM-BEV model source not found: {source}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("radar_bev.models.backbones")
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"radar_bev is already imported from a different source root: {module_path}"
        ) from error
    model_type = getattr(module, "ERODNetCoreBEV")
    return model_type, {
        "repository": "DREAM-BEV private implementation",
        "source_file": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": _sha256(source),
        "source_modified": False,
        "model_form": "direct cross-dataset reuse",
        "task_input_adaptation": "CARRADA polar grid only; causal eight-frame RA+RA-MTI",
        "retained_private_components": "private ERODNetCoreBEV stem, MetaFormer blocks, reducer, and BEV decoder",
    }


class PrivateERODNetCoreBEV(nn.Module):
    """Run the private ERODNetCoreBEV implementation on the CARRADA grid."""

    def __init__(
        self,
        private_root: Path,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CARRADABEVGeometry | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or CARRADABEVGeometry()
        model_type, self.private_source = _private_erodnet_types(private_root)
        self.model = model_type()
        self.model.geometry = self.geometry
        self.model.polar_to_bev = CARRADAPolarToBEV(range_grid, angle_grid, self.geometry)
        self.model.decoder.geometry = self.geometry
        support = (self.model.polar_to_bev.sampling_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
        self.register_buffer("valid_bev_mask", support, persistent=False)
        self._polar_feature: torch.Tensor | None = None
        self.model.reduce.register_forward_hook(self._capture_polar_feature)

    def _capture_polar_feature(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._polar_feature = output

    def forward(self, radar: torch.Tensor) -> PublicTemporalOutput:
        self._polar_feature = None
        output = self.model(radar)
        if self._polar_feature is None:
            raise RuntimeError("private ERODNetCoreBEV did not emit its polar feature")
        logits = torch.where(self.valid_bev_mask, output["logits"], torch.full_like(output["logits"], -20.0))
        return PublicTemporalOutput(logits=logits, polar_logits=self._polar_feature.mean(dim=1, keepdim=True))


class TRODNetCoreBEV(nn.Module):
    """Compact T-RODNet-style temporal core under CARRADA's fixed BEV contract."""

    def __init__(
        self,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CARRADABEVGeometry | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or CARRADABEVGeometry()
        self.stem = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.temporal_attention = nn.MultiheadAttention(64, num_heads=4, batch_first=True)
        self.temporal_norm = nn.LayerNorm(64)
        self.reduce = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
        )
        self.polar_to_bev = CARRADAPolarToBEV(range_grid, angle_grid, self.geometry)
        self.decoder = BEVDecoder(48, self.geometry)
        support = (self.polar_to_bev.sampling_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
        self.register_buffer("valid_bev_mask", support, persistent=False)

    def forward(self, radar: torch.Tensor) -> PublicTemporalOutput:
        if radar.ndim != 5 or radar.shape[1:] != (8, 2, 128, 128):
            raise ValueError("radar must be [batch,8,2,128,128] under the frozen causal contract")
        feature = self.stem(prepare_public_temporal_input(radar))
        batch, channels, frames, height, width = feature.shape
        tokens = feature.permute(0, 3, 4, 2, 1).reshape(batch * height * width, frames, channels)
        attended, _ = self.temporal_attention(tokens, tokens, tokens, need_weights=False)
        latest = self.temporal_norm(tokens[:, -1] + attended[:, -1])
        latest = latest.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        polar_feature = functional.interpolate(
            self.reduce(latest), size=(128, 128), mode="bilinear", align_corners=False
        )
        polar_logits = polar_feature.mean(dim=1, keepdim=True)
        shared_feature = self.decoder.encode_shared(self.polar_to_bev(polar_feature))
        logits = self.decoder.decode_shared(shared_feature)
        logits = torch.where(self.valid_bev_mask, logits, torch.full_like(logits, -20.0))
        return PublicTemporalOutput(logits=logits, polar_logits=polar_logits)


class PublicTemporalBEV(nn.Module):
    """Public temporal polar detector followed by the fixed sensor-to-BEV map."""

    def __init__(
        self,
        variant: PublicTemporalVariant,
        public_root: str | Path | None,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CARRADABEVGeometry | None = None,
        backbone: nn.Module | None = None,
        t_rodnet_configuration: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.geometry = geometry or CARRADABEVGeometry()
        if backbone is None:
            if public_root is None:
                raise ValueError("public_root is required when no test backbone is supplied")
            self.backbone, self.public_source = load_public_temporal_backbone(
                variant,
                public_root,
                t_rodnet_configuration=t_rodnet_configuration,
            )
        else:
            self.backbone = backbone
            self.public_source = {
                "test_backbone": type(backbone).__name__,
                "task_input_adaptation": (
                    "none"
                    if variant in {"t_rodnet", "t_rodnet_compact"}
                    else "none_public_in_channels_constructor"
                ),
            }
        self.polar_to_bev = CARRADAPolarToBEV(range_grid, angle_grid, self.geometry)
        support = (self.polar_to_bev.sampling_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
        self.register_buffer("valid_bev_mask", support, persistent=False)

    def forward(self, radar: torch.Tensor) -> PublicTemporalOutput:
        public_input = prepare_public_temporal_input(radar)
        if self.variant == "e_rodnet":
            polar_probability = select_current_polar_logits(self.backbone(prepare_e_rodnet_input(radar)))
            polar_logits = torch.logit(polar_probability.clamp(min=1e-4, max=1.0 - 1e-4))
        else:
            raw_output = self.backbone(public_input)
            polar_logits = select_current_polar_logits(raw_output)
        bev_logits = self.polar_to_bev(polar_logits)
        bev_logits = torch.where(
            self.valid_bev_mask,
            bev_logits,
            torch.full_like(bev_logits, -20.0),
        )
        return PublicTemporalOutput(logits=bev_logits, polar_logits=polar_logits)


def public_source_json(metadata: dict[str, object]) -> str:
    """Stable metadata rendering for run ledgers and focused tests."""

    return json.dumps(metadata, sort_keys=True)
