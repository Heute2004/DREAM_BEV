"""Source-pinned full public radar backbones under DREAM-BEV's BEV contract.

The public model bodies are not edited. This module applies documented,
parameter-free input layouts, selects the current polar response, and maps
that response to the fixed metric BEV lattice without learned parameters.
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

import torch
from torch import Tensor, nn

from .geometry import BEVGeometry, PolarToBEV


PublicFullVariant = Literal[
    "t_rodnet_compact",
    "rodnet_cdc",
    "rodnet_hg",
    "fcn8s",
    "e_rodnet",
]


@dataclass(frozen=True)
class PublicFullSpec:
    paper_name: str
    repository: str
    commit: str
    source_directory: str
    source_file: str
    source_sha256: str
    source_modified: bool = False


PUBLIC_FULL_SPECS: dict[PublicFullVariant, PublicFullSpec] = {
    "t_rodnet_compact": PublicFullSpec(
        paper_name="T-RODNet-core-adapted",
        repository="https://github.com/Zhuanglong2/T-RODNet.git",
        commit="39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9",
        source_directory="T-RODNet_original",
        source_file="Nets/T_RODNet.py",
        source_sha256="9822dff4bb282266bf4975a068d62b3da412ed384078cbbd977d238f7a57641d",
    ),
    "rodnet_cdc": PublicFullSpec(
        paper_name="RODNet-CDC-adapted",
        repository="https://github.com/Zhuanglong2/T-RODNet.git",
        commit="39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9",
        source_directory="RODNet_original",
        source_file="rodnet/models/backbones/cdc.py",
        source_sha256="0ec4705836bfc9798801019d74990cf48d6914b1c22e36976cf0c458a5d12292",
    ),
    "rodnet_hg": PublicFullSpec(
        paper_name="RODNet-HG-adapted",
        repository="https://github.com/Zhuanglong2/T-RODNet.git",
        commit="39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9",
        source_directory="RODNet_original",
        source_file="rodnet/models/backbones/hg.py",
        source_sha256="0a9ca14bb0d20e95aefb9c6d838c39c0212fdd1380469408bbe3da0deb38f48c",
    ),
    "fcn8s": PublicFullSpec(
        paper_name="FCN-8s-adapted",
        repository="https://github.com/wkentaro/pytorch-fcn.git",
        commit="4b988509bfd1a613d5e8482595a1c6654047d69b",
        source_directory="pytorch_fcn_original",
        source_file="torchfcn/models/fcn8s.py",
        source_sha256="102ac8534350964e498e02f28e1d76c93467fab51d8a561356289464d2d3605e",
    ),
    "e_rodnet": PublicFullSpec(
        paper_name="E-RODNet-adapted",
        repository="https://github.com/lupeng-xm/E-RODNet.git",
        commit="da4be8011090ad565a85cc39f849ad9f0688ef14",
        source_directory="E-RODNet_original",
        source_file="nets/E_RODNet/E_RODNet.py",
        source_sha256="76919b6edfdab1db73a70df940967b0c4659eb5df0e84c8b30d5538030ada937",
    ),
}

# Match the source constructor used by the paper's full RODNet-HG adaptation.
RODNET_STACKED_NUM = {"rodnet_hg": 2}
T_RODNET_COMPACT_CONFIGURATION = {
    "embed_dim": 64,
    "win_size": 4,
    "inactive_modules": [
        "swin_unet.layers.0.downsamplelast",
        "swin_unet.layers.1.downsample",
        "swin_unet.layers.2.downsamplelast",
    ],
}
E_RODNET_CONFIGURATION = {"sff_channels": (4, 32), "n_class": 1}
FCN8S_INPUT_CHANNELS = 32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_path(source_root: Path, variant: PublicFullVariant) -> Path:
    spec = PUBLIC_FULL_SPECS[variant]
    path = source_root / spec.source_directory / spec.source_file
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned public source: {path}")
    actual = _sha256(path)
    if actual.lower() != spec.source_sha256:
        raise RuntimeError(
            f"public source hash mismatch for {variant}: expected {spec.source_sha256}, got {actual}"
        )
    return path


def public_source_metadata(variant: PublicFullVariant, source_root: str | Path) -> dict[str, object]:
    """Verify and describe the unmodified source used for one adapted run."""

    root = Path(source_root)
    spec = PUBLIC_FULL_SPECS[variant]
    primary = _checked_path(root, variant)
    metadata: dict[str, object] = {
        "paper_name": spec.paper_name,
        "repository": spec.repository,
        "commit": spec.commit,
        "source_file": str(primary.relative_to(root)).replace("\\", "/"),
        "source_sha256": _sha256(primary),
        "source_modified": False,
        "input_adapter": "RA+RA-MTI two-channel layout permutation only",
        "output_adapter": "current polar response plus fixed parameter-free Polar-to-BEV sampler",
        "causal_input": "16-frame [t-15,...,t] history; no future frames",
    }
    if variant == "rodnet_hg":
        metadata["source_configuration"] = {"stacked_num": RODNET_STACKED_NUM[variant]}
    elif variant == "t_rodnet_compact":
        metadata["source_configuration"] = T_RODNET_COMPACT_CONFIGURATION
        metadata["execution_graph_adaptation"] = (
            "removes only canonical-contract inactive downsamplelast modules; public source is unchanged"
        )
    elif variant == "fcn8s":
        metadata["source_configuration"] = {
            "architecture": "FCN-8s",
            "input_channels": FCN8S_INPUT_CHANNELS,
            "task_input_adaptation": "replace only RGB conv1_1: 3 -> 32 channels",
            "initialization": "Kaiming-normal scratch initialization for radar adaptation",
        }
    elif variant == "e_rodnet":
        metadata["input_adapter"] = (
            "RA+RA-MTI causal layout plus parameter-free repetition into E-RODNet's four chirp slots"
        )
        metadata["source_configuration"] = E_RODNET_CONFIGURATION
    return metadata


def _load_standalone(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load public source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prune_t_rodnet_inactive_modules(model: nn.Module) -> nn.Module:
    """Remove T-RODNet modules unreachable by the canonical 16-frame contract.

    With ``embed_dim=64``, the source body takes layer 0's standard downsampler,
    layer 1's ``downsamplelast`` branch, and no downsampler in layer 2. Replacing
    the three other branches preserves the executed source graph while removing
    their unused parameters from the compact comparison model.
    """

    if getattr(model, "embed_dim", None) != T_RODNET_COMPACT_CONFIGURATION["embed_dim"]:
        raise ValueError("T-RODNet compact pruning requires embed_dim=64")
    try:
        layers = model.swin_unet.layers
    except AttributeError as error:
        raise ValueError("T-RODNet compact pruning requires swin_unet.layers") from error
    if len(layers) != 3:
        raise ValueError("T-RODNet compact pruning requires three source stages")
    inactive_modules = (
        (layers[0], "downsamplelast"),
        (layers[1], "downsample"),
        (layers[2], "downsamplelast"),
    )
    for layer, attribute in inactive_modules:
        if not hasattr(layer, attribute):
            raise ValueError(f"T-RODNet compact pruning requires source {attribute} modules")
        setattr(layer, attribute, nn.Identity())
    return model


def _load_t_rodnet(source_root: Path) -> nn.Module:
    variant: PublicFullVariant = "t_rodnet_compact"
    module = _load_standalone(_checked_path(source_root, variant), f"_dream_bev_{variant}")
    model = module.T_RODNet(
        num_classes=1,
        FLOPs=False,
        embed_dim=int(T_RODNET_COMPACT_CONFIGURATION["embed_dim"]),
        win_size=int(T_RODNET_COMPACT_CONFIGURATION["win_size"]),
    )
    return prune_t_rodnet_inactive_modules(model)


def _load_rodnet(variant: Literal["rodnet_cdc", "rodnet_hg"], source_root: Path) -> nn.Module:
    module = _load_standalone(_checked_path(source_root, variant), f"_dream_bev_{variant}")
    if variant == "rodnet_cdc":
        return module.RadarVanilla(in_channels=2, n_class=1, use_mse_loss=True)
    return module.RadarStackedHourglass(
        in_channels=2,
        n_class=1,
        stacked_num=RODNET_STACKED_NUM[variant],
        use_mse_loss=True,
    )


def _with_source_path(root: Path, *additional_roots: Path):
    class _SourcePath:
        def __enter__(self) -> None:
            self.inserted: list[str] = []
            for candidate in (root, *additional_roots):
                text = str(candidate)
                if text not in sys.path:
                    sys.path.insert(0, text)
                    self.inserted.append(text)

        def __exit__(self, *_: object) -> None:
            for inserted in self.inserted:
                sys.path.remove(inserted)

    return _SourcePath()


def _initialize_fcn8s_for_radar(model: nn.Module) -> None:
    """Initialize the unchanged FCN-8s topology for scratch radar training."""

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _load_fcn8s(source_root: Path) -> nn.Module:
    root = source_root / PUBLIC_FULL_SPECS["fcn8s"].source_directory
    public_source_metadata("fcn8s", source_root)
    package_root = root / "torchfcn"
    if not package_root.is_dir():
        raise FileNotFoundError(f"missing pytorch-fcn package: {package_root}")

    # The public source imports itself as fcn. The temporary alias lets us
    # execute the pinned source without editing its package layout.
    previous_fcn = sys.modules.get("fcn")
    shimmed = previous_fcn is None
    if shimmed:
        package = ModuleType("fcn")
        package.__path__ = [str(package_root)]
        sys.modules["fcn"] = package
    try:
        module = importlib.import_module("fcn.models.fcn8s")
        backbone = module.FCN8s(n_class=1)
    finally:
        if shimmed:
            for name in ("fcn.models.fcn8s", "fcn.models.fcn32s", "fcn.models.fcn16s", "fcn.models", "fcn"):
                sys.modules.pop(name, None)
        elif previous_fcn is not None:
            sys.modules["fcn"] = previous_fcn

    backbone.conv1_1 = nn.Conv2d(FCN8S_INPUT_CHANNELS, 64, kernel_size=3, padding=100)
    _initialize_fcn8s_for_radar(backbone)
    return backbone


def _load_e_rodnet(source_root: Path) -> nn.Module:
    """Load the official E-RODNet body without modifying its source file."""

    module = _load_standalone(_checked_path(source_root, "e_rodnet"), "_dream_bev_e_rodnet")
    try:
        return module.E_RODNet(
            SFF_channels=E_RODNET_CONFIGURATION["sff_channels"],
            n_class=E_RODNET_CONFIGURATION["n_class"],
        )
    except AttributeError as error:  # pragma: no cover - source-pinning diagnostic
        raise ImportError("pinned E-RODNet source does not expose E_RODNet") from error


def load_public_full_backbone(variant: PublicFullVariant, source_root: str | Path) -> tuple[nn.Module, dict[str, object]]:
    root = Path(source_root)
    metadata = public_source_metadata(variant, root)
    if variant == "t_rodnet_compact":
        return _load_t_rodnet(root), metadata
    if variant in {"rodnet_cdc", "rodnet_hg"}:
        return _load_rodnet(variant, root), metadata
    if variant == "fcn8s":
        return _load_fcn8s(root), metadata
    if variant == "e_rodnet":
        return _load_e_rodnet(root), metadata
    raise ValueError(f"unsupported public full variant: {variant}")


def prepare_public_full_input(radar: Tensor) -> Tensor:
    """Convert shared [B,16,2,128,128] causal clips to public [B,2,16,128,128]."""

    if radar.ndim != 5 or radar.shape[1:] != (16, 2, 128, 128):
        raise ValueError("radar must be [batch,16,2,128,128]")
    return radar.permute(0, 2, 1, 3, 4).contiguous()


def prepare_fcn8s_input(radar: Tensor) -> Tensor:
    """Stack the same causal sixteen-frame evidence window for FCN-8s."""

    if radar.ndim != 5 or radar.shape[1:] != (16, 2, 128, 128):
        raise ValueError("radar must be [batch,16,2,128,128]")
    return radar.reshape(radar.shape[0], FCN8S_INPUT_CHANNELS, 128, 128).contiguous()


def prepare_e_rodnet_input(radar: Tensor) -> Tensor:
    """Repeat existing RA/RA-MTI evidence into E-RODNet's required chirp slots.

    The common contract contains no raw chirp dimension. Repetition is
    parameter-free and introduces no additional observation; it only satisfies
    the unmodified public body's six-dimensional input interface.
    """

    public_input = prepare_public_full_input(radar)
    return public_input.unsqueeze(3).expand(-1, -1, -1, 4, -1, -1).contiguous()


def _select_current_polar_logits(output: Tensor | list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("public backbone returned an empty output stack")
        output = output[-1]
    if output.ndim == 5:
        if output.shape[1] != 1 or output.shape[-2:] != (128, 128):
            raise ValueError(f"expected [B,1,T,128,128], got {tuple(output.shape)}")
        output = output[:, :, -1]
    if output.ndim != 4 or output.shape[1:] != (1, 128, 128):
        raise ValueError(f"expected [B,1,128,128], got {tuple(output.shape)}")
    return output


class PublicFullPolarBEVAdapter(nn.Module):
    """An unmodified public backbone behind DREAM-BEV's common metric output."""

    def __init__(
        self,
        variant: PublicFullVariant,
        source_root: str | Path | None,
        geometry: BEVGeometry | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.geometry = geometry or BEVGeometry()
        if backbone is None:
            if source_root is None:
                raise ValueError("source_root is required when no test backbone is supplied")
            self.backbone, self.public_source = load_public_full_backbone(variant, source_root)
        else:
            self.backbone = backbone
            self.public_source = {"test_backbone": type(backbone).__name__}
        self.polar_to_bev = PolarToBEV(self.geometry)
        support = (self.polar_to_bev.sampling_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
        self.register_buffer("valid_bev_mask", support, persistent=False)

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        if self.variant == "fcn8s":
            polar_logits = _select_current_polar_logits(self.backbone(prepare_fcn8s_input(radar)))
        elif self.variant == "e_rodnet":
            # The official E-RODNet decoder returns a sigmoid confidence map.
            # Restore logit space before applying the shared focal-logit loss.
            polar_probability = _select_current_polar_logits(self.backbone(prepare_e_rodnet_input(radar)))
            polar_logits = torch.logit(polar_probability.clamp(1e-4, 1.0 - 1e-4))
        else:
            polar_logits = _select_current_polar_logits(self.backbone(prepare_public_full_input(radar)))
        logits = self.polar_to_bev(polar_logits)
        logits = torch.where(self.valid_bev_mask, logits, torch.full_like(logits, -20.0))
        return {"logits": logits, "polar_logits": polar_logits}


def public_source_json(metadata: dict[str, object]) -> str:
    return json.dumps(metadata, sort_keys=True)
