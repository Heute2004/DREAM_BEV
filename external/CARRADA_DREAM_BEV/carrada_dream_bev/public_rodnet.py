"""Thin adapters around unmodified public RODNet backbones.

The public source is loaded directly from the pinned T-RODNet repository. The
adapter only permutes the shared causal input tensor, selects the last temporal
output, and applies the fixed CARRADA Polar-to-BEV sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import torch
from torch import nn

from .canonical_dream_bev import CARRADAPolarToBEV
from .data import CARRADABEVGeometry


PublicRODNetVariant = Literal["rodnet_cdc", "rodnet_hg", "rodnet_hgwi"]

PUBLIC_REPOSITORY = "https://github.com/Zhuanglong2/T-RODNet.git"
PUBLIC_COMMIT = "39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9"
PUBLIC_SOURCE_FILES: dict[PublicRODNetVariant, tuple[str, str]] = {
    "rodnet_cdc": (
        "rodnet/models/backbones/cdc.py",
        "0ec4705836bfc9798801019d74990cf48d6914b1c22e36976cf0c458a5d12292",
    ),
    "rodnet_hg": (
        "rodnet/models/backbones/hg.py",
        "0a9ca14bb0d20e95aefb9c6d838c39c0212fdd1380469408bbe3da0deb38f48c",
    ),
    "rodnet_hgwi": (
        "rodnet/models/backbones/hgwi.py",
        "6e9901fed1a3d7bf424235caf0a76ee6fa57b90384be0ba4387b14c5c2b35754",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_module(path: Path, variant: PublicRODNetVariant) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_pinned_public_{variant}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load public RODNet source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_public_backbone(
    variant: PublicRODNetVariant,
    public_root: str | Path,
) -> tuple[nn.Module, dict[str, object]]:
    relative_path, expected_hash = PUBLIC_SOURCE_FILES[variant]
    source_path = Path(public_root) / relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"missing pinned public source: {source_path}")
    actual_hash = _sha256(source_path)
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(
            f"public source hash mismatch for {variant}: expected {expected_hash}, got {actual_hash}"
        )
    module = _load_source_module(source_path, variant)
    if variant == "rodnet_cdc":
        backbone = module.RadarVanilla(in_channels=2, n_class=1, use_mse_loss=True)
    else:
        backbone = module.RadarStackedHourglass(
            in_channels=2,
            n_class=1,
            stacked_num=1,
            use_mse_loss=True,
        )
    metadata = {
        "repository": PUBLIC_REPOSITORY,
        "commit": PUBLIC_COMMIT,
        "source_file": relative_path,
        "source_sha256": actual_hash,
        "source_modified": False,
    }
    return backbone, metadata


def prepare_public_rodnet_input(radar: torch.Tensor) -> torch.Tensor:
    """Convert a causal shared [B,T,C,R,A] input to public [B,C,T,R,A]."""

    if radar.ndim != 5 or radar.shape[1] < 1 or radar.shape[2:] != (2, 128, 128):
        raise ValueError("radar must be [batch,time,2,128,128]")
    return radar.permute(0, 2, 1, 3, 4).contiguous()


def select_last_polar_logits(output: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    """Use the last public stack and the current (last) causal frame only."""

    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("public RODNet returned an empty output stack")
        output = output[-1]
    if output.ndim != 5 or output.shape[1] != 1 or output.shape[-2:] != (128, 128):
        raise ValueError(f"expected public output [B,1,T,128,128], got {tuple(output.shape)}")
    return output[:, :, -1]


@dataclass(frozen=True)
class PublicRODNetOutput:
    logits: torch.Tensor
    polar_logits: torch.Tensor


class PublicRODNetBEV(nn.Module):
    """Unmodified public RODNet backbone under the shared CARRADA BEV contract."""

    def __init__(
        self,
        variant: PublicRODNetVariant,
        public_root: str | Path | None,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CARRADABEVGeometry | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.geometry = geometry or CARRADABEVGeometry()
        if backbone is None:
            if public_root is None:
                raise ValueError("public_root is required when no test backbone is supplied")
            self.backbone, self.public_source = load_public_backbone(variant, public_root)
        else:
            self.backbone = backbone
            self.public_source = {"test_backbone": type(backbone).__name__}
        self.polar_to_bev = CARRADAPolarToBEV(range_grid, angle_grid, self.geometry)
        sampling_grid = self.polar_to_bev.sampling_grid
        valid = (sampling_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
        self.register_buffer("valid_bev_mask", valid, persistent=False)

    def forward(self, radar: torch.Tensor) -> PublicRODNetOutput:
        public_input = prepare_public_rodnet_input(radar)
        polar_logits = select_last_polar_logits(self.backbone(public_input))
        bev_logits = self.polar_to_bev(polar_logits)
        bev_logits = torch.where(self.valid_bev_mask, bev_logits, torch.full_like(bev_logits, -20.0))
        return PublicRODNetOutput(
            logits=bev_logits,
            polar_logits=polar_logits,
        )
