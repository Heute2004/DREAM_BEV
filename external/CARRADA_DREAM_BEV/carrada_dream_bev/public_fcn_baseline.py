"""Source-pinned CARRADA FCN-8s baseline under the shared BEV contract.

The official CARRADA repository cites the public ``pytorch-fcn`` FCN-8s
implementation as its segmentation baseline.  This wrapper keeps that spatial
FCN-8s topology unchanged, adapting only its RGB stem to the frozen radar
evidence contract: the eight causal RA/RA-MTI frames are stacked as 16 input
channels.  It therefore receives exactly the same observed window as
DREAM-BEV, without a temporal recurrence or token mixer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import torch
from torch import nn

from .canonical_dream_bev import CARRADAPolarToBEV
from .data import CARRADABEVGeometry


CARRADA_REPOSITORY = "https://github.com/valeoai/carrada_dataset.git"
CARRADA_COMMIT = "f3f9beb56f2a30eecd5a8c4af8c76035027e6f7d"
PYTORCH_FCN_REPOSITORY = "https://github.com/wkentaro/pytorch-fcn.git"
PYTORCH_FCN_COMMIT = "4b988509bfd1a613d5e8482595a1c6654047d69b"
FCN8S_SOURCE = (
    "torchfcn/models/fcn8s.py",
    "102ac8534350964e498e02f28e1d76c93467fab51d8a561356289464d2d3605e",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_path(root: Path, relative_path: str, expected_hash: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"missing pinned public FCN source: {path}")
    actual_hash = _sha256(path)
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(
            f"public FCN source hash mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}"
        )
    return path


def _radar_initialize(model: nn.Module) -> None:
    """Initialize the unmodified public topology for non-RGB radar inputs."""

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def _load_fcn8s(
    public_root: Path,
    *,
    causal_frames: int = 8,
) -> tuple[nn.Module, dict[str, object]]:
    if causal_frames < 1:
        raise ValueError("causal_frames must be positive")
    source_path = _checked_path(public_root, *FCN8S_SOURCE)
    package_root = public_root / "torchfcn"
    if not package_root.is_dir():
        raise FileNotFoundError(f"missing pytorch-fcn package directory: {package_root}")

    # The upstream source imports itself as ``fcn`` although its install name is
    # ``torchfcn``.  A temporary namespace alias loads the untouched source in
    # the current runtime without editing third-party files.
    previous_fcn = sys.modules.get("fcn")
    shimmed = previous_fcn is None
    if shimmed:
        package = ModuleType("fcn")
        package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        sys.modules["fcn"] = package
    try:
        module = importlib.import_module("fcn.models.fcn8s")
        backbone = module.FCN8s(n_class=1)
    finally:
        if shimmed:
            sys.modules.pop("fcn.models.fcn8s", None)
            sys.modules.pop("fcn.models.fcn32s", None)
            sys.modules.pop("fcn.models.fcn16s", None)
            sys.modules.pop("fcn.models", None)
            sys.modules.pop("fcn", None)
        elif previous_fcn is not None:
            sys.modules["fcn"] = previous_fcn

    # The original FCN is an RGB network.  Only its input projection changes:
    # [RA(t-T+1), RA-MTI(t-T+1), ..., RA(t), RA-MTI(t)] -> 2T channels.
    backbone.conv1_1 = nn.Conv2d(2 * causal_frames, 64, kernel_size=3, padding=100)
    _radar_initialize(backbone)
    return backbone, {
        "carrada_repository": CARRADA_REPOSITORY,
        "carrada_commit": CARRADA_COMMIT,
        "carrada_baseline_reference": "official README credits pytorch-fcn for the FCN baseline",
        "repository": PYTORCH_FCN_REPOSITORY,
        "commit": PYTORCH_FCN_COMMIT,
        "source_file": FCN8S_SOURCE[0],
        "source_sha256": _sha256(source_path),
        "source_modified": False,
        "architecture": "FCN-8s",
        "input_contract": f"RA+RA-MTI stacked from causal {causal_frames}-frame evidence",
        "task_input_adaptation": f"replace only RGB conv1_1: 3 -> {2 * causal_frames} channels",
        "initialization": "Kaiming-normal scratch initialization for radar adaptation; no image pretraining",
        "execution": f"static spatial FCN over the same causally observed {causal_frames}-frame evidence",
    }


def prepare_fcn_input(radar: torch.Tensor, *, causal_frames: int = 8) -> torch.Tensor:
    """Stack one fixed causal evidence window as channels for the 2-D FCN."""

    if causal_frames < 1:
        raise ValueError("causal_frames must be positive")
    if radar.ndim != 5 or radar.shape[1:] != (causal_frames, 2, 128, 128):
        raise ValueError(
            f"radar must be [batch,{causal_frames},2,128,128] under the causal contract"
        )
    return radar.reshape(radar.shape[0], 2 * causal_frames, 128, 128).contiguous()


@dataclass(frozen=True)
class CARRADAFCNOutput:
    logits: torch.Tensor
    polar_logits: torch.Tensor


class CARRADAFCN8sBEV(nn.Module):
    """Public FCN-8s spatial head followed by the shared polar-to-BEV map."""

    def __init__(
        self,
        public_root: str | Path | None,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CARRADABEVGeometry | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry or CARRADABEVGeometry()
        if backbone is None:
            if public_root is None:
                raise ValueError("public_root is required when no test backbone is supplied")
            self.backbone, self.public_source = _load_fcn8s(Path(public_root))
        else:
            self.backbone = backbone
            self.public_source = {
                "test_backbone": type(backbone).__name__,
                "source_modified": False,
                "input_contract": "RA+RA-MTI stacked from causal [t-7,...,t]",
            }
        self.polar_to_bev = CARRADAPolarToBEV(range_grid, angle_grid, self.geometry)
        support = (self.polar_to_bev.sampling_grid.abs() <= 1.0).all(dim=-1).unsqueeze(1)
        self.register_buffer("valid_bev_mask", support, persistent=False)

    def forward(self, radar: torch.Tensor) -> CARRADAFCNOutput:
        polar_logits = self.backbone(prepare_fcn_input(radar))
        if polar_logits.ndim != 4 or polar_logits.shape[1:] != (1, 128, 128):
            raise ValueError(f"expected FCN output [B,1,128,128], got {tuple(polar_logits.shape)}")
        bev_logits = self.polar_to_bev(polar_logits)
        bev_logits = torch.where(
            self.valid_bev_mask,
            bev_logits,
            torch.full_like(bev_logits, -20.0),
        )
        return CARRADAFCNOutput(logits=bev_logits, polar_logits=polar_logits)
