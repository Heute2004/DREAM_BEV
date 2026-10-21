"""Thin adapters around vendored public radar models.

The adapters never alter an official model's internal architecture. They only make its
input layout and final-frame confidence map compatible with the fixed public protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .models import RadarOutput


def _official_rodnet_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "RODNet_original"


def _official_trodnet_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "T-RODNet_original"


def _load_rodnet_cdc() -> type[nn.Module]:
    root = _official_rodnet_root()
    if not (root / "rodnet" / "models" / "rodnet_cdc.py").is_file():
        raise FileNotFoundError(f"missing vendored official RODNet source: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from rodnet.models.rodnet_cdc import RODNetCDC

    return RODNetCDC


def _load_rodnet_hg() -> type[nn.Module]:
    root = _official_rodnet_root()
    if not (root / "rodnet" / "models" / "rodnet_hg.py").is_file():
        raise FileNotFoundError(f"missing vendored official RODNet source: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from rodnet.models.rodnet_hg import RODNetHG

    return RODNetHG


def _load_trodnet() -> type[nn.Module]:
    root = _official_trodnet_root()
    if not (root / "Nets" / "T_RODNet.py").is_file():
        raise FileNotFoundError(f"missing vendored official T-RODNet source: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from Nets.T_RODNet import T_RODNet

    return T_RODNet


class OfficialRODNetCDC(nn.Module):
    """Official RODNet CDC body with a last-frame adapter for the fixed causal protocol."""

    def __init__(self, window_length: int = 8) -> None:
        super().__init__()
        self.window_length = int(window_length)
        self.official = _load_rodnet_cdc()(in_channels=2, n_class=1)

    def forward(self, radar: torch.Tensor) -> RadarOutput:
        if radar.ndim != 5 or radar.shape[1:] != (self.window_length, 2, 128, 128):
            raise ValueError(f"expected [B,{self.window_length},2,128,128], got {tuple(radar.shape)}")
        probability_volume = self.official(radar.permute(0, 2, 1, 3, 4))
        probability = probability_volume[:, :, -1].clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(probability)
        return RadarOutput(logits=logits, feature=probability_volume)


class OfficialRODNetHG(nn.Module):
    """Official stacked-hourglass RODNet body with the same final-frame adapter."""

    def __init__(self, window_length: int = 8, stacked_num: int = 2) -> None:
        super().__init__()
        self.window_length = int(window_length)
        self.official = _load_rodnet_hg()(in_channels=2, n_class=1, stacked_num=stacked_num)

    def forward(self, radar: torch.Tensor) -> RadarOutput:
        if radar.ndim != 5 or radar.shape[1:] != (self.window_length, 2, 128, 128):
            raise ValueError(f"expected [B,{self.window_length},2,128,128], got {tuple(radar.shape)}")
        stages = self.official(radar.permute(0, 2, 1, 3, 4))
        probability_volume = stages[-1]
        probability = probability_volume[:, :, -1].clamp(1e-6, 1.0 - 1e-6)
        logits = torch.logit(probability)
        return RadarOutput(logits=logits, feature=probability_volume)


class OfficialTRODNet(nn.Module):
    """Official T-RODNet body with a last-frame adapter for the fixed causal protocol.

    The constructor follows the authors' public CRUW training entry point
    (``embed_dim=64``, ``win_size=4``); only the output is reduced to the
    currently observed frame for the shared pedestrian-only evaluator.
    """

    def __init__(self, window_length: int = 8) -> None:
        super().__init__()
        self.window_length = int(window_length)
        self.official = _load_trodnet()(num_classes=1, embed_dim=64, win_size=4)

    def forward(self, radar: torch.Tensor) -> RadarOutput:
        if radar.ndim != 5 or radar.shape[1:] != (self.window_length, 2, 128, 128):
            raise ValueError(f"expected [B,{self.window_length},2,128,128], got {tuple(radar.shape)}")
        logits_volume = self.official(radar.permute(0, 2, 1, 3, 4))
        return RadarOutput(logits=logits_volume[:, :, -1], feature=logits_volume)
