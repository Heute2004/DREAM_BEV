from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn

from .backbones import DREAMBEV, extract_dream_bev_state


class MultiScaleDetailAdapter(nn.Module):
    """Zero-initialized local and dilated detail extraction for one route."""

    def __init__(self, channels: int, detail_channels: int) -> None:
        super().__init__()
        if detail_channels < 8:
            raise ValueError("detail_channels must be at least 8")
        groups = 8 if detail_channels % 8 == 0 else 1
        self.project = nn.Conv2d(channels, detail_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, detail_channels)
        self.local = nn.Conv2d(
            detail_channels,
            detail_channels,
            kernel_size=3,
            padding=1,
            groups=detail_channels,
            bias=False,
        )
        self.context = nn.Conv2d(
            detail_channels,
            detail_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=detail_channels,
            bias=False,
        )
        self.merge = nn.Conv2d(detail_channels * 2, detail_channels, kernel_size=1, bias=False)
        gate_channels = max(detail_channels // 4, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(detail_channels, gate_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_channels, detail_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.up = nn.Conv2d(detail_channels, channels, kernel_size=1, bias=True)
        # Exact identity at initialization. The final projection receives a
        # non-zero gradient immediately, then activates the detail branches.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: Tensor) -> Tensor:
        value = torch.nn.functional.silu(self.norm(self.project(feature)))
        local = torch.nn.functional.silu(self.local(value))
        context = torch.nn.functional.silu(self.context(value))
        value = torch.nn.functional.silu(self.merge(torch.cat((local, context), dim=1)))
        return self.up(value * self.gate(value))


class DREAMRouteExpertBEV(nn.Module):
    """Frozen DREAM-BEV backbone plus one dual-domain route-specific expert."""

    def __init__(self, polar_detail_channels: int = 64, bev_detail_channels: int = 64) -> None:
        super().__init__()
        self.backbone = DREAMBEV()
        self.polar_adapter = MultiScaleDetailAdapter(48, polar_detail_channels)
        self.bev_adapter = MultiScaleDetailAdapter(128, bev_detail_channels)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "DREAMRouteExpertBEV":
        super().train(mode)
        self.backbone.eval()
        return self

    def load_base_checkpoint(self, checkpoint_path: str | Path) -> str:
        path = Path(checkpoint_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(extract_dream_bev_state(payload), strict=True)
        self._freeze_backbone()
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def expert_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.polar_adapter.parameters()
        yield from self.bev_adapter.parameters()

    def expert_state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {
            "polar_adapter": self.polar_adapter.state_dict(),
            "bev_adapter": self.bev_adapter.state_dict(),
        }

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        self.backbone.eval()
        with torch.no_grad():
            polar_feature = self.backbone.forward_polar_feature(radar)
            base_shared_feature = self.backbone.shared_from_polar_feature(polar_feature)
            base_logits = self.backbone.decoder.decode_shared(base_shared_feature)
        polar_residual = self.polar_adapter(polar_feature)
        shared_feature = self.backbone.shared_from_polar_feature(polar_feature + polar_residual)
        bev_residual = self.bev_adapter(shared_feature)
        adapted_feature = shared_feature + bev_residual
        logits = self.backbone.decoder.decode_shared(adapted_feature)
        return {
            "logits": logits,
            "shared_feature": adapted_feature,
            "base_logits": base_logits,
            "polar_adapter_residual": polar_residual,
            "bev_adapter_residual": bev_residual,
        }



