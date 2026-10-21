"""Common CRUW evidence contract for source-backed public fusion cores."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RadarEvidence:
    polar_feature: Tensor
    bev_feature: Tensor
    logits: Tensor

    def validate(self) -> None:
        batch = self.logits.shape[0]
        expected = {
            "polar_feature": (batch, 48, 128, 128),
            "bev_feature": (batch, 128, 50, 59),
            "logits": (batch, 1, 200, 234),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")


@dataclass(frozen=True)
class VisualBearingEvidence:
    theta: Tensor
    sigma_theta: Tensor
    reliability: Tensor
    temporal_feature: Tensor

    def validate(self) -> None:
        batch = self.theta.shape[0]
        if tuple(self.theta.shape) != (batch,):
            raise ValueError("theta must be [batch]")
        if tuple(self.sigma_theta.shape) != (batch,):
            raise ValueError("sigma_theta must be [batch]")
        if tuple(self.reliability.shape) != (batch,):
            raise ValueError("reliability must be [batch]")
        if tuple(self.temporal_feature.shape) != (batch, 96):
            raise ValueError("temporal_feature must be [batch,96]")
        if not self.sigma_theta.gt(0).all():
            raise ValueError("sigma_theta must be positive")
        if not ((self.reliability >= 0) & (self.reliability <= 1)).all():
            raise ValueError("reliability must be in [0,1]")


VisualEvidence = VisualBearingEvidence


@dataclass(frozen=True)
class ExternalFusionOutput:
    fused_logits: Tensor
    radar_logits: Tensor
    delta_logits: Tensor
    metadata: dict[str, object]


def visual_evidence_to_bev_ray_map(
    visual: VisualBearingEvidence,
    *,
    spatial_size: tuple[int, int] = (50, 59),
    x_extent_m: tuple[float, float] = (-10.0, 10.0),
    y_extent_m: tuple[float, float] = (0.0, 23.4),
) -> Tensor:
    """Render causal RGB bearing evidence as a depth-free soft BEV ray."""

    visual.validate()
    height, width = spatial_size
    device = visual.theta.device
    dtype = visual.temporal_feature.dtype
    x = torch.linspace(x_extent_m[0], x_extent_m[1], height, device=device, dtype=dtype)
    y = torch.linspace(y_extent_m[0], y_extent_m[1], width, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid_theta = torch.atan2(grid_x, grid_y.clamp_min(1e-3))
    difference = torch.atan2(
        torch.sin(grid_theta[None] - visual.theta[:, None, None]),
        torch.cos(grid_theta[None] - visual.theta[:, None, None]),
    )
    sigma = visual.sigma_theta[:, None, None].clamp(
        min=math.radians(0.25), max=math.radians(15.0)
    )
    support = torch.exp(-0.5 * (difference / sigma).square())
    support = support * visual.reliability[:, None, None]
    auxiliary = torch.stack(
        (
            torch.sin(visual.theta),
            torch.cos(visual.theta),
            visual.sigma_theta / math.pi,
            visual.reliability,
        ),
        dim=1,
    ).repeat(1, 8)
    feature = torch.cat((visual.temporal_feature, auxiliary.to(dtype)), dim=1)
    return feature[:, :, None, None] * support[:, None]
