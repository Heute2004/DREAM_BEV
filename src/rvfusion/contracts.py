from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RadarEvidence:
    polar_feature: Tensor
    bev_feature: Tensor
    logits: Tensor
    position_override_xy_m: Tensor | None = None
    position_override_mask: Tensor | None = None

    def validate(self) -> None:
        batch = self.logits.shape[0]
        if tuple(self.polar_feature.shape) != (batch, 48, 128, 128):
            raise ValueError("polar_feature must be [batch,48,128,128]")
        if tuple(self.bev_feature.shape) != (batch, 128, 50, 59):
            raise ValueError("bev_feature must be [batch,128,50,59]")
        if tuple(self.logits.shape) != (batch, 1, 200, 234):
            raise ValueError("logits must be [batch,1,200,234]")
        if (self.position_override_xy_m is None) != (
            self.position_override_mask is None
        ):
            raise ValueError(
                "position override coordinates and mask must be supplied together"
            )
        if self.position_override_xy_m is not None:
            assert self.position_override_mask is not None
            if tuple(self.position_override_xy_m.shape) != (batch, 2):
                raise ValueError("position_override_xy_m must be [batch,2]")
            if not torch.isfinite(self.position_override_xy_m).all():
                raise ValueError("position_override_xy_m must be finite")
            if (
                tuple(self.position_override_mask.shape) != (batch,)
                or self.position_override_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "position_override_mask must be boolean [batch]"
                )


@dataclass(frozen=True)
class VisualEvidence:
    boxes: Tensor
    raw_scores: Tensor
    raw_tokens: Tensor
    temporal_tokens: Tensor
    temporal_support: Tensor
    valid_mask: Tensor

    def validate(self) -> None:
        if self.boxes.ndim != 4 or self.boxes.shape[-1] != 4:
            raise ValueError("boxes must be [batch,time,proposals,4]")
        prefix = self.boxes.shape[:3]
        if tuple(self.raw_scores.shape) != tuple(prefix):
            raise ValueError("raw_scores must match boxes")
        if tuple(self.temporal_support.shape) != tuple(prefix):
            raise ValueError("temporal_support must match boxes")
        if tuple(self.valid_mask.shape) != tuple(prefix):
            raise ValueError("valid_mask must match boxes")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        if tuple(self.raw_tokens.shape[:3]) != tuple(prefix) or self.raw_tokens.shape[-1] != 128:
            raise ValueError("raw_tokens must be [batch,time,proposals,128]")
        if tuple(self.temporal_tokens.shape) != tuple(self.raw_tokens.shape):
            raise ValueError("temporal_tokens must match raw_tokens")


@dataclass(frozen=True)
class ConditionPrior:
    global_route: tuple[str, ...]
    visual_condition: tuple[str, ...]
    visual_reliability: tuple[str, ...]
    weather_surface: tuple[str, ...]

    def validate(self, batch_size: int | None = None) -> None:
        lengths = {
            len(self.global_route),
            len(self.visual_condition),
            len(self.visual_reliability),
            len(self.weather_surface),
        }
        if len(lengths) != 1:
            raise ValueError("condition fields must have the same batch length")
        if batch_size is not None and lengths != {batch_size}:
            raise ValueError(f"condition batch must contain {batch_size} items")


@dataclass(frozen=True)
class FusionOutput:
    fused_logits: Tensor
    radar_logits: Tensor
    delta_logits: Tensor
    acceptance: Tensor
    effective_acceptance: Tensor
    covariance: Tensor
    trust_weights: Tensor
    visual_trust_cap: Tensor
    visual_has_evidence: Tensor


@dataclass(frozen=True)
class TemporalBearingEvidence:
    theta: Tensor
    sigma_theta: Tensor
    temporal_feature: Tensor
    reliability: Tensor
    valid: Tensor

    def validate(self, batch_size: int | None = None) -> None:
        batch = int(self.theta.shape[0])
        if batch_size is not None and batch != batch_size:
            raise ValueError(f"bearing batch must contain {batch_size} items")
        if tuple(self.theta.shape) != (batch, 1):
            raise ValueError("theta must be [batch,1]")
        if tuple(self.sigma_theta.shape) != (batch, 1):
            raise ValueError("sigma_theta must be [batch,1]")
        if self.temporal_feature.ndim != 2 or self.temporal_feature.shape[0] != batch:
            raise ValueError("temporal_feature must be [batch,feature]")
        if tuple(self.reliability.shape) != (batch, 1):
            raise ValueError("reliability must be [batch,1]")
        if tuple(self.valid.shape) != (batch,) or self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [batch]")
        for name, value in (
            ("theta", self.theta),
            ("sigma_theta", self.sigma_theta),
            ("temporal_feature", self.temporal_feature),
            ("reliability", self.reliability),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if not (self.sigma_theta > 0.0).all():
            raise ValueError("sigma_theta must be positive")
        if not ((self.reliability >= 0.0) & (self.reliability <= 1.0)).all():
            raise ValueError("reliability must be in [0,1]")



