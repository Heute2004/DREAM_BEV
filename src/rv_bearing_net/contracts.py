from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RVBearingOutput:
    bearing_logits: Tensor
    theta: Tensor
    sigma_theta: Tensor
    reliability: Tensor
    temporal_feature: Tensor
    person_u_logits: Tensor
    valid: Tensor

    def validate(
        self,
        *,
        batch_size: int,
        angle_bins: int,
        temporal_dim: int,
    ) -> None:
        expected = {
            "bearing_logits": (batch_size, angle_bins),
            "theta": (batch_size, 1),
            "sigma_theta": (batch_size, 1),
            "reliability": (batch_size, 1),
            "temporal_feature": (batch_size, temporal_dim),
            "person_u_logits": (batch_size, angle_bins),
            "valid": (batch_size,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} must have shape {shape}, got {tuple(value.shape)}"
                )
            if name != "valid" and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean")
        if not (self.sigma_theta > 0.0).all():
            raise ValueError("sigma_theta must be positive")
        if not (
            (self.reliability >= 0.0) & (self.reliability <= 1.0)
        ).all():
            raise ValueError("reliability must be in [0,1]")




