from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .geometry import BEVGeometry


@dataclass(frozen=True)
class BEVLossOutput:
    total: Tensor
    heatmap: Tensor
    coordinate: Tensor
    target: Tensor


class QualityAwareBEVLoss(nn.Module):
    """Metric Gaussian focal supervision for reviewed and radar-assisted tracks."""

    VARIANTS = (
        "bce_fixed_gaussian",
        "focal_fixed_gaussian",
        "quality_aware_focal",
        "quality_aware_focal_coordinate",
    )

    def __init__(
        self,
        geometry: BEVGeometry | None = None,
        alpha: float = 2.0,
        beta: float = 4.0,
        coordinate_weight: float = 0.05,
        softargmax_temperature: float = 0.25,
        variant: str = "quality_aware_focal_coordinate",
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unsupported loss variant: {variant}")
        self.geometry = geometry or BEVGeometry()
        self.alpha = alpha
        self.beta = beta
        self.coordinate_weight = coordinate_weight
        self.softargmax_temperature = softargmax_temperature
        self.variant = variant
        self.register_buffer("metric_grid", self.geometry.coordinate_grid(), persistent=False)

    def build_target(self, xy_m: Tensor, sample_weight: Tensor) -> Tensor:
        if xy_m.ndim != 2 or xy_m.shape[-1] != 2:
            raise ValueError("xy_m must be [batch, 2]")
        if sample_weight.shape != xy_m.shape[:1]:
            raise ValueError("sample_weight must be [batch]")
        grid = self.metric_grid.to(device=xy_m.device, dtype=xy_m.dtype)
        delta_x = grid[..., 0].unsqueeze(0) - xy_m[:, 0, None, None]
        delta_y = grid[..., 1].unsqueeze(0) - xy_m[:, 1, None, None]
        if self.variant.startswith("quality_aware_"):
            # 0.30 m for reviewed camera tracks; wider radar-assisted targets.
            sigma_y = 0.30 + (1.0 - sample_weight[:, None, None]) * 0.80
            sigma_x = sigma_y + xy_m[:, 1, None, None].clamp_min(0.0) * 0.006
        else:
            sigma_y = torch.full_like(delta_y, 0.30)
            sigma_x = torch.full_like(delta_x, 0.30)
        target = torch.exp(-0.5 * (delta_x.square() / sigma_x.square() + delta_y.square() / sigma_y.square()))
        indices = self.geometry.metric_to_index(xy_m)
        target[torch.arange(xy_m.shape[0], device=xy_m.device), indices[:, 0], indices[:, 1]] = 1.0
        return target.unsqueeze(1)

    def _softargmax_xy(self, logits: Tensor) -> Tensor:
        # BF16 is useful for the backbone, but coordinate normalization and a
        # 46,800-way softmax must retain FP32 headroom when logits saturate.
        stable_logits = logits.float()
        grid = self.metric_grid.to(device=logits.device, dtype=torch.float32).reshape(-1, 2)
        probability = torch.softmax(stable_logits.flatten(start_dim=1) / self.softargmax_temperature, dim=-1)
        return probability @ grid

    def forward(
        self,
        logits: Tensor,
        xy_m: Tensor,
        sample_weight: Tensor,
        *,
        teacher_logits: Tensor | None = None,
        teacher_peak_xy_m: Tensor | None = None,
    ) -> BEVLossOutput:
        if logits.ndim != 4 or logits.shape[1:] != (1, self.geometry.x_bins, self.geometry.y_bins):
            raise ValueError("logits must be [batch,1,200,234]")
        # Autocast BF16 rounds 1 - 1e-5 back to 1.0. The former probability-
        # space focal expression could therefore form 0 * log(0) after a
        # confident prediction. Compute both log-probabilities in FP32 logits
        # space, where softplus is finite for arbitrarily large finite logits.
        stable_logits = logits.float()
        stable_xy = xy_m.float()
        stable_weight = sample_weight.float()
        target = self.build_target(stable_xy, stable_weight)
        probability = stable_logits.sigmoid()
        effective_weight = stable_weight if self.variant.startswith("quality_aware_") else torch.ones_like(stable_weight)
        if self.variant == "bce_fixed_gaussian":
            heatmap_per_sample = functional.binary_cross_entropy_with_logits(
                stable_logits, target, reduction="none"
            ).flatten(start_dim=1).mean(dim=1)
        else:
            positive = target.eq(1.0)
            negative_weight = (1.0 - target).pow(self.beta)
            log_probability = -functional.softplus(-stable_logits)
            log_inverse_probability = -functional.softplus(stable_logits)
            positive_focal = -(1.0 - probability).pow(self.alpha) * log_probability
            negative_focal = -negative_weight * probability.pow(self.alpha) * log_inverse_probability
            focal = torch.where(positive, positive_focal, negative_focal)
            heatmap_per_sample = focal.flatten(start_dim=1).sum(dim=1)
        weighted_heatmap = (heatmap_per_sample * effective_weight).sum() / effective_weight.sum().clamp_min(1e-6)

        if self.variant == "quality_aware_focal_coordinate":
            predicted_xy = self._softargmax_xy(stable_logits[:, 0])
            coordinate_per_sample = functional.smooth_l1_loss(predicted_xy, stable_xy, reduction="none").mean(dim=-1)
            coordinate = (coordinate_per_sample * effective_weight).sum() / effective_weight.sum().clamp_min(1e-6)
            total = weighted_heatmap + self.coordinate_weight * coordinate
        else:
            coordinate = torch.zeros((), device=stable_logits.device, dtype=stable_logits.dtype)
            total = weighted_heatmap
        return BEVLossOutput(
            total=total,
            heatmap=weighted_heatmap,
            coordinate=coordinate,
            target=target,
        )


class DistilledQualityAwareBEVLoss(QualityAwareBEVLoss):
    """Add frozen-teacher distribution and metric-peak supervision during training only."""

    def __init__(
        self,
        *args: object,
        teacher_temperature: float = 2.0,
        teacher_heatmap_weight: float = 0.10,
        teacher_peak_weight: float = 0.05,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if teacher_temperature <= 0.0:
            raise ValueError("teacher_temperature must be positive")
        self.teacher_temperature = teacher_temperature
        self.teacher_heatmap_weight = teacher_heatmap_weight
        self.teacher_peak_weight = teacher_peak_weight

    def forward(
        self,
        logits: Tensor,
        xy_m: Tensor,
        sample_weight: Tensor,
        *,
        teacher_logits: Tensor | None = None,
        teacher_peak_xy_m: Tensor | None = None,
    ) -> BEVLossOutput:
        base = super().forward(logits, xy_m, sample_weight)
        if teacher_logits is None or teacher_peak_xy_m is None:
            # Development evaluation intentionally stays label-only. Teacher
            # targets exist for Train frames only and never enter scoring.
            return base
        if teacher_logits.shape != logits.shape:
            raise ValueError("teacher logits must match student logit shape")
        if teacher_peak_xy_m.shape != xy_m.shape:
            raise ValueError("teacher peak coordinates must match target coordinates")
        temperature = self.teacher_temperature
        student_flat = logits.float().flatten(start_dim=1) / temperature
        teacher_flat = teacher_logits.detach().float().flatten(start_dim=1) / temperature
        teacher_heatmap = functional.kl_div(
            functional.log_softmax(student_flat, dim=-1),
            functional.softmax(teacher_flat, dim=-1),
            reduction="batchmean",
        ) * temperature * temperature
        student_peak = self._softargmax_xy(logits[:, 0])
        teacher_peak = functional.smooth_l1_loss(student_peak, teacher_peak_xy_m.detach().float())
        total = base.total + self.teacher_heatmap_weight * teacher_heatmap + self.teacher_peak_weight * teacher_peak
        return BEVLossOutput(total=total, heatmap=base.heatmap, coordinate=base.coordinate, target=base.target)



