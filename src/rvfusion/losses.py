from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from radar_bev.geometry import BEVGeometry
from radar_bev.losses import QualityAwareBEVLoss

from .contracts import FusionOutput


@dataclass(frozen=True)
class FusionLossWeights:
    bev: float = 1.0
    offset: float = 0.5
    regret: float = 0.5
    identity: float = 0.2
    temporal: float = 0.1
    gate: float = 0.1


@dataclass(frozen=True)
class FusionLossOutput:
    total: Tensor
    terms: dict[str, Tensor]


def _softargmax_xy(logits: Tensor, geometry: BEVGeometry, temperature: float = 0.25) -> Tensor:
    probability = torch.softmax(logits.float().flatten(start_dim=1) / temperature, dim=1)
    grid = geometry.coordinate_grid(logits.device).reshape(-1, 2)
    return probability @ grid


def compute_rvfusion_loss(
    output: FusionOutput,
    target_xy_m: Tensor,
    sample_weight: Tensor,
    unreliable_mask: Tensor,
    *,
    previous_fused_xy_m: Tensor | None = None,
    previous_target_xy_m: Tensor | None = None,
    weights: FusionLossWeights = FusionLossWeights(),
    geometry: BEVGeometry | None = None,
) -> FusionLossOutput:
    geometry = geometry or BEVGeometry()
    criterion = QualityAwareBEVLoss(
        geometry=geometry,
        variant="quality_aware_focal_coordinate",
        coordinate_weight=0.05,
    )
    bev_result = criterion(output.fused_logits, target_xy_m, sample_weight)
    fused_xy = _softargmax_xy(output.fused_logits[:, 0], geometry)
    with torch.no_grad():
        base_xy = _softargmax_xy(output.radar_logits[:, 0], geometry)
        base_error = (base_xy - target_xy_m).norm(dim=1)
    fused_error = (fused_xy - target_xy_m).norm(dim=1)
    offset = functional.smooth_l1_loss(fused_xy, target_xy_m, reduction="none").mean(dim=1)
    offset = (offset * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
    regret = torch.relu(fused_error - base_error).mean()

    if unreliable_mask.any():
        identity = output.delta_logits[unreliable_mask].abs().mean()
    else:
        identity = output.delta_logits.sum() * 0.0

    if previous_fused_xy_m is not None and previous_target_xy_m is not None:
        predicted_velocity = fused_xy - previous_fused_xy_m
        target_velocity = target_xy_m - previous_target_xy_m
        temporal = functional.huber_loss(predicted_velocity, target_velocity)
    else:
        temporal = output.fused_logits.sum() * 0.0

    utility_target = (fused_error.detach() < base_error).float().unsqueeze(1)
    acceptance_logit = torch.logit(output.acceptance.float().clamp(1e-6, 1.0 - 1e-6))
    gate = functional.binary_cross_entropy_with_logits(acceptance_logit, utility_target.float())
    terms = {
        "bev": bev_result.total,
        "offset": offset,
        "regret": regret,
        "identity": identity,
        "temporal": temporal,
        "gate": gate,
    }
    total = (
        weights.bev * terms["bev"]
        + weights.offset * terms["offset"]
        + weights.regret * terms["regret"]
        + weights.identity * terms["identity"]
        + weights.temporal * terms["temporal"]
        + weights.gate * terms["gate"]
    )
    return FusionLossOutput(total=total, terms=terms)



