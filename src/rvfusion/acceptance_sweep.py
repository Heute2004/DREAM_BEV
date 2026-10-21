from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AcceptanceThresholdOutput:
    score: Tensor
    prediction_xy_m: Tensor
    accepted: Tensor


def apply_acceptance_threshold(
    *,
    radar_xy_m: Tensor,
    candidate_xy_m: Tensor,
    radar_score: Tensor,
    candidate_score: Tensor,
    effective_acceptance: Tensor,
    identity_mask: Tensor,
    selected_non_noop: Tensor,
    threshold: float,
) -> AcceptanceThresholdOutput:
    batch = radar_xy_m.shape[0]
    if tuple(radar_xy_m.shape) != (batch, 2):
        raise ValueError("radar_xy_m must be [batch,2]")
    if tuple(candidate_xy_m.shape) != (batch, 2):
        raise ValueError("candidate_xy_m must be [batch,2]")
    for name, value in (
        ("radar_score", radar_score),
        ("candidate_score", candidate_score),
        ("effective_acceptance", effective_acceptance),
    ):
        if tuple(value.shape) != (batch,):
            raise ValueError(f"{name} must be [batch]")
    for name, value in (
        ("identity_mask", identity_mask),
        ("selected_non_noop", selected_non_noop),
    ):
        if tuple(value.shape) != (batch,) or value.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean [batch]")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")

    accepted = (
        ~identity_mask
        & selected_non_noop
        & (effective_acceptance >= float(threshold))
    )
    prediction_xy_m = torch.where(
        accepted[:, None],
        candidate_xy_m,
        radar_xy_m,
    )
    score = torch.where(accepted, candidate_score, radar_score)
    return AcceptanceThresholdOutput(
        score=score,
        prediction_xy_m=prediction_xy_m,
        accepted=accepted,
    )



