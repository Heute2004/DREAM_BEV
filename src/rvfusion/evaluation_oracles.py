from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BeneficialCandidateOracleOutput:
    decoded_xy_m: Tensor
    accepted: Tensor
    radar_error_m: Tensor
    candidate_error_m: Tensor


@dataclass(frozen=True)
class BeneficialCandidateSetOracleOutput:
    decoded_xy_m: Tensor
    accepted: Tensor
    selected_index: Tensor
    radar_error_m: Tensor
    candidate_error_m: Tensor


def apply_beneficial_candidate_oracle(
    radar_xy_m: Tensor,
    candidate_xy_m: Tensor,
    target_xy_m: Tensor,
    correction_allowed: Tensor,
    *,
    minimum_gain_m: float = 1e-6,
) -> BeneficialCandidateOracleOutput:
    batch = radar_xy_m.shape[0]
    for name, value in (
        ("radar_xy_m", radar_xy_m),
        ("candidate_xy_m", candidate_xy_m),
        ("target_xy_m", target_xy_m),
    ):
        if tuple(value.shape) != (batch, 2):
            raise ValueError(f"{name} must be [batch,2]")
    if tuple(correction_allowed.shape) != (batch,):
        raise ValueError("correction_allowed must be [batch]")
    if correction_allowed.dtype != torch.bool:
        raise ValueError("correction_allowed must be boolean")
    if minimum_gain_m < 0.0:
        raise ValueError("minimum_gain_m must be non-negative")
    if not torch.equal(candidate_xy_m[:, 1], radar_xy_m[:, 1]):
        raise ValueError("candidate y must equal radar y")

    radar_error_m = torch.linalg.vector_norm(
        radar_xy_m - target_xy_m,
        dim=1,
    )
    candidate_error_m = torch.linalg.vector_norm(
        candidate_xy_m - target_xy_m,
        dim=1,
    )
    accepted = correction_allowed & (
        candidate_error_m + float(minimum_gain_m) < radar_error_m
    )
    decoded_xy_m = torch.where(
        accepted[:, None],
        candidate_xy_m,
        radar_xy_m,
    )
    return BeneficialCandidateOracleOutput(
        decoded_xy_m=decoded_xy_m,
        accepted=accepted,
        radar_error_m=radar_error_m,
        candidate_error_m=candidate_error_m,
    )


def apply_beneficial_candidate_set_oracle(
    radar_xy_m: Tensor,
    candidate_xy_m: Tensor,
    target_xy_m: Tensor,
    correction_allowed: Tensor,
    *,
    minimum_gain_m: float = 1e-6,
) -> BeneficialCandidateSetOracleOutput:
    batch = radar_xy_m.shape[0]
    if tuple(radar_xy_m.shape) != (batch, 2):
        raise ValueError("radar_xy_m must be [batch,2]")
    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != batch:
        raise ValueError("candidate_xy_m must be [batch,candidates,2]")
    if candidate_xy_m.shape[2] != 2:
        raise ValueError("candidate_xy_m must end with x,y")
    if tuple(target_xy_m.shape) != (batch, 2):
        raise ValueError("target_xy_m must be [batch,2]")
    if tuple(correction_allowed.shape) != (batch,):
        raise ValueError("correction_allowed must be [batch]")
    if correction_allowed.dtype != torch.bool:
        raise ValueError("correction_allowed must be boolean")
    if minimum_gain_m < 0.0:
        raise ValueError("minimum_gain_m must be non-negative")
    expected_y = radar_xy_m[:, None, 1].expand_as(candidate_xy_m[:, :, 1])
    if not torch.equal(candidate_xy_m[:, :, 1], expected_y):
        raise ValueError("candidate y must equal radar y")

    radar_error_m = torch.linalg.vector_norm(
        radar_xy_m - target_xy_m,
        dim=1,
    )
    all_candidate_error_m = torch.linalg.vector_norm(
        candidate_xy_m - target_xy_m[:, None],
        dim=2,
    )
    candidate_error_m, selected_index = all_candidate_error_m.min(dim=1)
    row_index = torch.arange(batch, device=radar_xy_m.device)
    selected_xy_m = candidate_xy_m[row_index, selected_index]
    accepted = correction_allowed & (
        candidate_error_m + float(minimum_gain_m) < radar_error_m
    )
    decoded_xy_m = torch.where(
        accepted[:, None],
        selected_xy_m,
        radar_xy_m,
    )
    return BeneficialCandidateSetOracleOutput(
        decoded_xy_m=decoded_xy_m,
        accepted=accepted,
        selected_index=selected_index,
        radar_error_m=radar_error_m,
        candidate_error_m=candidate_error_m,
    )



