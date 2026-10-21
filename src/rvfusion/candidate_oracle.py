from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class CandidateCoordinates(Protocol):
    x_m: Tensor
    valid_mask: Tensor
    no_op_index: int

    def validate(self) -> None: ...


@dataclass(frozen=True)
class CandidateOracleOutput:
    radar_xy_m: Tensor
    decoded_xy_m: Tensor
    best_index: Tensor
    candidate_error_m: Tensor
    no_op_error_m: Tensor

    def validate(self) -> None:
        batch = self.radar_xy_m.shape[0]
        if tuple(self.radar_xy_m.shape) != (batch, 2):
            raise ValueError("radar_xy_m must be [batch,2]")
        if tuple(self.decoded_xy_m.shape) != (batch, 2):
            raise ValueError("decoded_xy_m must be [batch,2]")
        if tuple(self.best_index.shape) != (batch,):
            raise ValueError("best_index must be [batch]")
        for name, value in (
            ("candidate_error_m", self.candidate_error_m),
            ("no_op_error_m", self.no_op_error_m),
        ):
            if tuple(value.shape) != (batch,):
                raise ValueError(f"{name} must be [batch]")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if not torch.equal(self.decoded_xy_m[:, 1], self.radar_xy_m[:, 1]):
            raise ValueError("candidate oracle must copy radar y exactly")


def select_candidate_oracle(
    candidates: CandidateCoordinates,
    radar_xy_m: Tensor,
    target_xy_m: Tensor,
    correction_allowed: Tensor,
) -> CandidateOracleOutput:
    candidates.validate()
    batch, candidate_count = candidates.x_m.shape
    if tuple(radar_xy_m.shape) != (batch, 2):
        raise ValueError("radar_xy_m must be [batch,2]")
    if tuple(target_xy_m.shape) != (batch, 2):
        raise ValueError("target_xy_m must be [batch,2]")
    if tuple(correction_allowed.shape) != (batch,):
        raise ValueError("correction_allowed must be [batch]")
    if correction_allowed.dtype != torch.bool:
        raise ValueError("correction_allowed must be boolean")

    candidate_y = radar_xy_m[:, 1:2].expand(-1, candidate_count)
    target_x = target_xy_m[:, 0:1]
    target_y = target_xy_m[:, 1:2]
    error = torch.sqrt(
        (candidates.x_m - target_x).square()
        + (candidate_y - target_y).square()
    )
    allowed_mask = candidates.valid_mask.clone()
    allowed_mask[:, 1:] &= correction_allowed[:, None]
    masked_error = error.masked_fill(~allowed_mask, torch.inf)
    best_index = masked_error.argmin(dim=1)
    selected_x = candidates.x_m.gather(1, best_index[:, None])[:, 0]
    decoded_xy_m = torch.stack((selected_x, radar_xy_m[:, 1]), dim=1)
    row_index = torch.arange(batch, device=best_index.device)
    output = CandidateOracleOutput(
        radar_xy_m=radar_xy_m,
        decoded_xy_m=decoded_xy_m,
        best_index=best_index,
        candidate_error_m=error[row_index, best_index],
        no_op_error_m=error[:, candidates.no_op_index],
    )
    output.validate()
    return output



