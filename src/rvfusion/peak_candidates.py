from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .local_candidates import extract_local_candidates
from .local_lattice import LocalXLattice


SOURCE_NOOP = 0
SOURCE_PEAK = 1
SOURCE_VISUAL = 2
SOURCE_DIRECT = 3


@dataclass(frozen=True)
class PeakCandidateSet:
    x_m: Tensor
    y_m: Tensor
    valid_mask: Tensor
    source_id: Tensor
    response: Tensor
    prominence: Tensor
    no_op_index: int = 0

    def validate(self) -> None:
        batch, count = self.x_m.shape
        for name, value in (
            ("y_m", self.y_m),
            ("valid_mask", self.valid_mask),
            ("source_id", self.source_id),
            ("response", self.response),
            ("prominence", self.prominence),
        ):
            if tuple(value.shape) != (batch, count):
                raise ValueError(f"{name} must match x_m")
        if self.no_op_index != 0 or self.valid_mask.dtype != torch.bool:
            raise ValueError("no-op must be valid index zero")
        if self.source_id.dtype != torch.long:
            raise ValueError("source_id must be int64")
        if not self.valid_mask[:, 0].all() or not torch.equal(
            self.source_id[:, 0],
            torch.zeros(batch, dtype=torch.long, device=self.source_id.device),
        ):
            raise ValueError("no-op candidate is invalid")
        for name, value in (
            ("x_m", self.x_m),
            ("y_m", self.y_m),
            ("response", self.response),
            ("prominence", self.prominence),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")


def _sample_response(radar_probability: Tensor, x_m: Tensor, y_m: Tensor) -> Tensor:
    batch = radar_probability.shape[0]
    row = torch.arange(batch, device=radar_probability.device)[:, None]
    x_index = ((x_m + 10.0) / 0.1).floor().long().clamp(0, 199)
    y_index = (y_m / 0.1).floor().long().clamp(0, 233)
    return radar_probability[row, 0, x_index, y_index]


def _deduplicate(x_m: Tensor, valid_mask: Tensor) -> Tensor:
    deduplicated = valid_mask.clone()
    for batch_index in range(x_m.shape[0]):
        accepted: list[Tensor] = []
        for candidate_index in range(x_m.shape[1]):
            if not bool(deduplicated[batch_index, candidate_index]):
                continue
            value = x_m[batch_index, candidate_index]
            if any(bool((value - prior).abs() <= 1e-6) for prior in accepted):
                deduplicated[batch_index, candidate_index] = False
                continue
            accepted.append(value)
    return deduplicated


def build_peak_candidate_set(
    *,
    radar_probability: Tensor,
    radar_xy_m: Tensor,
    visual_x_m: Tensor,
    direct_x_m: Tensor,
    radius_m: float = 0.5,
    top_k: int = 4,
) -> PeakCandidateSet:
    batch = radar_probability.shape[0]
    if tuple(radar_xy_m.shape) != (batch, 2):
        raise ValueError("radar_xy_m must be [batch,2]")
    for name, value in (("visual_x_m", visual_x_m), ("direct_x_m", direct_x_m)):
        if tuple(value.shape) != (batch, 1):
            raise ValueError(f"{name} must be [batch,1]")
    local = extract_local_candidates(
        radar_probability, radar_xy_m, visual_x_m, radius_m=radius_m, top_k=top_k
    )
    minimum = radar_xy_m[:, 0:1] - float(radius_m)
    maximum = radar_xy_m[:, 0:1] + float(radius_m)
    visual = visual_x_m.to(radar_probability.dtype).clamp(minimum, maximum)
    direct = direct_x_m.to(radar_probability.dtype).clamp(minimum, maximum)
    x_m = torch.cat((local.x_m[:, : top_k + 1], visual, direct), dim=1)
    valid_mask = torch.cat(
        (
            local.valid_mask[:, : top_k + 1],
            torch.ones((batch, 2), device=radar_probability.device, dtype=torch.bool),
        ),
        dim=1,
    )
    source_id = torch.cat(
        (
            torch.full((batch, 1), SOURCE_NOOP, device=radar_probability.device, dtype=torch.long),
            torch.full((batch, top_k), SOURCE_PEAK, device=radar_probability.device, dtype=torch.long),
            torch.full((batch, 1), SOURCE_VISUAL, device=radar_probability.device, dtype=torch.long),
            torch.full((batch, 1), SOURCE_DIRECT, device=radar_probability.device, dtype=torch.long),
        ),
        dim=1,
    )
    y_m = radar_xy_m[:, 1:2].to(radar_probability.dtype).expand_as(x_m)
    response = _sample_response(radar_probability, x_m, y_m)
    prominence = torch.cat(
        (
            local.prominence[:, : top_k + 1],
            torch.zeros((batch, 2), device=radar_probability.device, dtype=radar_probability.dtype),
        ),
        dim=1,
    )
    result = PeakCandidateSet(
        x_m=x_m,
        y_m=y_m,
        valid_mask=_deduplicate(x_m, valid_mask),
        source_id=source_id,
        response=response,
        prominence=prominence,
    )
    result.validate()
    return result


def build_peak_x_lattice(
    *,
    radar_probability: Tensor,
    radar_xy_m: Tensor,
    visual_x_m: Tensor,
    direct_x_m: Tensor,
    visual_sigma_x_m: Tensor,
    radius_m: float = 0.5,
    top_k: int = 4,
) -> LocalXLattice:
    peak = build_peak_candidate_set(
        radar_probability=radar_probability,
        radar_xy_m=radar_xy_m,
        visual_x_m=visual_x_m,
        direct_x_m=direct_x_m,
        radius_m=radius_m,
        top_k=top_k,
    )
    if tuple(visual_sigma_x_m.shape) != (radar_xy_m.shape[0], 1):
        raise ValueError("visual_sigma_x_m must be [batch,1]")
    sigma = visual_sigma_x_m.to(peak.x_m.dtype).clamp_min(0.025)
    visual_log_likelihood = -0.5 * ((peak.x_m - visual_x_m.to(peak.x_m.dtype)) / sigma).square()
    visual_log_likelihood = torch.where(
        peak.valid_mask, visual_log_likelihood, torch.full_like(visual_log_likelihood, -1e4)
    )
    result = LocalXLattice(
        x_m=peak.x_m,
        relative_x_m=peak.x_m - radar_xy_m[:, 0:1].to(peak.x_m.dtype),
        valid_mask=peak.valid_mask,
        source_id=peak.source_id,
        radar_support=peak.response,
        visual_log_likelihood=visual_log_likelihood,
    )
    result.validate()
    return result
