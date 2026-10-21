from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LocalCandidateSet:
    x_m: Tensor
    valid_mask: Tensor
    source_id: Tensor
    response: Tensor
    prominence: Tensor
    local_profile: Tensor
    local_profile_mask: Tensor
    no_op_index: int = 0

    def validate(self) -> None:
        batch, candidate_count = self.x_m.shape
        expected_candidates = (batch, candidate_count)
        for name, value in (
            ("valid_mask", self.valid_mask),
            ("source_id", self.source_id),
            ("response", self.response),
            ("prominence", self.prominence),
        ):
            if tuple(value.shape) != expected_candidates:
                raise ValueError(f"{name} must match x_m")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean")
        if self.source_id.dtype != torch.long:
            raise ValueError("source_id must be int64")
        if self.local_profile.ndim != 2 or self.local_profile.shape[0] != batch:
            raise ValueError("local_profile must be [batch,local_bins]")
        if tuple(self.local_profile_mask.shape) != tuple(self.local_profile.shape):
            raise ValueError("local_profile_mask must match local_profile")
        if self.local_profile_mask.dtype != torch.bool:
            raise ValueError("local_profile_mask must be boolean")
        if self.no_op_index != 0:
            raise ValueError("no-op candidate must occupy index zero")
        if not self.valid_mask[:, self.no_op_index].all():
            raise ValueError("no-op candidate must always be valid")
        for name, value in (
            ("x_m", self.x_m),
            ("response", self.response),
            ("prominence", self.prominence),
            ("local_profile", self.local_profile),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")


def _metric_to_index(x_m: Tensor, bins: int) -> Tensor:
    return ((x_m + 10.0) / 0.1).floor().long().clamp(0, bins - 1)


def extract_local_candidates(
    radar_probability: Tensor,
    radar_xy_m: Tensor,
    visual_x_m: Tensor,
    *,
    radius_m: float = 0.5,
    top_k: int = 4,
    visual_support_radius_m: float = 0.15,
) -> LocalCandidateSet:
    if radar_probability.ndim != 4 or radar_probability.shape[1:] != (
        1,
        200,
        234,
    ):
        raise ValueError("radar_probability must be [batch,1,200,234]")
    batch = radar_probability.shape[0]
    if tuple(radar_xy_m.shape) != (batch, 2):
        raise ValueError("radar_xy_m must be [batch,2]")
    if tuple(visual_x_m.shape) != (batch, 1):
        raise ValueError("visual_x_m must be [batch,1]")
    if radius_m <= 0.0 or top_k < 1:
        raise ValueError("radius_m must be positive and top_k at least one")
    if visual_support_radius_m <= 0.0:
        raise ValueError("visual_support_radius_m must be positive")
    radius_bins = int(round(float(radius_m) / 0.1))
    if radius_bins < 1 or abs(radius_bins * 0.1 - float(radius_m)) > 1e-6:
        raise ValueError("radius_m must be a positive multiple of 0.1 m")

    device = radar_probability.device
    dtype = radar_probability.dtype
    x_bins = radar_probability.shape[2]
    y_bins = radar_probability.shape[3]
    row_index = torch.arange(batch, device=device)
    radar_x_index = _metric_to_index(radar_xy_m[:, 0], x_bins)
    radar_y_index = (radar_xy_m[:, 1] / 0.1).floor().long().clamp(0, y_bins - 1)
    x_profile = radar_probability[
        row_index,
        0,
        :,
        radar_y_index,
    ]
    x_profile = x_profile / x_profile.sum(dim=1, keepdim=True).clamp_min(1e-12)

    offsets = torch.arange(
        -radius_bins,
        radius_bins + 1,
        device=device,
    )
    raw_local_index = radar_x_index[:, None] + offsets[None]
    local_profile_mask = (raw_local_index >= 0) & (raw_local_index < x_bins)
    local_index = raw_local_index.clamp(0, x_bins - 1)
    local_profile = x_profile.gather(1, local_index)
    local_profile = torch.where(
        local_profile_mask,
        local_profile,
        torch.zeros_like(local_profile),
    )

    left = torch.roll(local_profile, shifts=1, dims=1)
    right = torch.roll(local_profile, shifts=-1, dims=1)
    internal = torch.ones_like(local_profile_mask)
    internal[:, 0] = False
    internal[:, -1] = False
    strict_peak = (
        (local_profile >= left)
        & (local_profile >= right)
        & ((local_profile > left) | (local_profile > right))
    )
    center = offsets == 0
    peak_mask = (
        strict_peak
        & internal
        & local_profile_mask
        & ~center[None]
    )
    peak_score = torch.where(
        peak_mask,
        local_profile,
        torch.full_like(local_profile, -torch.inf),
    )
    order = torch.argsort(
        peak_score,
        dim=1,
        descending=True,
        stable=True,
    )[:, :top_k]
    radar_candidate_valid = peak_mask.gather(1, order)
    radar_candidate_index = local_index.gather(1, order)
    radar_candidate_x = -10.0 + (
        radar_candidate_index.to(dtype) + 0.5
    ) * 0.1
    radar_candidate_x = torch.where(
        radar_candidate_valid,
        radar_candidate_x,
        radar_xy_m[:, 0:1].expand(-1, top_k),
    )
    radar_response = local_profile.gather(1, order)
    radar_response = torch.where(
        radar_candidate_valid,
        radar_response,
        torch.zeros_like(radar_response),
    )
    local_prominence = (
        local_profile - torch.maximum(left, right)
    ).clamp_min(0.0)
    radar_prominence = local_prominence.gather(1, order)
    radar_prominence = torch.where(
        radar_candidate_valid,
        radar_prominence,
        torch.zeros_like(radar_prominence),
    )

    minimum_x = (radar_xy_m[:, 0:1] - radius_m).clamp_min(-10.0)
    maximum_x = (radar_xy_m[:, 0:1] + radius_m).clamp_max(9.9)
    clipped_visual_x = torch.minimum(
        torch.maximum(visual_x_m.to(dtype), minimum_x),
        maximum_x,
    )
    local_x_m = -10.0 + local_index.to(dtype) * 0.1
    visual_support_mask = (
        strict_peak
        & internal
        & local_profile_mask
        & (
            (local_x_m - clipped_visual_x).abs()
            <= float(visual_support_radius_m) + 1e-6
        )
    )
    visual_score = torch.where(
        visual_support_mask,
        local_profile,
        torch.full_like(local_profile, -torch.inf),
    )
    visual_local_position = visual_score.argmax(dim=1, keepdim=True)
    visual_candidate_valid = visual_support_mask.gather(
        1,
        visual_local_position,
    )
    visual_candidate_index = local_index.gather(1, visual_local_position)
    visual_candidate_x = -10.0 + (
        visual_candidate_index.to(dtype) + 0.5
    ) * 0.1
    visual_candidate_x = torch.where(
        visual_candidate_valid,
        visual_candidate_x,
        radar_xy_m[:, 0:1].to(dtype),
    )
    visual_response = local_profile.gather(1, visual_local_position)
    visual_response = torch.where(
        visual_candidate_valid,
        visual_response,
        torch.zeros_like(visual_response),
    )
    visual_prominence = local_prominence.gather(1, visual_local_position)
    visual_prominence = torch.where(
        visual_candidate_valid,
        visual_prominence,
        torch.zeros_like(visual_prominence),
    )

    no_op_response = x_profile.gather(1, radar_x_index[:, None])
    no_op_local_position = torch.full(
        (batch, 1),
        radius_bins,
        device=device,
        dtype=torch.long,
    )
    no_op_prominence = local_prominence.gather(1, no_op_local_position)
    x_m = torch.cat(
        (
            radar_xy_m[:, 0:1].to(dtype),
            radar_candidate_x,
            visual_candidate_x,
        ),
        dim=1,
    )
    valid_mask = torch.cat(
        (
            torch.ones(batch, 1, device=device, dtype=torch.bool),
            radar_candidate_valid,
            visual_candidate_valid,
        ),
        dim=1,
    )
    source_id = torch.cat(
        (
            torch.zeros(batch, 1, device=device, dtype=torch.long),
            torch.ones(batch, top_k, device=device, dtype=torch.long),
            torch.full((batch, 1), 2, device=device, dtype=torch.long),
        ),
        dim=1,
    )
    response = torch.cat(
        (no_op_response, radar_response, visual_response),
        dim=1,
    )
    prominence = torch.cat(
        (no_op_prominence, radar_prominence, visual_prominence),
        dim=1,
    )
    result = LocalCandidateSet(
        x_m=x_m,
        valid_mask=valid_mask,
        source_id=source_id,
        response=response,
        prominence=prominence,
        local_profile=local_profile,
        local_profile_mask=local_profile_mask,
    )
    result.validate()
    return result



