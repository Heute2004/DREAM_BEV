"""Candidate-local, range-preserving RVFusion for the CRUW transfer protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .allscene import CRUWBEVGeometry
from .cruw_expert import _crop_bev, _crop_polar


class _LocalEncoder(nn.Module):
    def __init__(self, input_channels: int, output_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(32, output_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.project(self.body(value).flatten(1))


@dataclass(frozen=True)
class CRUWRVFusionOutput:
    xy_m: torch.Tensor
    candidate_xy_m: torch.Tensor
    candidate_logits: torch.Tensor
    posterior: torch.Tensor
    selected_index: torch.Tensor
    candidate_valid_mask: torch.Tensor | None = None


class CRUWRangePreservingRVFusion(nn.Module):
    """Use causal RGB bearing evidence to rerank bounded lateral radar candidates."""

    def __init__(
        self,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CRUWBEVGeometry,
        offsets_m: tuple[float, ...] = (0.0, -0.50, -0.30, -0.15, 0.15, 0.30, 0.50),
        visual_dim: int = 96,
    ) -> None:
        super().__init__()
        if not offsets_m or offsets_m[0] != 0.0:
            raise ValueError("candidate zero must be the exact expert no-op")
        self.geometry = geometry
        self.register_buffer("range_grid", torch.as_tensor(range_grid, dtype=torch.float32))
        self.register_buffer("angle_grid", torch.as_tensor(angle_grid, dtype=torch.float32))
        self.register_buffer("offsets_m", torch.tensor(offsets_m, dtype=torch.float32))
        self.bev_encoder = _LocalEncoder(128, 32)
        self.polar_encoder = _LocalEncoder(48, 24)
        self.raw_encoder = _LocalEncoder(2, 16)
        self.visual_project = nn.Sequential(nn.Linear(visual_dim, 32), nn.SiLU(), nn.LayerNorm(32))
        self.token_project = nn.Sequential(nn.Linear(32 + 24 + 16 + 32 + 7, 96), nn.SiLU(), nn.LayerNorm(96))
        self.attention = nn.MultiheadAttention(96, num_heads=4, batch_first=True)
        self.output_norm = nn.LayerNorm(96)
        self.selector = nn.Linear(96, 1)
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        self.noop_bias = nn.Parameter(torch.tensor(1e-3))
        self.register_buffer("nonidentity_bias", torch.tensor(0.0))

    def _candidates(self, expert_xy_m: torch.Tensor) -> torch.Tensor:
        offsets = self.offsets_m.to(device=expert_xy_m.device, dtype=expert_xy_m.dtype)
        shifted_x = expert_xy_m[:, :1] + offsets[None]
        shifted_x = shifted_x.clamp(self.geometry.x_min_m, self.geometry.x_max_m - self.geometry.resolution_m)
        shifted_x[:, 0] = expert_xy_m[:, 0]
        fixed_y = expert_xy_m[:, 1:2].expand_as(shifted_x)
        return torch.stack((shifted_x, fixed_y), dim=-1)

    def forward(
        self,
        *,
        expert_xy_m: torch.Tensor,
        base_logits: torch.Tensor,
        shared_feature: torch.Tensor,
        polar_feature: torch.Tensor,
        radar: torch.Tensor,
        radar_score: torch.Tensor,
        visual_theta: torch.Tensor,
        visual_sigma: torch.Tensor,
        visual_reliability: torch.Tensor,
        visual_feature: torch.Tensor,
    ) -> CRUWRVFusionOutput:
        candidates = self._candidates(expert_xy_m)
        batch, count = candidates.shape[:2]
        bev_crop = _crop_bev(shared_feature, candidates, self.geometry)
        polar_crop = _crop_polar(polar_feature, candidates, self.range_grid, self.angle_grid)
        raw_crop = _crop_polar(radar[:, -1], candidates, self.range_grid, self.angle_grid)
        bev_embedding = self.bev_encoder(bev_crop.reshape(batch * count, *bev_crop.shape[2:])).reshape(batch, count, -1)
        polar_embedding = self.polar_encoder(polar_crop.reshape(batch * count, *polar_crop.shape[2:])).reshape(batch, count, -1)
        raw_embedding = self.raw_encoder(raw_crop.reshape(batch * count, *raw_crop.shape[2:])).reshape(batch, count, -1)
        visual_embedding = self.visual_project(visual_feature)[:, None].expand(-1, count, -1)

        local_logit = _crop_bev(base_logits, candidates, self.geometry, size=1).flatten(2).squeeze(-1)
        relative_logit = local_logit - local_logit[:, :1]
        candidate_theta = torch.atan2(candidates[..., 0], candidates[..., 1])
        theta = visual_theta.reshape(batch, 1)
        sigma = visual_sigma.reshape(batch, 1).clamp(0.01, 0.80)
        angular_error = torch.atan2(torch.sin(candidate_theta - theta), torch.cos(candidate_theta - theta)) / sigma
        max_offset = self.offsets_m.abs().max().clamp_min(1e-4)
        normalized_offset = (candidates[..., 0] - expert_xy_m[:, :1]) / max_offset
        reliability = visual_reliability.reshape(batch, 1).expand(-1, count)
        confidence = radar_score.reshape(batch, 1).expand(-1, count)
        sigma_feature = (sigma / 0.80).expand(-1, count)
        noop = candidates.new_zeros((batch, count))
        noop[:, 0] = 1.0
        scalars = torch.stack(
            (relative_logit, angular_error, normalized_offset, confidence, reliability, sigma_feature, noop), dim=-1
        )
        tokens = self.token_project(
            torch.cat((bev_embedding, polar_embedding, raw_embedding, visual_embedding, scalars), dim=-1)
        )
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.output_norm(tokens + attended)
        candidate_logits = self.selector(tokens).squeeze(-1)
        candidate_logits[:, 0] = candidate_logits[:, 0] + self.noop_bias
        candidate_logits[:, 1:] = candidate_logits[:, 1:] + self.nonidentity_bias
        posterior = candidate_logits.softmax(dim=1)
        selected_index = candidate_logits.argmax(dim=1)
        selected = candidates[torch.arange(batch, device=candidates.device), selected_index]
        return CRUWRVFusionOutput(
            selected,
            candidates,
            candidate_logits,
            posterior,
            selected_index,
        )


class CRUWXYRadarCandidateRVFusion(nn.Module):
    """Fuse local lateral offsets with globally supported radar XY candidates."""

    def __init__(
        self,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CRUWBEVGeometry,
        lateral_offsets_m: tuple[float, ...] = (0.0, -0.50, -0.30, -0.15, 0.15, 0.30, 0.50),
    ) -> None:
        super().__init__()
        if not lateral_offsets_m or lateral_offsets_m[0] != 0.0:
            raise ValueError("candidate zero must be the exact expert no-op")
        self.geometry = geometry
        self.register_buffer("range_grid", torch.as_tensor(range_grid, dtype=torch.float32))
        self.register_buffer("angle_grid", torch.as_tensor(angle_grid, dtype=torch.float32))
        self.register_buffer("lateral_offsets_m", torch.tensor(lateral_offsets_m, dtype=torch.float32))
        self.bev_encoder = _LocalEncoder(128, 32)
        self.polar_encoder = _LocalEncoder(48, 24)
        self.raw_encoder = _LocalEncoder(2, 16)
        self.visual_project = nn.Sequential(nn.Linear(96, 32), nn.SiLU(), nn.LayerNorm(32))
        self.token_project = nn.Sequential(nn.Linear(32 + 24 + 16 + 32 + 9, 96), nn.SiLU(), nn.LayerNorm(96))
        self.attention = nn.MultiheadAttention(96, num_heads=4, batch_first=True)
        self.output_norm = nn.LayerNorm(96)
        self.selector = nn.Linear(96, 1)
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)
        self.noop_bias = nn.Parameter(torch.tensor(1e-3))
        self.register_buffer("nonidentity_bias", torch.tensor(0.0))

    def _candidates(
        self,
        expert_xy_m: torch.Tensor,
        radar_candidate_xy_m: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if radar_candidate_xy_m.ndim != 3 or radar_candidate_xy_m.shape[0] != expert_xy_m.shape[0]:
            raise ValueError("radar candidates must have shape [batch,count,2]")
        offsets = self.lateral_offsets_m.to(device=expert_xy_m.device, dtype=expert_xy_m.dtype)
        shifted_x = expert_xy_m[:, :1] + offsets[None]
        shifted_x = shifted_x.clamp(self.geometry.x_min_m, self.geometry.x_max_m - self.geometry.resolution_m)
        shifted_x[:, 0] = expert_xy_m[:, 0]
        local = torch.stack((shifted_x, expert_xy_m[:, 1:2].expand_as(shifted_x)), dim=-1)
        radar_candidates = radar_candidate_xy_m.to(device=expert_xy_m.device, dtype=expert_xy_m.dtype).clone()
        radar_candidates[..., 0].clamp_(self.geometry.x_min_m, self.geometry.x_max_m - self.geometry.resolution_m)
        radar_candidates[..., 1].clamp_(self.geometry.y_min_m, self.geometry.y_max_m - self.geometry.resolution_m)
        candidates = torch.cat((local, radar_candidates), dim=1)

        valid = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
        duplicate_radius = 0.5 * self.geometry.resolution_m
        for index in range(1, candidates.shape[1]):
            prior_distance = torch.linalg.vector_norm(candidates[:, index : index + 1] - candidates[:, :index], dim=-1)
            valid[:, index] = prior_distance.amin(dim=1) > duplicate_radius
        valid[:, 0] = True
        return candidates, valid

    def forward(
        self,
        *,
        expert_xy_m: torch.Tensor,
        radar_candidate_xy_m: torch.Tensor,
        base_logits: torch.Tensor,
        shared_feature: torch.Tensor,
        polar_feature: torch.Tensor,
        radar: torch.Tensor,
        radar_score: torch.Tensor,
        visual_theta: torch.Tensor,
        visual_sigma: torch.Tensor,
        visual_reliability: torch.Tensor,
        visual_feature: torch.Tensor,
    ) -> CRUWRVFusionOutput:
        candidates, valid = self._candidates(expert_xy_m, radar_candidate_xy_m)
        batch, count = candidates.shape[:2]
        bev_crop = _crop_bev(shared_feature, candidates, self.geometry)
        polar_crop = _crop_polar(polar_feature, candidates, self.range_grid, self.angle_grid)
        raw_crop = _crop_polar(radar[:, -1], candidates, self.range_grid, self.angle_grid)
        bev_embedding = self.bev_encoder(bev_crop.reshape(batch * count, *bev_crop.shape[2:])).reshape(batch, count, -1)
        polar_embedding = self.polar_encoder(polar_crop.reshape(batch * count, *polar_crop.shape[2:])).reshape(batch, count, -1)
        raw_embedding = self.raw_encoder(raw_crop.reshape(batch * count, *raw_crop.shape[2:])).reshape(batch, count, -1)
        visual_embedding = self.visual_project(visual_feature)[:, None].expand(-1, count, -1)

        local_logit = _crop_bev(base_logits, candidates, self.geometry, size=1).flatten(2).squeeze(-1)
        relative_logit = local_logit - local_logit[:, :1]
        candidate_theta = torch.atan2(candidates[..., 0], candidates[..., 1].clamp_min(1e-4))
        theta = visual_theta.reshape(batch, 1)
        sigma = visual_sigma.reshape(batch, 1).clamp(0.01, 0.80)
        angular_error = torch.atan2(torch.sin(candidate_theta - theta), torch.cos(candidate_theta - theta)) / sigma
        normalized_delta = (candidates - expert_xy_m[:, None]) / candidates.new_tensor((10.0, 23.4))
        reliability = visual_reliability.reshape(batch, 1).expand(-1, count)
        confidence = radar_score.reshape(batch, 1).expand(-1, count)
        sigma_feature = (sigma / 0.80).expand(-1, count)
        noop = candidates.new_zeros((batch, count))
        noop[:, 0] = 1.0
        radar_owned = candidates.new_zeros((batch, count))
        radar_owned[:, self.lateral_offsets_m.numel() :] = 1.0
        scalars = torch.cat(
            (
                relative_logit[..., None],
                angular_error[..., None],
                normalized_delta,
                confidence[..., None],
                reliability[..., None],
                sigma_feature[..., None],
                noop[..., None],
                radar_owned[..., None],
            ),
            dim=-1,
        )
        tokens = self.token_project(
            torch.cat((bev_embedding, polar_embedding, raw_embedding, visual_embedding, scalars), dim=-1)
        )
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.output_norm(tokens + attended)
        candidate_logits = self.selector(tokens).squeeze(-1)
        candidate_logits[:, 0] = candidate_logits[:, 0] + self.noop_bias
        candidate_logits[:, 1:] = candidate_logits[:, 1:] + self.nonidentity_bias
        candidate_logits = candidate_logits.masked_fill(~valid, -1e4)
        posterior = candidate_logits.softmax(dim=1)
        selected_index = candidate_logits.argmax(dim=1)
        selected = candidates[torch.arange(batch, device=candidates.device), selected_index]
        return CRUWRVFusionOutput(selected, candidates, candidate_logits, posterior, selected_index, valid)


def lateral_utility_targets(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.05,
) -> torch.Tensor:
    """Supervise visual intervention only when a lateral candidate materially helps."""
    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != target_xy_m.shape[0]:
        raise ValueError("candidate and target batches must align")
    errors = (candidate_xy_m[..., 0] - target_xy_m[:, None, 0]).abs()
    best = errors.argmin(dim=1)
    improvement = errors[:, 0] - errors[torch.arange(errors.shape[0], device=errors.device), best]
    return torch.where(improvement >= minimum_improvement_m, best, torch.zeros_like(best))


def soft_lateral_utility_targets(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.05,
    sigma_m: float = 0.10,
) -> torch.Tensor:
    """Return posterior-compatible lateral targets with exact no-op fallback.

    A hard nearest-bin label conflicts with a posterior-mean readout whenever
    the target falls between adjacent lateral candidates. This loss-only target
    shares probability across non-identity candidates near the target while
    preserving candidate zero for frames without a material lateral gain.
    """

    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != target_xy_m.shape[0]:
        raise ValueError("candidate and target batches must align")
    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive")

    lateral_error = (candidate_xy_m[..., 0] - target_xy_m[:, None, 0]).abs()
    best = lateral_error.argmin(dim=1)
    rows = torch.arange(lateral_error.shape[0], device=lateral_error.device)
    improvement = lateral_error[:, 0] - lateral_error[rows, best]
    should_intervene = best.ne(0) & improvement.ge(minimum_improvement_m)

    nonidentity_squared_error = (candidate_xy_m[:, 1:, 0] - target_xy_m[:, None, 0]).square()
    logits = -nonidentity_squared_error / (2.0 * sigma_m * sigma_m)
    nonidentity_targets = logits.softmax(dim=1)
    targets = torch.zeros_like(lateral_error)
    targets[:, 0] = 1.0
    targets[:, 1:] = torch.where(
        should_intervene[:, None], nonidentity_targets, torch.zeros_like(nonidentity_targets)
    )
    targets[:, 0] = (~should_intervene).to(targets.dtype)
    return targets


def reachable_lateral_utility_targets(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.05,
    success_radius_m: float = 0.30,
) -> torch.Tensor:
    """Intervene only when a lateral candidate can achieve the target radius.

    Every candidate inherits the frozen expert range. Frames dominated by an
    unrecoverable range error are therefore supervised as exact no-op rather
    than teaching the lateral selector a correction that cannot improve the
    stated localization objective.
    """

    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != target_xy_m.shape[0]:
        raise ValueError("candidate and target batches must align")
    if success_radius_m <= 0.0:
        raise ValueError("success_radius_m must be positive")
    errors = torch.linalg.vector_norm(candidate_xy_m - target_xy_m[:, None], dim=-1)
    best = errors.argmin(dim=1)
    rows = torch.arange(errors.shape[0], device=errors.device)
    improvement = errors[:, 0] - errors[rows, best]
    reachable = errors[rows, best] <= success_radius_m
    should_intervene = best.ne(0) & reachable & improvement.ge(minimum_improvement_m)
    return torch.where(should_intervene, best, torch.zeros_like(best))


def ap30_transition_utility_targets(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.0,
    success_radius_m: float = 0.30,
) -> torch.Tensor:
    """Target only candidate changes that convert an AP@0.30 miss to a hit."""

    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != target_xy_m.shape[0]:
        raise ValueError("candidate and target batches must align")
    if success_radius_m <= 0.0:
        raise ValueError("success_radius_m must be positive")
    errors = torch.linalg.vector_norm(candidate_xy_m - target_xy_m[:, None], dim=-1)
    best = errors.argmin(dim=1)
    rows = torch.arange(errors.shape[0], device=errors.device)
    improvement = errors[:, 0] - errors[rows, best]
    should_intervene = (
        errors[:, 0].gt(success_radius_m)
        & best.ne(0)
        & errors[rows, best].le(success_radius_m)
        & improvement.ge(minimum_improvement_m)
    )
    return torch.where(should_intervene, best, torch.zeros_like(best))


def ap30_recoverable_transition_mask(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    success_radius_m: float = 0.30,
) -> torch.Tensor:
    """Mark frames an existing non-zero candidate can repair at AP@0.30."""

    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != target_xy_m.shape[0]:
        raise ValueError("candidate and target batches must align")
    if success_radius_m <= 0.0:
        raise ValueError("success_radius_m must be positive")
    errors = torch.linalg.vector_norm(candidate_xy_m - target_xy_m[:, None], dim=-1)
    return errors[:, 0].gt(success_radius_m) & errors[:, 1:].amin(dim=1).le(success_radius_m)


def ap30_transition_soft_targets(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.0,
    success_radius_m: float = 0.30,
    sigma_m: float = 0.10,
) -> torch.Tensor:
    """Posterior-compatible supervision for AP@0.30 miss-to-hit transitions."""

    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive")
    labels = ap30_transition_utility_targets(
        candidate_xy_m,
        target_xy_m,
        minimum_improvement_m=minimum_improvement_m,
        success_radius_m=success_radius_m,
    )
    errors = torch.linalg.vector_norm(candidate_xy_m - target_xy_m[:, None], dim=-1)
    nonidentity_errors = errors[:, 1:]
    nonidentity_logits = -nonidentity_errors.square() / (2.0 * sigma_m * sigma_m)
    valid_success = nonidentity_errors.le(success_radius_m)
    nonidentity_logits = nonidentity_logits.masked_fill(~valid_success, -1e4)
    nonidentity_targets = nonidentity_logits.softmax(dim=1)
    should_intervene = labels.ne(0)
    targets = torch.zeros_like(errors)
    targets[:, 0] = (~should_intervene).to(targets.dtype)
    targets[:, 1:] = torch.where(
        should_intervene[:, None], nonidentity_targets, torch.zeros_like(nonidentity_targets)
    )
    return targets


def xy_utility_targets(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.05,
    candidate_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose a radar-supported XY candidate only when total error improves."""
    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != target_xy_m.shape[0]:
        raise ValueError("candidate and target batches must align")
    errors = torch.linalg.vector_norm(candidate_xy_m - target_xy_m[:, None], dim=-1)
    if candidate_valid_mask is not None:
        if candidate_valid_mask.shape != errors.shape:
            raise ValueError("candidate validity mask must align with candidates")
        errors = errors.masked_fill(~candidate_valid_mask, torch.inf)
    best = errors.argmin(dim=1)
    rows = torch.arange(errors.shape[0], device=errors.device)
    improvement = errors[:, 0] - errors[rows, best]
    return torch.where(improvement >= minimum_improvement_m, best, torch.zeros_like(best))
