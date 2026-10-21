"""Visual-guided association over globally extracted radar candidates."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .allscene import CRUWBEVGeometry
from .cruw_expert import (
    _CropEncoder,
    _crop_bev,
    _crop_polar,
    _decode_peak_tensor,
    _metric_candidates,
)


def metric_global_candidates(
    base_logits: torch.Tensor,
    previous_logits: torch.Tensor,
    geometry: CRUWBEVGeometry,
    *,
    count: int = 12,
) -> torch.Tensor:
    """Return current identity, global current peaks, and one causal prior peak."""

    if count < 3:
        raise ValueError("global candidate count must be at least three")
    if base_logits.shape != previous_logits.shape:
        raise ValueError("current and previous logits must have identical shape")
    base_xy = _decode_peak_tensor(base_logits, geometry)
    current = _metric_candidates(base_logits, geometry, count - 1)
    previous_xy = _decode_peak_tensor(previous_logits, geometry)
    # current[:, 0] is the same global maximum as base_xy. Keep one exact
    # identity candidate and use the remaining NMS peaks as alternatives.
    return torch.cat((base_xy[:, None], current[:, 1:], previous_xy[:, None]), dim=1)


class CRUWGlobalAssociationExpert(nn.Module):
    """Select one radar-owned metric candidate using radar and RGB bearing evidence."""

    candidate_count = 12

    def __init__(
        self,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        geometry: CRUWBEVGeometry,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.register_buffer("range_grid", torch.as_tensor(range_grid, dtype=torch.float32))
        self.register_buffer("angle_grid", torch.as_tensor(angle_grid, dtype=torch.float32))
        self.bev_encoder = _CropEncoder(128, 48)
        self.polar_encoder = _CropEncoder(48, 40)
        self.raw_encoder = _CropEncoder(2, 24)
        self.visual_project = nn.Sequential(
            nn.Linear(96, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
        )
        self.type_embedding = nn.Embedding(3, 8)
        self.token_project = nn.Sequential(
            nn.Linear(48 + 40 + 24 + 32 + 8 + 14, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.attention = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.selector = nn.Linear(128, 1)
        self.register_buffer("nonbase_bias", torch.tensor(0.0))
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)

    def forward(
        self,
        *,
        polar_feature: torch.Tensor,
        shared_feature: torch.Tensor,
        radar: torch.Tensor,
        base_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        visual_theta: torch.Tensor,
        visual_sigma: torch.Tensor,
        visual_reliability: torch.Tensor,
        visual_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidates = metric_global_candidates(
            base_logits,
            previous_logits,
            self.geometry,
            count=self.candidate_count,
        )
        base_xy = candidates[:, 0]
        previous_xy = candidates[:, -1]
        batch, count = candidates.shape[:2]
        if visual_feature.shape != (batch, 96):
            raise ValueError("visual_feature must have shape [batch,96]")

        types = torch.ones((batch, count), device=radar.device, dtype=torch.long)
        types[:, 0] = 0
        types[:, -1] = 2
        bev_crop = _crop_bev(shared_feature, candidates, self.geometry)
        polar_crop = _crop_polar(
            polar_feature,
            candidates,
            self.range_grid,
            self.angle_grid,
        )
        raw_crop = _crop_polar(
            radar[:, -1],
            candidates,
            self.range_grid,
            self.angle_grid,
        )
        bev_embedding = self.bev_encoder(
            bev_crop.reshape(batch * count, *bev_crop.shape[2:])
        ).reshape(batch, count, -1)
        polar_embedding = self.polar_encoder(
            polar_crop.reshape(batch * count, *polar_crop.shape[2:])
        ).reshape(batch, count, -1)
        raw_embedding = self.raw_encoder(
            raw_crop.reshape(batch * count, *raw_crop.shape[2:])
        ).reshape(batch, count, -1)

        candidate_range = torch.linalg.vector_norm(candidates, dim=-1)
        candidate_theta = torch.atan2(candidates[..., 0], candidates[..., 1].clamp_min(1e-4))
        angular_error = torch.atan2(
            torch.sin(candidate_theta - visual_theta[:, None]),
            torch.cos(candidate_theta - visual_theta[:, None]),
        )
        sigma = visual_sigma[:, None].clamp_min(math.radians(0.25))
        visual_match = torch.exp(-0.5 * (angular_error / sigma).square())
        reliability = visual_reliability[:, None].expand(-1, count)
        normalized_rank = torch.linspace(0.0, 1.0, count, device=radar.device, dtype=radar.dtype)[None].expand(batch, -1)
        geometry_features = torch.cat(
            (
                candidates / candidates.new_tensor((10.0, 23.4)),
                (candidates - base_xy[:, None]) / candidates.new_tensor((5.0, 10.0)),
                (candidates - previous_xy[:, None]) / candidates.new_tensor((5.0, 10.0)),
                (candidate_range / 25.5)[..., None],
                (candidate_theta / math.pi)[..., None],
                (angular_error / math.pi)[..., None],
                visual_match[..., None],
                reliability[..., None],
                (sigma / math.pi).expand(-1, count)[..., None],
                types.eq(0).to(radar.dtype)[..., None],
                normalized_rank[..., None],
            ),
            dim=-1,
        )
        visual_embedding = self.visual_project(visual_feature)[:, None].expand(-1, count, -1)
        tokens = self.token_project(
            torch.cat(
                (
                    bev_embedding,
                    polar_embedding,
                    raw_embedding,
                    visual_embedding,
                    self.type_embedding(types),
                    geometry_features,
                ),
                dim=-1,
            )
        )
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        selector_logits = self.selector(tokens).squeeze(-1)
        selector_logits = selector_logits + torch.cat(
            (
                selector_logits.new_zeros((batch, 1)),
                self.nonbase_bias.expand(batch, count - 1),
            ),
            dim=1,
        )
        selected_index = selector_logits.argmax(dim=1)
        proposal_xy = candidates[torch.arange(batch, device=radar.device), selected_index]
        return {
            "candidate_xy_m": candidates,
            "selector_logits": selector_logits,
            "selected_index": selected_index,
            "proposal_xy_m": proposal_xy,
            "base_xy_m": base_xy,
            "previous_xy_m": previous_xy,
            "candidate_visual_match": visual_match,
        }


def global_association_target_indices(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.05,
) -> torch.Tensor:
    """Choose a nonidentity candidate only when it materially improves error."""

    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != target_xy_m.shape[0]:
        raise ValueError("candidate and target batches must align")
    errors = torch.linalg.vector_norm(candidate_xy_m - target_xy_m[:, None], dim=-1)
    best = errors.argmin(dim=1)
    rows = torch.arange(errors.shape[0], device=errors.device)
    improvement = errors[:, 0] - errors[rows, best]
    return torch.where(
        improvement >= minimum_improvement_m,
        best,
        torch.zeros_like(best),
    )
