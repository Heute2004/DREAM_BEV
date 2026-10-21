"""CRUW PB-MHBR-style Polar-BEV candidate reranking expert."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .allscene import CRUWBEVGeometry
from .metrics import decode_bev_logits


def _metric_candidates(base_logits: torch.Tensor, geometry: CRUWBEVGeometry, count: int) -> torch.Tensor:
    confidence = torch.sigmoid(base_logits)
    maxima = confidence.eq(F.max_pool2d(confidence, kernel_size=9, stride=1, padding=4))
    suppressed = confidence * maxima
    flat = suppressed.flatten(1)
    indices = flat.topk(count, dim=1).indices
    x_index = indices // geometry.y_bins
    y_index = indices % geometry.y_bins
    x_m = geometry.x_min_m + x_index.to(base_logits.dtype) * geometry.resolution_m
    y_m = geometry.y_min_m + y_index.to(base_logits.dtype) * geometry.resolution_m
    return torch.stack((x_m, y_m), dim=-1)


def _decode_peak_tensor(logits: torch.Tensor, geometry: CRUWBEVGeometry) -> torch.Tensor:
    peak = logits.flatten(1).argmax(dim=1)
    x_index = peak // geometry.y_bins
    y_index = peak % geometry.y_bins
    return torch.stack(
        (
            geometry.x_min_m + x_index.to(logits.dtype) * geometry.resolution_m,
            geometry.y_min_m + y_index.to(logits.dtype) * geometry.resolution_m,
        ),
        dim=1,
    )


def _crop_bev(feature: torch.Tensor, xy_m: torch.Tensor, geometry: CRUWBEVGeometry, size: int = 7) -> torch.Tensor:
    batch, candidates = xy_m.shape[:2]
    x_norm = 2.0 * (xy_m[..., 0] - geometry.x_min_m) / (geometry.x_max_m - geometry.x_min_m) - 1.0
    y_norm = 2.0 * (xy_m[..., 1] - geometry.y_min_m) / (geometry.y_max_m - geometry.y_min_m) - 1.0
    x_step = 2.0 * 1.0 / max(feature.shape[2] - 1, 1)
    y_step = 2.0 * 1.0 / max(feature.shape[3] - 1, 1)
    offsets_x = torch.linspace(-(size // 2), size // 2, size, device=feature.device, dtype=feature.dtype) * x_step
    offsets_y = torch.linspace(-(size // 2), size // 2, size, device=feature.device, dtype=feature.dtype) * y_step
    grid_x, grid_y = torch.meshgrid(offsets_x, offsets_y, indexing="ij")
    # grid_sample coordinates are ordered [width=y, height=x].
    grid = torch.stack(
        (
            y_norm[:, :, None, None] + grid_y,
            x_norm[:, :, None, None] + grid_x,
        ),
        dim=-1,
    ).reshape(batch * candidates, size, size, 2)
    repeated = feature.repeat_interleave(candidates, dim=0)
    crop = F.grid_sample(repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return crop.reshape(batch, candidates, feature.shape[1], size, size)


def _crop_polar(
    feature: torch.Tensor,
    xy_m: torch.Tensor,
    range_grid: torch.Tensor,
    angle_grid: torch.Tensor,
    size: int = 7,
) -> torch.Tensor:
    batch, candidates = xy_m.shape[:2]
    ranges = torch.linalg.vector_norm(xy_m, dim=-1)
    angles = torch.atan2(xy_m[..., 0], xy_m[..., 1])
    range_index = torch.searchsorted(range_grid, ranges).clamp(1, 127)
    angle_index = torch.searchsorted(angle_grid, angles).clamp(1, 127)
    range_norm = 2.0 * range_index.to(feature.dtype) / 127.0 - 1.0
    angle_norm = 2.0 * angle_index.to(feature.dtype) / 127.0 - 1.0
    offsets = torch.linspace(-(size // 2), size // 2, size, device=feature.device, dtype=feature.dtype) * (2.0 / 127.0)
    grid_r, grid_a = torch.meshgrid(offsets, offsets, indexing="ij")
    grid = torch.stack(
        (angle_norm[:, :, None, None] + grid_a, range_norm[:, :, None, None] + grid_r), dim=-1
    ).reshape(batch * candidates, size, size, 2)
    repeated = feature.repeat_interleave(candidates, dim=0)
    crop = F.grid_sample(repeated, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return crop.reshape(batch, candidates, feature.shape[1], size, size)


class _CropEncoder(nn.Module):
    def __init__(self, input_channels: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
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
        return self.project(self.network(value).flatten(1))


class CRUWPBMHBRExpert(nn.Module):
    """Rerank six causal radar candidates while retaining candidate zero as no-op."""

    candidate_count = 6

    def __init__(self, range_grid: np.ndarray, angle_grid: np.ndarray, geometry: CRUWBEVGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.register_buffer("range_grid", torch.as_tensor(range_grid, dtype=torch.float32))
        self.register_buffer("angle_grid", torch.as_tensor(angle_grid, dtype=torch.float32))
        self.bev_encoder = _CropEncoder(128, 48)
        self.polar_encoder = _CropEncoder(48, 40)
        self.raw_encoder = _CropEncoder(2, 24)
        self.type_embedding = nn.Embedding(3, 8)
        self.token_project = nn.Sequential(nn.Linear(48 + 40 + 24 + 8 + 6, 96), nn.SiLU(), nn.Linear(96, 96))
        self.attention = nn.MultiheadAttention(96, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(96)
        self.selector = nn.Linear(96, 1)
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
    ) -> dict[str, torch.Tensor]:
        base_xy = _decode_peak_tensor(base_logits, self.geometry)
        top = _metric_candidates(base_logits, self.geometry, self.candidate_count - 1)
        previous_xy = _decode_peak_tensor(previous_logits, self.geometry)
        # Candidate zero is forced to be the exact base peak; ranks 2-5 and the
        # previous causal peak complete the fixed candidate set.
        candidates = torch.cat((base_xy[:, None], top[:, 1:], previous_xy[:, None]), dim=1)
        types = torch.tensor((0, 1, 1, 1, 1, 2), device=radar.device).expand(radar.shape[0], -1)
        batch, count = candidates.shape[:2]
        bev_crop = _crop_bev(shared_feature, candidates, self.geometry)
        polar_crop = _crop_polar(polar_feature, candidates, self.range_grid, self.angle_grid)
        raw_crop = _crop_polar(radar[:, -1], candidates, self.range_grid, self.angle_grid)
        bev_embedding = self.bev_encoder(bev_crop.reshape(batch * count, *bev_crop.shape[2:])).reshape(batch, count, -1)
        polar_embedding = self.polar_encoder(polar_crop.reshape(batch * count, *polar_crop.shape[2:])).reshape(batch, count, -1)
        raw_embedding = self.raw_encoder(raw_crop.reshape(batch * count, *raw_crop.shape[2:])).reshape(batch, count, -1)
        geometry_features = torch.cat(
            (
                candidates / candidates.new_tensor((10.0, 23.4)),
                (candidates - base_xy[:, None]) / 2.0,
                (candidates - previous_xy[:, None]) / 2.0,
            ),
            dim=-1,
        )
        tokens = self.token_project(
            torch.cat((bev_embedding, polar_embedding, raw_embedding, self.type_embedding(types), geometry_features), dim=-1)
        )
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attended)
        selector_logits = self.selector(tokens).squeeze(-1)
        selector_logits = selector_logits + torch.cat(
            (selector_logits.new_zeros((batch, 1)), self.nonbase_bias.expand(batch, count - 1)), dim=1
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
        }


def expert_target_indices(
    candidate_xy_m: torch.Tensor,
    target_xy_m: torch.Tensor,
    *,
    minimum_improvement_m: float = 0.05,
) -> torch.Tensor:
    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[0] != target_xy_m.shape[0]:
        raise ValueError("candidate and target batches must align")
    errors = torch.linalg.vector_norm(candidate_xy_m - target_xy_m[:, None], dim=-1)
    best = errors.argmin(dim=1)
    improvement = errors[:, 0] - errors[torch.arange(errors.shape[0], device=errors.device), best]
    return torch.where(improvement >= minimum_improvement_m, best, torch.zeros_like(best))
