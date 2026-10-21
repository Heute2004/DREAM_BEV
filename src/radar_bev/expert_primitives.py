from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as functional

from .geometry import BEVGeometry


def causal_previous_window(radar: Tensor) -> Tensor:
    """Shift a causal radar window by one frame with first-frame padding."""
    if radar.ndim != 5 or radar.shape[1] < 1:
        raise ValueError("radar must be [batch,frames,channels,height,width] with at least one frame")
    return torch.cat((radar[:, :1], radar[:, :-1]), dim=1)


def _validate_logits(logits: Tensor, geometry: BEVGeometry) -> None:
    if logits.ndim != 4 or logits.shape[1:] != (1, geometry.x_bins, geometry.y_bins):
        raise ValueError(f"logits must be [batch,1,{geometry.x_bins},{geometry.y_bins}]")


def decode_peak_xy(logits: Tensor, geometry: BEVGeometry | None = None) -> tuple[Tensor, Tensor]:
    """Decode the deterministic base peak in frozen metric BEV coordinates."""
    geometry = geometry or BEVGeometry()
    _validate_logits(logits, geometry)
    score, index = logits[:, 0].flatten(start_dim=1).max(dim=1)
    x_index = torch.div(index, geometry.y_bins, rounding_mode="floor")
    y_index = index.remainder(geometry.y_bins)
    xy = geometry.index_to_metric(torch.stack((x_index, y_index), dim=-1))
    return xy, score.unsqueeze(1)


def local_maximum_candidates(
    logits: Tensor,
    count: int,
    geometry: BEVGeometry | None = None,
    suppression_kernel: int = 5,
) -> tuple[Tensor, Tensor]:
    """Return deterministic local-maxima candidates sorted by their BEV logits."""
    geometry = geometry or BEVGeometry()
    _validate_logits(logits, geometry)
    if count < 1:
        raise ValueError("count must be positive")
    if suppression_kernel < 1 or suppression_kernel % 2 != 1:
        raise ValueError("suppression_kernel must be a positive odd integer")
    pooled = functional.max_pool2d(logits, kernel_size=suppression_kernel, stride=1, padding=suppression_kernel // 2)
    candidates = logits.masked_fill(logits.ne(pooled), torch.finfo(logits.dtype).min)
    scores, flat_indices = candidates[:, 0].flatten(start_dim=1).topk(count, dim=1)
    x_indices = torch.div(flat_indices, geometry.y_bins, rounding_mode="floor")
    y_indices = flat_indices.remainder(geometry.y_bins)
    xy = geometry.index_to_metric(torch.stack((x_indices, y_indices), dim=-1))
    return xy, scores


def _metric_local_grid(centers_xy_m: Tensor, span_m: float, output_size: int) -> Tensor:
    if centers_xy_m.ndim != 2 or centers_xy_m.shape[1] != 2:
        raise ValueError("centers_xy_m must be [batch,2]")
    if span_m <= 0.0:
        raise ValueError("span_m must be positive")
    if output_size < 2:
        raise ValueError("output_size must be at least two")
    offset = torch.linspace(
        -span_m / 2.0,
        span_m / 2.0,
        output_size,
        device=centers_xy_m.device,
        dtype=centers_xy_m.dtype,
    )
    offset_x, offset_y = torch.meshgrid(offset, offset, indexing="ij")
    return centers_xy_m[:, None, None, :] + torch.stack((offset_x, offset_y), dim=-1).unsqueeze(0)


def metric_bev_crop(
    feature: Tensor,
    centers_xy_m: Tensor,
    geometry: BEVGeometry | None = None,
    span_m: float = 2.0,
    output_size: int = 7,
) -> Tensor:
    """Sample a metric local ROI from a BEV feature whose axes are [x,y]."""
    geometry = geometry or BEVGeometry()
    if feature.ndim != 4:
        raise ValueError("feature must be [batch,channels,height,width]")
    if feature.shape[0] != centers_xy_m.shape[0]:
        raise ValueError("feature and centers_xy_m must have the same batch size")
    metric = _metric_local_grid(centers_xy_m, span_m, output_size)
    x_norm = 2.0 * (metric[..., 0] - geometry.x_min_m) / (geometry.x_max_m - geometry.x_min_m) - 1.0
    y_norm = 2.0 * (metric[..., 1] - geometry.y_min_m) / (geometry.y_max_m - geometry.y_min_m) - 1.0
    # grid_sample receives [width, height]; this feature's width is the metric y-axis.
    grid = torch.stack((y_norm, x_norm), dim=-1)
    return functional.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def metric_polar_crop(
    feature: Tensor,
    centers_xy_m: Tensor,
    geometry: BEVGeometry | None = None,
    span_m: float = 2.0,
    output_size: int = 9,
) -> Tensor:
    """Sample native polar evidence using a local metric BEV tube around a target prior."""
    geometry = geometry or BEVGeometry()
    if feature.ndim != 4 or feature.shape[-2:] != (geometry.range_bins, geometry.azimuth_bins):
        raise ValueError(f"feature must be [batch,channels,{geometry.range_bins},{geometry.azimuth_bins}]")
    if feature.shape[0] != centers_xy_m.shape[0]:
        raise ValueError("feature and centers_xy_m must have the same batch size")
    metric = _metric_local_grid(centers_xy_m, span_m, output_size)
    x, y = metric.unbind(dim=-1)
    range_m = torch.sqrt(x.square() + y.square())
    sin_azimuth = torch.where(range_m > 1e-6, x / range_m, torch.zeros_like(range_m))
    range_index = range_m / geometry.range_resolution_m
    azimuth_index = (sin_azimuth + 1.0) * (geometry.azimuth_bins / 2.0)
    range_norm = 2.0 * range_index / (geometry.range_bins - 1) - 1.0
    azimuth_norm = 2.0 * azimuth_index / (geometry.azimuth_bins - 1) - 1.0
    grid = torch.stack((azimuth_norm, range_norm), dim=-1)
    return functional.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def gaussian_logit_delta(
    base_xy_m: Tensor,
    proposal_xy_m: Tensor,
    strength: Tensor,
    geometry: BEVGeometry | None = None,
    sigma_m: float = 0.30,
) -> Tensor:
    """Render a bounded local residual as a positive proposal and negative base Gaussian."""
    geometry = geometry or BEVGeometry()
    if base_xy_m.shape != proposal_xy_m.shape or base_xy_m.ndim != 2 or base_xy_m.shape[1] != 2:
        raise ValueError("base_xy_m and proposal_xy_m must both be [batch,2]")
    if strength.ndim == 1:
        strength = strength.unsqueeze(1)
    if strength.shape != (base_xy_m.shape[0], 1):
        raise ValueError("strength must be [batch] or [batch,1]")
    if sigma_m <= 0.0:
        raise ValueError("sigma_m must be positive")
    grid = geometry.coordinate_grid(device=base_xy_m.device, dtype=base_xy_m.dtype)

    def render(center_xy_m: Tensor) -> Tensor:
        distance_squared = (
            (grid[..., 0].unsqueeze(0) - center_xy_m[:, 0, None, None]).square()
            + (grid[..., 1].unsqueeze(0) - center_xy_m[:, 1, None, None]).square()
        )
        return torch.exp(-0.5 * distance_squared / (sigma_m * sigma_m))

    residual = render(proposal_xy_m) - render(base_xy_m)
    return strength[:, :, None, None] * residual.unsqueeze(1)



