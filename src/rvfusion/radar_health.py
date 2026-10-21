from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from radar_bev.geometry import BEVGeometry
from torch import Tensor
from torch.nn import functional


@dataclass(frozen=True)
class RadarHealthEvidence:
    vector: Tensor
    top1_probability: Tensor
    top1_top2_margin: Tensor
    entropy: Tensor
    covariance_trace: Tensor


@dataclass(frozen=True)
class PersistentRadarState:
    assist_enabled: Tensor
    uncertainty_strength: Tensor


def _validate_inputs(
    logits: Tensor,
    ra_current: Tensor,
    ra_mti_current: Tensor,
    previous_xy_m: Tensor,
    predicted_xy_m: Tensor,
) -> None:
    batch = logits.shape[0]
    if tuple(logits.shape) != (batch, 1, 200, 234):
        raise ValueError("radar logits must be [batch,1,200x234]")
    if tuple(ra_current.shape) != (batch, 128, 128):
        raise ValueError("ra_current must be [batch,128,128]")
    if tuple(ra_mti_current.shape) != tuple(ra_current.shape):
        raise ValueError("ra_mti_current must match ra_current")
    if tuple(previous_xy_m.shape) != (batch, 2):
        raise ValueError("previous_xy_m must be [batch,2]")
    if tuple(predicted_xy_m.shape) != (batch, 2):
        raise ValueError("predicted_xy_m must be [batch,2]")
    for name, value in (
        ("logits", logits),
        ("ra_current", ra_current),
        ("ra_mti_current", ra_mti_current),
        ("previous_xy_m", previous_xy_m),
        ("predicted_xy_m", predicted_xy_m),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")


def _decode_bev_peak(probability_map: Tensor) -> Tensor:
    batch, _, _, width = probability_map.shape
    flat_index = probability_map.flatten(1).argmax(dim=1)
    x_index = torch.div(flat_index, width, rounding_mode="floor")
    y_index = flat_index.remainder(width)
    x_m = -10.0 + x_index.to(probability_map.dtype) * 0.1
    y_m = y_index.to(probability_map.dtype) * 0.1
    return torch.stack((x_m, y_m), dim=1)


def _coordinate_covariance_trace(probability_map: Tensor) -> Tensor:
    batch, _, height, width = probability_map.shape
    probability = probability_map[:, 0]
    x = -10.0 + (
        torch.arange(height, device=probability.device, dtype=probability.dtype)
        + 0.5
    ) * 0.1
    y = (
        torch.arange(width, device=probability.device, dtype=probability.dtype)
        + 0.5
    ) * 0.1
    x_marginal = probability.sum(dim=2)
    y_marginal = probability.sum(dim=1)
    mean_x = (x_marginal * x.unsqueeze(0)).sum(dim=1)
    mean_y = (y_marginal * y.unsqueeze(0)).sum(dim=1)
    var_x = (x_marginal * (x.unsqueeze(0) - mean_x[:, None]).square()).sum(dim=1)
    var_y = (y_marginal * (y.unsqueeze(0) - mean_y[:, None]).square()).sum(dim=1)
    trace = var_x + var_y
    if tuple(trace.shape) != (batch,):
        raise RuntimeError("invalid covariance trace shape")
    return trace


def _polar_patches_from_xy(
    ra: Tensor,
    ra_mti: Tensor,
    xy_m: Tensor,
    *,
    geometry: BEVGeometry,
    half_width: int,
) -> tuple[Tensor, Tensor]:
    radial_range = xy_m.square().sum(dim=1).sqrt()
    sine_azimuth = xy_m[:, 0] / radial_range.clamp_min(1e-6)
    range_index = (
        radial_range / geometry.range_resolution_m
    ).round().long().clamp(half_width, geometry.range_bins - half_width - 1)
    azimuth_index = (
        (sine_azimuth + 1.0) * (geometry.azimuth_bins / 2.0)
    ).round().long().clamp(half_width, geometry.azimuth_bins - half_width - 1)
    offsets = torch.arange(
        -half_width,
        half_width + 1,
        device=ra.device,
        dtype=torch.long,
    )
    range_rows = range_index[:, None, None] + offsets[None, :, None]
    azimuth_columns = azimuth_index[:, None, None] + offsets[None, None, :]
    batch_index = torch.arange(ra.shape[0], device=ra.device)[:, None, None]
    ra_patch = ra[batch_index, range_rows, azimuth_columns]
    mti_patch = ra_mti[batch_index, range_rows, azimuth_columns]
    return ra_patch.flatten(1), mti_patch.flatten(1)


def _centered_cosine(first: Tensor, second: Tensor) -> Tensor:
    first = first - first.mean(dim=1, keepdim=True)
    second = second - second.mean(dim=1, keepdim=True)
    cosine = functional.cosine_similarity(first, second, dim=1, eps=1e-6)
    return ((cosine + 1.0) * 0.5).clamp(0.0, 1.0)


def _bev_peak_to_background(probability_map: Tensor) -> Tensor:
    top1 = probability_map.flatten(1).amax(dim=1)
    background = probability_map.flatten(1).mean(dim=1).clamp_min(1e-12)
    ratio = top1 / background
    return (1.0 - torch.exp(-ratio / 20.0)).clamp(0.0, 1.0)


def compute_radar_health(
    logits: Tensor,
    ra_current: Tensor,
    ra_mti_current: Tensor,
    previous_xy_m: Tensor,
    predicted_xy_m: Tensor,
    *,
    geometry: BEVGeometry | None = None,
) -> RadarHealthEvidence:
    _validate_inputs(
        logits,
        ra_current,
        ra_mti_current,
        previous_xy_m,
        predicted_xy_m,
    )
    geometry = geometry or BEVGeometry()
    probability_map = logits.float().flatten(1).softmax(dim=1).view_as(logits)
    probability = probability_map.flatten(1)
    top2 = probability.topk(k=2, dim=1).values
    top1 = top2[:, 0]
    margin = top1 - top2[:, 1]
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)
    entropy = entropy / math.log(probability.shape[1])
    covariance_trace = _coordinate_covariance_trace(probability_map)
    decoded_xy_m = _decode_bev_peak(probability_map)
    ra_patch, mti_patch = _polar_patches_from_xy(
        ra_current.float(),
        ra_mti_current.float(),
        decoded_xy_m,
        geometry=geometry,
        half_width=2,
    )
    ra_mti_agreement = _centered_cosine(ra_patch, mti_patch)
    peak_to_background = _bev_peak_to_background(probability_map)
    peak_displacement = torch.linalg.vector_norm(
        decoded_xy_m - previous_xy_m.float(),
        dim=1,
    ).clamp(0.0, 1.0)
    tracker_innovation = torch.linalg.vector_norm(
        decoded_xy_m - predicted_xy_m.float(),
        dim=1,
    ).clamp(0.0, 1.0)
    vector = torch.stack(
        (
            1.0 - top1,
            1.0 - margin,
            entropy,
            covariance_trace.clamp(0.0, 2.0) / 2.0,
            1.0 - peak_to_background,
            1.0 - ra_mti_agreement,
            peak_displacement,
            tracker_innovation,
        ),
        dim=1,
    ).clamp(0.0, 1.0)
    return RadarHealthEvidence(
        vector=vector,
        top1_probability=top1,
        top1_top2_margin=margin,
        entropy=entropy,
        covariance_trace=covariance_trace,
    )


def persistent_uncertainty_state(
    history: Tensor,
    *,
    frame_ids: Tensor | None = None,
    enter_frames: int = 3,
    leave_frames: int = 5,
) -> PersistentRadarState:
    if history.ndim != 3 or history.shape[2] != 8:
        raise ValueError("history must be [batch,time,8]")
    if history.shape[1] < 1:
        raise ValueError("history must contain at least one frame")
    if enter_frames < 1 or leave_frames < 1:
        raise ValueError("enter_frames and leave_frames must be positive")
    if not torch.isfinite(history).all():
        raise ValueError("history must be finite")
    if not ((0.0 <= history) & (history <= 1.0)).all():
        raise ValueError("history values must be in [0,1]")
    if frame_ids is not None:
        if tuple(frame_ids.shape) != tuple(history.shape[:2]):
            raise ValueError("frame_ids must be [batch,time]")
        if frame_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("frame_ids must be integer")

    weights = history.new_tensor((0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10))
    uncertainty_score = (history * weights).sum(dim=2)
    uncertain = uncertainty_score > 0.50
    enabled = torch.zeros(
        history.shape[0],
        dtype=torch.bool,
        device=history.device,
    )
    uncertain_count = torch.zeros_like(enabled, dtype=torch.int64)
    stable_count = torch.zeros_like(enabled, dtype=torch.int64)
    for time_index in range(history.shape[1]):
        current = uncertain[:, time_index]
        is_new_frame = (
            torch.ones_like(enabled)
            if frame_ids is None or time_index == 0
            else frame_ids[:, time_index] != frame_ids[:, time_index - 1]
        )
        next_uncertain_count = torch.where(
            current,
            uncertain_count + 1,
            torch.zeros_like(uncertain_count),
        )
        next_stable_count = torch.where(
            current,
            torch.zeros_like(stable_count),
            stable_count + 1,
        )
        uncertain_count = torch.where(
            is_new_frame,
            next_uncertain_count,
            uncertain_count,
        )
        stable_count = torch.where(
            is_new_frame,
            next_stable_count,
            stable_count,
        )
        enabled = torch.where(
            uncertain_count >= enter_frames,
            torch.ones_like(enabled),
            enabled,
        )
        enabled = torch.where(
            stable_count >= leave_frames,
            torch.zeros_like(enabled),
            enabled,
        )
    raw_strength = torch.sigmoid(8.0 * (uncertainty_score[:, -1] - 0.50))
    strength = raw_strength * enabled.to(raw_strength.dtype)
    return PersistentRadarState(
        assist_enabled=enabled,
        uncertainty_strength=strength,
    )



