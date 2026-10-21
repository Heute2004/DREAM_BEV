from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TEACHER_CACHE_KEYS = (
    "frame_ids",
    "geometry_theta_rad",
    "geometry_sigma_rad",
    "geometry_valid",
    "candidate_theta_rad",
    "candidate_support",
    "candidate_valid",
    "candidate_positive_index",
)


@dataclass(frozen=True)
class GeometryCalibration:
    slope: float
    offset_rad: float
    residual_scale_rad: float
    inlier_count: int


@dataclass(frozen=True)
class RadarCandidates:
    theta_rad: np.ndarray
    support: np.ndarray
    valid: np.ndarray
    positive_index: int


def fit_geometry_calibration(
    camera_theta_rad: np.ndarray,
    target_theta_rad: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    minimum_samples: int = 8,
    slope_bounds: tuple[float, float] = (0.90, 1.10),
    offset_bound_rad: float = math.radians(6.0),
) -> GeometryCalibration:
    camera = np.asarray(camera_theta_rad, dtype=np.float64).reshape(-1)
    target = np.asarray(target_theta_rad, dtype=np.float64).reshape(-1)
    if camera.shape != target.shape:
        raise ValueError("camera and target angles must have identical shapes")
    mask = np.isfinite(camera) & np.isfinite(target)
    if valid is not None:
        valid_array = np.asarray(valid, dtype=bool).reshape(-1)
        if valid_array.shape != mask.shape:
            raise ValueError("valid mask must match angle arrays")
        mask &= valid_array
    x = camera[mask]
    y = target[mask]
    if x.size < 2:
        return GeometryCalibration(1.0, 0.0, math.radians(3.0), int(x.size))

    delta_x = x[:, None] - x[None, :]
    delta_y = y[:, None] - y[None, :]
    upper = np.triu(np.ones(delta_x.shape, dtype=bool), k=1)
    pair_mask = upper & (np.abs(delta_x) > 1e-4)
    slopes = delta_y[pair_mask] / delta_x[pair_mask]
    slope = float(np.median(slopes)) if slopes.size else 1.0
    slope = float(np.clip(slope, *slope_bounds))
    offset = float(np.median(y - slope * x))
    offset = float(np.clip(offset, -offset_bound_rad, offset_bound_rad))

    residual = y - (slope * x + offset)
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    robust_scale = max(1.4826 * mad, math.radians(0.15))
    inliers = np.abs(residual - center) <= max(
        3.0 * robust_scale,
        math.radians(0.5),
    )
    if int(inliers.sum()) >= max(2, minimum_samples):
        design = np.stack((x[inliers], np.ones(int(inliers.sum()))), axis=1)
        fitted, *_ = np.linalg.lstsq(design, y[inliers], rcond=None)
        slope = float(np.clip(fitted[0], *slope_bounds))
        offset = float(np.median(y[inliers] - slope * x[inliers]))
        offset = float(np.clip(offset, -offset_bound_rad, offset_bound_rad))
        residual = y[inliers] - (slope * x[inliers] + offset)
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        robust_scale = max(1.4826 * mad, math.radians(0.15))
    return GeometryCalibration(
        slope=slope,
        offset_rad=offset,
        residual_scale_rad=robust_scale,
        inlier_count=int(inliers.sum()),
    )


def _unit_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    clean = np.where(finite, values, np.nan)
    lower = float(np.nanpercentile(clean, 5.0))
    upper = float(np.nanpercentile(clean, 99.5))
    if upper <= lower + 1e-6:
        upper = float(np.nanmax(clean))
    if upper <= lower + 1e-6:
        return np.zeros_like(values)
    return np.clip((np.nan_to_num(clean, nan=lower) - lower) / (upper - lower), 0.0, 1.0)


def _theta_to_column(theta_rad: float, azimuth_bins: int) -> float:
    return (math.sin(theta_rad) + 1.0) * (azimuth_bins / 2.0)


def _column_to_theta(column: float, azimuth_bins: int) -> float:
    sine = column / (azimuth_bins / 2.0) - 1.0
    return math.asin(float(np.clip(sine, -1.0, 1.0)))


def _interpolate_profile(profile: np.ndarray, column: float) -> float:
    column = float(np.clip(column, 0.0, profile.size - 1.0))
    left = int(math.floor(column))
    right = min(profile.size - 1, left + 1)
    alpha = column - left
    return float((1.0 - alpha) * profile[left] + alpha * profile[right])


def extract_radar_candidates(
    *,
    ra: np.ndarray,
    ra_mti: np.ndarray,
    target_theta_rad: float,
    target_range_m: float,
    range_resolution_m: float,
    candidate_count: int = 8,
    range_half_width_bins: int = 1,
    minimum_separation_rad: float = math.radians(1.5),
) -> RadarCandidates:
    ra = np.asarray(ra, dtype=np.float32)
    ra_mti = np.asarray(ra_mti, dtype=np.float32)
    if ra.shape != ra_mti.shape or ra.ndim != 2:
        raise ValueError("RA and RA-MTI must be matching two-dimensional maps")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    if range_resolution_m <= 0.0:
        raise ValueError("range_resolution_m must be positive")
    range_bins, azimuth_bins = ra.shape
    row = int(round(target_range_m / range_resolution_m))
    row = int(np.clip(row, 0, range_bins - 1))
    start = max(0, row - range_half_width_bins)
    stop = min(range_bins, row + range_half_width_bins + 1)
    ra_patch = _unit_normalize(ra[start:stop])
    mti_patch = _unit_normalize(ra_mti[start:stop])
    profile = np.maximum(ra_patch, mti_patch).max(axis=0)

    theta = np.zeros(candidate_count, dtype=np.float32)
    support = np.zeros(candidate_count, dtype=np.float32)
    valid = np.zeros(candidate_count, dtype=bool)
    theta[0] = float(target_theta_rad)
    target_column = _theta_to_column(target_theta_rad, azimuth_bins)
    support[0] = _interpolate_profile(profile, target_column)
    valid[0] = True

    if float(profile.max(initial=0.0)) <= 0.0:
        return RadarCandidates(theta, support, valid, 0)
    left = np.r_[-np.inf, profile[:-1]]
    right = np.r_[profile[1:], -np.inf]
    maxima = np.flatnonzero((profile >= left) & (profile >= right) & (profile > 0.0))
    ordered = maxima[np.argsort(profile[maxima])[::-1]]
    selected = 1
    for column in ordered.tolist():
        candidate_theta = _column_to_theta(float(column), azimuth_bins)
        if abs(candidate_theta - target_theta_rad) < minimum_separation_rad:
            continue
        if any(
            abs(candidate_theta - float(theta[index])) < minimum_separation_rad
            for index in range(1, selected)
        ):
            continue
        theta[selected] = candidate_theta
        support[selected] = float(profile[column])
        valid[selected] = True
        selected += 1
        if selected >= candidate_count:
            break
    return RadarCandidates(theta, support, valid, 0)


def load_teacher_sequence(path: str | Path) -> tuple[dict[str, np.ndarray], dict[int, int]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing training teacher cache: {path}")
    with np.load(path, allow_pickle=False) as payload:
        missing = set(TEACHER_CACHE_KEYS) - set(payload.files)
        if missing:
            raise ValueError(f"teacher cache lacks {sorted(missing)}: {path}")
        values = {key: np.array(payload[key], copy=True) for key in TEACHER_CACHE_KEYS}
    frame_ids = values["frame_ids"].astype(np.int64, copy=False)
    if frame_ids.ndim != 1 or len(set(frame_ids.tolist())) != frame_ids.size:
        raise ValueError(f"teacher frame IDs must be unique and one-dimensional: {path}")
    if any(value.shape[0] != frame_ids.size for value in values.values()):
        raise ValueError(f"teacher arrays are not frame aligned: {path}")
    if values["candidate_theta_rad"].ndim != 2:
        raise ValueError(f"candidate arrays must be [frames,K]: {path}")
    candidate_shape = values["candidate_theta_rad"].shape
    for key in ("candidate_support", "candidate_valid"):
        if values[key].shape != candidate_shape:
            raise ValueError(f"{key} does not match candidate shape: {path}")
    frame_index = {int(frame_id): index for index, frame_id in enumerate(frame_ids.tolist())}
    return values, frame_index



