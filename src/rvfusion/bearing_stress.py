"""Deterministic Development-only perturbations for frozen bearing evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


class BearingStressError(ValueError):
    """Raised when a bearing-cache perturbation violates its contract."""


@dataclass(frozen=True)
class BearingStressSummary:
    mode: str
    affected_frames: int
    total_frames: int


_REQUIRED_KEYS = {
    "frame_ids",
    "theta",
    "sigma_theta",
    "reliability",
    "temporal_feature",
    "valid",
}


def _copy_cache(cache: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    missing = _REQUIRED_KEYS - set(cache)
    if missing:
        raise BearingStressError(f"cache lacks {sorted(missing)}")
    frame_ids = np.asarray(cache["frame_ids"])
    if frame_ids.ndim != 1 or frame_ids.size == 0:
        raise BearingStressError("frame_ids must be a non-empty vector")
    total = frame_ids.size
    if np.asarray(cache["theta"]).shape != (total,):
        raise BearingStressError("theta shape does not match frame_ids")
    if np.asarray(cache["sigma_theta"]).shape != (total,):
        raise BearingStressError("sigma_theta shape does not match frame_ids")
    if np.asarray(cache["reliability"]).shape != (total,):
        raise BearingStressError("reliability shape does not match frame_ids")
    if np.asarray(cache["valid"]).shape != (total,):
        raise BearingStressError("valid shape does not match frame_ids")
    if np.asarray(cache["temporal_feature"]).shape[0] != total:
        raise BearingStressError("temporal_feature does not match frame_ids")
    return {name: np.array(value, copy=True) for name, value in cache.items()}


def apply_bearing_bias(
    cache: Mapping[str, np.ndarray], *, bias_rad: float
) -> tuple[dict[str, np.ndarray], BearingStressSummary]:
    """Offset every valid bearing while retaining cache uncertainty and support."""

    if not np.isfinite(bias_rad):
        raise BearingStressError("bias_rad must be finite")
    result = _copy_cache(cache)
    valid = result["valid"].astype(bool, copy=False)
    theta = result["theta"].astype(np.float32, copy=False)
    theta[valid] += np.float32(bias_rad)
    result["theta"] = theta
    return result, BearingStressSummary(
        mode="bearing_bias",
        affected_frames=int(valid.sum()),
        total_frames=int(valid.size),
    )


def apply_temporal_lag(
    cache: Mapping[str, np.ndarray], *, lag_frames: int
) -> tuple[dict[str, np.ndarray], BearingStressSummary]:
    """Replace evidence at t by evidence from t-lag without future access."""

    if lag_frames < 1:
        raise BearingStressError("lag_frames must be positive")
    result = _copy_cache(cache)
    total = result["frame_ids"].size
    lag = min(int(lag_frames), total)
    for name in ("theta", "sigma_theta", "reliability", "temporal_feature", "valid"):
        source = np.asarray(cache[name])
        target = result[name]
        if lag < total:
            target[lag:] = source[:-lag]
    result["valid"][:lag] = False
    result["reliability"][:lag] = np.float32(0.0)
    result["temporal_feature"][:lag] = np.float32(0.0)
    result["sigma_theta"][:lag] = np.float32(max(1.0, float(np.max(cache["sigma_theta"]))))
    return result, BearingStressSummary(
        mode="temporal_lag",
        affected_frames=lag,
        total_frames=total,
    )


def apply_visual_dropout(
    cache: Mapping[str, np.ndarray], *, probability: float, seed: int
) -> tuple[dict[str, np.ndarray], BearingStressSummary]:
    """Invalidate a deterministic subset of valid visual evidence frames."""

    if not 0.0 <= probability <= 1.0:
        raise BearingStressError("dropout probability must be in [0, 1]")
    result = _copy_cache(cache)
    valid = result["valid"].astype(bool, copy=False)
    generator = np.random.default_rng(seed)
    dropped = valid & (generator.random(valid.size) < probability)
    result["valid"][dropped] = False
    result["reliability"][dropped] = np.float32(0.0)
    result["temporal_feature"][dropped] = np.float32(0.0)
    result["sigma_theta"][dropped] = np.float32(max(1.0, float(np.max(cache["sigma_theta"]))))
    return result, BearingStressSummary(
        mode="visual_dropout",
        affected_frames=int(dropped.sum()),
        total_frames=int(valid.size),
    )
