from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


SIGNATURE_FIELDS = (
    "ra_peak_to_background_db",
    "mti_peak_to_background_db",
    "ra_separated_peak_margin_db",
    "mti_separated_peak_margin_db",
    "ra_normalized_entropy",
    "mti_normalized_entropy",
    "ra_mti_peak_distance_bins",
    "ra_mti_correlation",
    "mti_temporal_peak_jump_bins",
    "ra_diffuse_fraction_6db",
    "mti_diffuse_fraction_6db",
)


def _validate_map(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError(f"expected a two-dimensional radar map, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("radar map contains non-finite values")
    return array


def _peak_index(values: np.ndarray) -> tuple[int, int]:
    return tuple(int(value) for value in np.unravel_index(np.argmax(values), values.shape))


def separated_peak_margin_db(values: np.ndarray, exclusion_radius: int = 3) -> float:
    """Peak-to-separated-second-peak margin on a dB-like radar map."""

    array = _validate_map(values)
    if exclusion_radius < 0:
        raise ValueError("exclusion_radius must be non-negative")
    peak_row, peak_col = _peak_index(array)
    masked = array.copy()
    row_min = max(0, peak_row - exclusion_radius)
    row_max = min(array.shape[0], peak_row + exclusion_radius + 1)
    col_min = max(0, peak_col - exclusion_radius)
    col_max = min(array.shape[1], peak_col + exclusion_radius + 1)
    masked[row_min:row_max, col_min:col_max] = -np.inf
    second = float(np.max(masked))
    return float(array[peak_row, peak_col] - second)


def normalized_power_entropy(values: np.ndarray) -> float:
    """Entropy of relative linear power, normalized to [0, 1]."""

    array = _validate_map(values)
    relative_power = np.power(10.0, np.clip(array - array.max(), -120.0, 0.0) / 10.0)
    probability = relative_power.ravel() / relative_power.sum()
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-15)))
    return float(entropy / np.log(probability.size))


def frame_mechanism_signature(
    ra: np.ndarray,
    ra_mti: np.ndarray,
    *,
    previous_mti_peak: tuple[int, int] | None = None,
) -> tuple[dict[str, float], tuple[int, int]]:
    ra_array = _validate_map(ra)
    mti_array = _validate_map(ra_mti)
    if ra_array.shape != mti_array.shape:
        raise ValueError("RA and RA-MTI maps must have identical shapes")
    ra_peak = _peak_index(ra_array)
    mti_peak = _peak_index(mti_array)
    peak_distance = float(np.linalg.norm(np.subtract(ra_peak, mti_peak)))
    temporal_jump = (
        0.0
        if previous_mti_peak is None
        else float(np.linalg.norm(np.subtract(mti_peak, previous_mti_peak)))
    )
    ra_flat = ra_array.ravel()
    mti_flat = mti_array.ravel()
    ra_std = float(ra_flat.std())
    mti_std = float(mti_flat.std())
    correlation = (
        0.0
        if ra_std <= 1e-12 or mti_std <= 1e-12
        else float(np.corrcoef(ra_flat, mti_flat)[0, 1])
    )
    signature = {
        "ra_peak_to_background_db": float(ra_array.max() - np.median(ra_array)),
        "mti_peak_to_background_db": float(mti_array.max() - np.median(mti_array)),
        "ra_separated_peak_margin_db": separated_peak_margin_db(ra_array),
        "mti_separated_peak_margin_db": separated_peak_margin_db(mti_array),
        "ra_normalized_entropy": normalized_power_entropy(ra_array),
        "mti_normalized_entropy": normalized_power_entropy(mti_array),
        "ra_mti_peak_distance_bins": peak_distance,
        "ra_mti_correlation": correlation,
        "mti_temporal_peak_jump_bins": temporal_jump,
        "ra_diffuse_fraction_6db": float(np.mean(ra_array >= ra_array.max() - 6.0)),
        "mti_diffuse_fraction_6db": float(np.mean(mti_array >= mti_array.max() - 6.0)),
    }
    return signature, mti_peak


def zscore_signature_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, float]]:
    if not rows:
        raise ValueError("cannot standardize empty signature rows")
    matrix = np.asarray(
        [[float(row[field]) for field in SIGNATURE_FIELDS] for row in rows],
        dtype=np.float64,
    )
    center = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-9, scale, np.std(matrix, axis=0))
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (matrix - center) / scale
    return [
        {field: float(standardized[index, column]) for column, field in enumerate(SIGNATURE_FIELDS)}
        for index in range(len(rows))
    ]


def route_signature_score(route: str, standardized: Mapping[str, float]) -> float:
    z = lambda field: float(standardized[field])
    if route == "open_indoor_los":
        return float(
            0.25 * (z("ra_peak_to_background_db") + z("mti_peak_to_background_db"))
            + 0.25 * z("ra_mti_correlation")
            - 0.25 * z("mti_temporal_peak_jump_bins")
            - 0.25 * z("ra_mti_peak_distance_bins")
        )
    if route == "indoor_clutter":
        return float(
            0.25 * (z("ra_normalized_entropy") + z("mti_normalized_entropy"))
            - 0.25 * (z("ra_separated_peak_margin_db") + z("mti_separated_peak_margin_db"))
            + 0.25 * z("mti_temporal_peak_jump_bins")
            - 0.25 * z("ra_mti_correlation")
        )
    if route == "open_outdoor_weak_return":
        return float(
            -0.25 * (z("ra_peak_to_background_db") + z("mti_peak_to_background_db"))
            + 0.25 * z("ra_normalized_entropy")
            + 0.25 * z("mti_normalized_entropy")
            - 0.25 * z("ra_mti_correlation")
        )
    if route == "outdoor_dry_complex":
        return float(
            -0.25 * (z("ra_separated_peak_margin_db") + z("mti_separated_peak_margin_db"))
            + 0.25 * z("mti_temporal_peak_jump_bins")
            + 0.25 * z("ra_mti_peak_distance_bins")
            + 0.25 * z("ra_peak_to_background_db")
        )
    if route == "outdoor_wet_return":
        return float(
            0.25 * (z("ra_diffuse_fraction_6db") + z("mti_diffuse_fraction_6db"))
            + 0.25 * (z("ra_normalized_entropy") + z("mti_normalized_entropy"))
            + 0.25 * z("ra_mti_peak_distance_bins")
            - 0.25 * z("ra_mti_correlation")
        )
    raise ValueError(f"unsupported route: {route}")


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Spearman inputs must be aligned one-dimensional arrays")
    if len(left) < 3:
        return float("nan")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    if left_rank.std() <= 1e-12 or right_rank.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])
