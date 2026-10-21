"""Train-only selection utilities for the causal consistency guard."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class CalibrationRow:
    """One frozen causal replay result at a candidate EMA threshold."""

    threshold: float
    clean_ap30: float
    biased_ap30: float


def choose_guard_threshold(
    rows: Iterable[CalibrationRow],
    *,
    radar_clean_ap30: float,
    radar_biased_ap30: float,
    minimum_clean_gain: float,
) -> CalibrationRow | None:
    """Return the most protective threshold that satisfies the frozen rule."""

    if minimum_clean_gain < 0.0:
        raise ValueError("minimum_clean_gain must be non-negative")
    eligible = [
        row
        for row in rows
        if row.clean_ap30 >= radar_clean_ap30 + minimum_clean_gain
        and row.biased_ap30 >= radar_biased_ap30
    ]
    return min(eligible, key=lambda row: row.threshold) if eligible else None
