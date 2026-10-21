from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[percentile - 1]


def _summarize_rows(
    rows: list[Mapping[str, object]],
    *,
    error_tolerance_m: float,
    noop_tolerance_m: float,
) -> dict[str, object]:
    policy_identity = 0
    visual_admitted = 0
    health_diagnostic_active = 0
    selected_noop = 0
    improved = 0
    harmed = 0
    unchanged = 0
    final_abs_delta_x: list[float] = []
    admitted_abs_delta_x: list[float] = []

    for row in rows:
        if "accepted" not in row:
            raise KeyError(
                "accepted is required: selectivity must be reconstructed from the "
                "deployed compositor gate, never from a diagnostic health field"
            )
        deployed_visual_active = _as_bool(row["accepted"])
        health_diagnostic_active += int(_as_bool(row.get("assist_enabled", False)))
        radar_x = float(row["radar_x_m"])
        radar_y = float(row["radar_y_m"])
        candidate_x = float(row["candidate_x_m"])
        target_x = float(row["target_x_m"])
        target_y = float(row["target_y_m"])

        fused_x = row.get("fused_x_m")
        fused_y = row.get("fused_y_m")
        if fused_x is not None and fused_y is not None:
            final_x = float(fused_x)
            final_y = float(fused_y)
        elif deployed_visual_active:
            final_x = candidate_x
            final_y = radar_y
        else:
            final_x = radar_x
            final_y = radar_y

        if deployed_visual_active:
            visual_admitted += 1
            admitted_abs_delta_x.append(abs(final_x - radar_x))
            if int(float(row.get("selected_index", 1))) == 0:
                selected_noop += 1
        else:
            policy_identity += 1

        delta_x = abs(final_x - radar_x)
        final_abs_delta_x.append(delta_x)
        radar_error = math.hypot(radar_x - target_x, radar_y - target_y)
        final_error = math.hypot(final_x - target_x, final_y - target_y)
        error_delta = final_error - radar_error
        if error_delta < -error_tolerance_m:
            improved += 1
        elif error_delta > error_tolerance_m:
            harmed += 1
        else:
            unchanged += 1

    frames = len(rows)
    return {
        "frames": frames,
        "policy_identity": {
            "frames": policy_identity,
            "rate": policy_identity / frames if frames else 0.0,
        },
        "visual_admitted": {
            "frames": visual_admitted,
            "rate": visual_admitted / frames if frames else 0.0,
            "selected_noop_frames": selected_noop,
            "selected_noop_rate_within_admitted": selected_noop / visual_admitted if visual_admitted else 0.0,
        },
        "health_diagnostic": {
            "active_frames": health_diagnostic_active,
            "active_rate": health_diagnostic_active / frames if frames else 0.0,
        },
        "final_outcome": {
            "improved_frames": improved,
            "improved_rate": improved / frames if frames else 0.0,
            "harmed_frames": harmed,
            "harmed_rate": harmed / frames if frames else 0.0,
            "unchanged_frames": unchanged,
            "unchanged_rate": unchanged / frames if frames else 0.0,
        },
        "final_update": {
            "near_noop_tolerance_m": noop_tolerance_m,
            "near_noop_frames": sum(value <= noop_tolerance_m for value in final_abs_delta_x),
            "near_noop_rate": (
                sum(value <= noop_tolerance_m for value in final_abs_delta_x) / frames if frames else 0.0
            ),
            "median_abs_delta_x_m": statistics.median(final_abs_delta_x) if final_abs_delta_x else 0.0,
            "p95_abs_delta_x_m": _percentile(final_abs_delta_x, 95),
            "max_abs_delta_x_m": max(final_abs_delta_x, default=0.0),
            "admitted_median_abs_delta_x_m": (
                statistics.median(admitted_abs_delta_x) if admitted_abs_delta_x else 0.0
            ),
            "admitted_p95_abs_delta_x_m": _percentile(admitted_abs_delta_x, 95),
            "admitted_max_abs_delta_x_m": max(admitted_abs_delta_x, default=0.0),
            "threshold_counts": {
                "le_0p001m": sum(value <= 0.001 for value in final_abs_delta_x),
                "le_0p01m": sum(value <= 0.01 for value in final_abs_delta_x),
                "le_0p025m": sum(value <= 0.025 for value in final_abs_delta_x),
                "le_0p05m": sum(value <= 0.05 for value in final_abs_delta_x),
            },
        },
    }


def summarize_selectivity_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    error_tolerance_m: float = 1e-6,
    noop_tolerance_m: float = 1e-3,
) -> dict[str, object]:
    materialized = list(rows)
    summary = _summarize_rows(
        materialized,
        error_tolerance_m=error_tolerance_m,
        noop_tolerance_m=noop_tolerance_m,
    )
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in materialized:
        grouped[str(row["sequence"])].append(row)

    per_sequence: dict[str, dict[str, object]] = {}
    for sequence, sequence_rows in sorted(grouped.items()):
        result = _summarize_rows(
            sequence_rows,
            error_tolerance_m=error_tolerance_m,
            noop_tolerance_m=noop_tolerance_m,
        )
        outcome = result["final_outcome"]
        identity = result["policy_identity"]
        admitted = result["visual_admitted"]
        update = result["final_update"]
        per_sequence[sequence] = {
            "frames": result["frames"],
            "policy_identity_frames": identity["frames"],
            "visual_admitted_frames": admitted["frames"],
            "selected_noop_frames": admitted["selected_noop_frames"],
            "improved_frames": outcome["improved_frames"],
            "harmed_frames": outcome["harmed_frames"],
            "unchanged_frames": outcome["unchanged_frames"],
            "median_abs_delta_x_m": update["median_abs_delta_x_m"],
            "p95_abs_delta_x_m": update["p95_abs_delta_x_m"],
        }
    summary["per_sequence"] = per_sequence
    summary["definitions"] = {
        "policy_identity": "deployed accepted gate is false; output x equals expert-radar x exactly",
        "visual_admitted": "deployed accepted gate is true; output x uses the local visual posterior",
        "health_diagnostic": "assist_enabled is retained only as a diagnostic; it never selects deployed output",
        "final_outcome": "Euclidean error of deployed output compared with expert-radar anchor",
        "near_noop": f"absolute deployed lateral correction <= {noop_tolerance_m:g} m",
        "error_tolerance_m": error_tolerance_m,
    }
    return summary



