"""Metric utilities for route-only counterfactual evaluations."""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping

import torch

from radar_bev.evaluation import evaluate_single_person_predictions


_IMAGE_PATTERN = re.compile(r"(seq_\d{3}).*?frame_(\d{4})", re.IGNORECASE)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _trace_timeline(
    records: Iterable[Mapping[str, object]],
) -> dict[str, tuple[list[int], list[str]]]:
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for record in records:
        image = str(record.get("image", "")).replace("\\", "/")
        match = _IMAGE_PATTERN.search(image)
        prediction = record.get("prediction")
        if match is None or not isinstance(prediction, Mapping):
            raise ValueError("reference policy trace has an invalid policy record")
        route = prediction.get("expert_route")
        if not isinstance(route, str):
            raise ValueError("reference policy trace is missing expert_route")
        frame = int(record.get("policy_segment_start_frame", match.group(2)))
        grouped[match.group(1).lower()].append((frame, route))

    timeline: dict[str, tuple[list[int], list[str]]] = {}
    for sequence, values in grouped.items():
        values.sort(key=lambda item: item[0])
        starts, routes = zip(*values)
        timeline[sequence] = (list(starts), list(routes))
    return timeline


def _lookup_route(
    timeline: dict[str, tuple[list[int], list[str]]], sequence: str, frame_id: int
) -> str:
    starts, routes = timeline[sequence.lower()]
    index = bisect.bisect_right(starts, frame_id) - 1
    if index < 0:
        raise ValueError(f"no causal policy record for {sequence} frame {frame_id}")
    return routes[index]


def _evaluate_grouped_rows(
    grouped: Mapping[str, list[Mapping[str, object]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for route, rows in sorted(grouped.items()):
        score = torch.tensor([float(row["radar_score"]) for row in rows])
        target = torch.tensor(
            [
                (float(row["target_x_m"]), float(row["target_y_m"]))
                for row in rows
            ]
        )
        prediction = torch.tensor(
            [
                (
                    float(row["candidate_x_m"])
                    if _as_bool(row["accepted"])
                    else float(row["radar_x_m"]),
                    float(row["radar_y_m"]),
                )
                for row in rows
            ]
        )
        result[route] = {
            "frames": float(len(rows)),
            **evaluate_single_person_predictions(
                score,
                prediction,
                target,
                distance_thresholds=(0.25, 0.30, 0.50),
            ),
        }
    return result


def evaluate_audit_by_reference_route(
    audit_rows: Iterable[Mapping[str, object]],
    reference_trace: Iterable[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    """Score deployed fused outputs against the unmodified trace's route groups.

    The evaluator audit stores the candidate coordinate even for a denied frame.
    This function reconstructs the deployed coordinate by using the candidate
    x only when the frozen decoder's `accepted` flag is true; range always
    remains the radar coordinate. Warm-up/unknown reference-route frames are
    excluded from the five-mechanism matrix.
    """

    timeline = _trace_timeline(reference_trace)
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in audit_rows:
        sequence = str(row["sequence"])
        frame_id = int(row["frame_id"])
        route = _lookup_route(timeline, sequence, frame_id)
        if route != "unknown":
            grouped[route].append(row)
    return _evaluate_grouped_rows(grouped)


def evaluate_audit_by_ledger_route(
    audit_rows: Iterable[Mapping[str, object]],
    reference_trace: Iterable[Mapping[str, object]],
    sequence_routes: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """Score active policy frames by an independent frozen sequence-route ledger.

    The canonical trace is used only to exclude causal identity warm-up frames.
    Each active frame is grouped by the externally supplied frozen route ledger,
    rather than by the policy route emitted during the replay.
    """

    timeline = _trace_timeline(reference_trace)
    normalized_ledger = {
        str(sequence).lower(): str(route) for sequence, route in sequence_routes.items()
    }
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in audit_rows:
        sequence = str(row["sequence"]).lower()
        frame_id = int(row["frame_id"])
        if _lookup_route(timeline, sequence, frame_id) == "unknown":
            continue
        try:
            route = normalized_ledger[sequence]
        except KeyError as error:
            raise ValueError(
                f"sequence missing from frozen route ledger: {sequence}"
            ) from error
        grouped[route].append(row)
    return _evaluate_grouped_rows(grouped)
