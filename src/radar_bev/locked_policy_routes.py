"""Read the frozen causal policy trace used by locked route-expert replay."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


_SEQUENCE_PATTERN = re.compile(r"seq_\d+")


@dataclass(frozen=True)
class RouteSegment:
    start_frame: int
    end_frame: int
    route: str


@dataclass(frozen=True)
class PolicyRouteSchedule:
    by_sequence: dict[str, tuple[RouteSegment, ...]]

    def route_for(self, sequence: str, frame_id: int) -> str:
        for segment in self.by_sequence.get(sequence, ()):
            if segment.start_frame <= frame_id <= segment.end_frame:
                return segment.route
        raise KeyError(f"no frozen policy segment covers {sequence} frame={frame_id}")


def _sequence_from_record(record: dict[str, object]) -> str:
    for value in (record.get("id"), record.get("image")):
        match = _SEQUENCE_PATTERN.search(str(value))
        if match is not None:
            return match.group(0)
    raise ValueError("policy trace record has no canonical seq_* identifier")


def load_policy_route_schedule(path: str | Path) -> PolicyRouteSchedule:
    grouped: dict[str, list[RouteSegment]] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if not isinstance(record, dict):
            raise ValueError("policy trace line must be an object")
        prediction = record.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError("policy trace record has no prediction object")
        route = str(prediction.get("expert_route", "")).strip().lower()
        if not route:
            raise ValueError("policy trace record has no expert route")
        start = int(record["policy_segment_start_frame"])
        end = int(record["policy_segment_end_frame"])
        if start < 0 or end < start:
            raise ValueError("policy segment bounds are invalid")
        sequence = _sequence_from_record(record)
        grouped.setdefault(sequence, []).append(RouteSegment(start, end, route))

    if not grouped:
        raise ValueError("policy trace is empty")
    normalized: dict[str, tuple[RouteSegment, ...]] = {}
    for sequence, segments in grouped.items():
        ordered = tuple(sorted(segments, key=lambda item: item.start_frame))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_frame <= previous.end_frame:
                raise ValueError(f"policy segments overlap for {sequence}")
        normalized[sequence] = ordered
    return PolicyRouteSchedule(normalized)
