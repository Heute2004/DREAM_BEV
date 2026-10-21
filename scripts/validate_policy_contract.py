"""Validate that a slow policy never switches inside its causal hold window."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


FRAME_PATTERN = re.compile(r"(seq_\d{3}).*?frame_(\d{4})", re.IGNORECASE)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"accepted must be boolean, got {value!r}")


def _route(value: object) -> str:
    route = str(value).strip().lower()
    if not route:
        raise ValueError("route must be a non-empty string")
    return route


def _policy_sequence(value: object) -> str:
    match = FRAME_PATTERN.search(str(value).replace("\\", "/"))
    if match is None:
        raise ValueError(f"cannot parse sequence from policy image: {value!r}")
    return match.group(1).lower()


def _policy_admits(prediction: Mapping[str, object], *, trust_policy: str) -> bool:
    route = _route(prediction.get("expert_route", "unknown"))
    if route in {"unknown", "corner_nlos"}:
        return False
    reliability = str(prediction.get("visual_reliability_prior", "unknown")).strip().lower()
    condition = str(prediction.get("visual_condition", "unknown")).strip().lower()
    if trust_policy == "soft":
        reliability_cap = {"usable": 1.0, "cautious": 0.45, "unreliable": 0.0, "unknown": 0.25}
        condition_cap = {
            "normal": 1.0,
            "normal_visibility": 1.0,
            "low_light": 0.25,
            "low_light_extreme": 0.10,
            "exposure": 0.15,
            "fog": 0.35,
            "dense_fog": 0.20,
            "lens_droplets": 0.30,
            "severe_blur": 0.20,
            "rain_or_wet": 0.40,
            "mixed_degradation": 0.15,
            "unknown": 0.25,
        }
        return min(
            reliability_cap.get(reliability, reliability_cap["unknown"]),
            condition_cap.get(condition, condition_cap["unknown"]),
        ) > 0.0
    if trust_policy == "strict_usable":
        return reliability == "usable" and condition in {"normal", "normal_visibility"}
    if trust_policy == "normal_nonunreliable":
        return reliability != "unreliable" and condition in {"normal", "normal_visibility"}
    if trust_policy == "normal_or_rain_nonunreliable":
        return reliability != "unreliable" and condition in {
            "normal",
            "normal_visibility",
            "rain_or_wet",
        }
    raise ValueError(f"unknown trust policy: {trust_policy}")


def validate_deployed_policy_rows(
    policy_rows: Iterable[dict[str, Any]],
    audit_rows: Iterable[dict[str, Any]],
    *,
    trust_policy: str = "soft",
) -> dict[str, int | str]:
    """Verify every deployed route and gate against materialized causal policy."""

    expected: dict[tuple[str, int], tuple[str, bool]] = {}
    for row in policy_rows:
        sequence = _policy_sequence(row.get("image", ""))
        start = int(row["policy_available_frame_id"])
        end = int(row["policy_segment_end_frame_id"])
        if start < 0 or end < start:
            raise ValueError(f"invalid policy interval for {sequence}: {start}..{end}")
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            raise ValueError("materialized policy prediction must be an object")
        state = (_route(prediction.get("expert_route", "unknown")), _policy_admits(prediction, trust_policy=trust_policy))
        for frame_id in range(start, end + 1):
            key = (sequence, frame_id)
            if key in expected:
                raise ValueError(f"overlapping materialized policy interval: {sequence}:{frame_id}")
            expected[key] = state
    if not expected:
        raise ValueError("materialized policy is empty")

    audited = 0
    identity = 0
    admitted = 0
    for row in audit_rows:
        sequence = str(row.get("sequence", "")).strip().lower()
        frame_id = int(row["frame_id"])
        expected_state = expected.get((sequence, frame_id))
        if expected_state is None:
            raise ValueError(f"audit frame has no materialized policy interval: {sequence}:{frame_id}")
        actual_state = (_route(row["route"]), _as_bool(row["accepted"]))
        if actual_state != expected_state:
            raise ValueError(
                "deployed audit does not match materialized policy: "
                f"{sequence}:{frame_id} expected={expected_state} actual={actual_state}"
            )
        audited += 1
        admitted += int(actual_state[1])
        identity += int(not actual_state[1])
    if audited != len(expected):
        raise ValueError(
            f"policy/audit coverage mismatch: policy={len(expected)} audit={audited}"
        )
    return {
        "policy_frame_count": len(expected),
        "audited_frame_count": audited,
        "policy_identity_frames": identity,
        "policy_admitted_frames": admitted,
        "trust_policy": trust_policy,
    }


def validate_segment_policy_rows(
    rows: Iterable[dict[str, Any]], *, hold_frames: int
) -> dict[str, int]:
    """Return an audit report or fail when a route/policy changes within a hold."""
    if hold_frames < 1:
        raise ValueError("hold_frames must be positive")

    grouped: dict[str, dict[int, set[tuple[str, bool]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    row_count = 0
    for row in rows:
        sequence = str(row["sequence"]).strip().lower()
        frame_id = int(row["frame_id"])
        if not sequence or frame_id < 0:
            raise ValueError(f"invalid sequence/frame: {sequence!r}, {frame_id}")
        state = (_route(row["route"]), _as_bool(row["accepted"]))
        grouped[sequence][frame_id // hold_frames].add(state)
        row_count += 1

    if not grouped:
        raise ValueError("policy audit is empty")

    mixed: list[tuple[str, int, list[tuple[str, bool]]]] = []
    mixed_route_count = 0
    mixed_acceptance_count = 0
    transitions = 0
    segment_count = 0
    for sequence, by_segment in grouped.items():
        states: list[tuple[int, tuple[str, bool]]] = []
        for segment, values in sorted(by_segment.items()):
            segment_count += 1
            if len(values) != 1:
                mixed.append((sequence, segment, sorted(values)))
                if len({route for route, _ in values}) > 1:
                    mixed_route_count += 1
                if len({accepted for _, accepted in values}) > 1:
                    mixed_acceptance_count += 1
                continue
            states.append((segment, next(iter(values))))
        transitions += sum(
            previous_state != state
            for (_, previous_state), (_, state) in zip(states, states[1:])
        )

    if mixed:
        preview = "; ".join(
            f"{sequence}/segment={segment}/states={states}"
            for sequence, segment, states in mixed[:5]
        )
        raise ValueError(
            "framewise route/policy switch detected inside a causal hold: "
            + preview
        )

    return {
        "row_count": row_count,
        "sequence_count": len(grouped),
        "hold_frames": hold_frames,
        "segment_count": segment_count,
        "mixed_segment_count": 0,
        "mixed_route_segment_count": 0,
        "mixed_acceptance_segment_count": 0,
        "segment_state_transitions": transitions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that accepted RVFusion policy state is constant in each hold window."
    )
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--policy-jsonl", type=Path, required=True)
    parser.add_argument(
        "--trust-policy",
        choices=(
            "soft",
            "strict_usable",
            "normal_nonunreliable",
            "normal_or_rain_nonunreliable",
        ),
        default="soft",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    with args.policy_jsonl.open(encoding="utf-8") as handle:
        policy_rows = [json.loads(line) for line in handle if line.strip()]
    with args.audit_csv.open(encoding="utf-8", newline="") as handle:
        report = validate_deployed_policy_rows(
            policy_rows,
            csv.DictReader(handle),
            trust_policy=args.trust_policy,
        )
    payload = json.dumps(report, indent=2) + "\n"
    if args.output_json is not None:
        if args.output_json.exists():
            raise FileExistsError(f"refusing to overwrite {args.output_json}")
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()



