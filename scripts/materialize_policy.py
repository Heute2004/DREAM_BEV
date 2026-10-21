"""Materialize a delayed, segment-held RGB policy timeline.

The policy consumes only a current RGB frame. A completed prediction becomes
available after a measured availability delay and remains fixed for one full
radar segment. The initial warm-up always uses the radar-only identity policy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SOURCE_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from condition_policy.contract import safe_dispatch_prediction


FRAME_PATTERN = re.compile(r"(seq_\d{3}).*?frame_(\d{4})", re.IGNORECASE)


def _sequence_frame(image: str) -> tuple[str, int]:
    match = FRAME_PATTERN.search(str(image).replace("\\", "/"))
    if match is None:
        raise ValueError(f"cannot parse sequence/frame from image path: {image}")
    return match.group(1).lower(), int(match.group(2))


def _with_frame(image: str, frame_id: int) -> str:
    replaced, count = re.subn(
        r"frame_\d{4}",
        f"frame_{int(frame_id):04d}",
        str(image).replace("\\", "/"),
        count=1,
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise ValueError(f"cannot write activation frame into image path: {image}")
    return replaced


def _prediction(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def materialize_delayed_policy_rows(
    rows: Iterable[dict[str, object]],
    *,
    frames_per_sequence: int,
    input_step_frames: int,
    capture_interval_frames: int,
    availability_delay_frames: int,
    hold_frames: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return identity warm-up plus causally available full-segment policies."""
    values = (
        frames_per_sequence,
        input_step_frames,
        capture_interval_frames,
        availability_delay_frames,
        hold_frames,
    )
    if any(value < 1 for value in values):
        raise ValueError("all frame intervals must be positive")
    if capture_interval_frames % input_step_frames:
        raise ValueError("capture interval must align with policy input spacing")

    grouped: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        sequence, frame_id = _sequence_frame(str(row.get("image", "")))
        if frame_id in grouped[sequence]:
            raise ValueError(f"duplicate RGB policy record: {sequence}:{frame_id}")
        grouped[sequence][frame_id] = row
    if not grouped:
        raise ValueError("policy prediction input is empty")

    expected_inputs = set(range(0, frames_per_sequence, input_step_frames))
    output: list[dict[str, object]] = []
    invalid_predictions = 0
    segment_count = 0
    for sequence in sorted(grouped):
        by_frame = grouped[sequence]
        if set(by_frame) != expected_inputs:
            missing = sorted(expected_inputs.difference(by_frame))
            unexpected = sorted(set(by_frame).difference(expected_inputs))
            raise ValueError(
                f"unexpected policy input coverage for {sequence}: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        first = by_frame[0]
        warmup_end = min(availability_delay_frames, frames_per_sequence) - 1
        output.append(
            {
                "id": f"{first.get('id', sequence)}__identity_warmup",
                "image": _with_frame(str(first["image"]), 0),
                "prediction": safe_dispatch_prediction(None),
                "policy_capture_frame_id": None,
                "policy_available_frame_id": 0,
                "policy_segment_end_frame_id": warmup_end,
                "policy_source": "identity_warmup",
            }
        )
        for segment_start in range(availability_delay_frames, frames_per_sequence, hold_frames):
            capture_frame = ((segment_start - availability_delay_frames) // hold_frames) * capture_interval_frames
            source = by_frame[capture_frame]
            raw_prediction = _prediction(source.get("prediction"))
            deployed_prediction = safe_dispatch_prediction(raw_prediction)
            invalid_predictions += int(raw_prediction != deployed_prediction)
            output.append(
                {
                    "id": f"{source.get('id', sequence)}__available_{segment_start:04d}",
                    "image": _with_frame(str(source["image"]), segment_start),
                    "prediction": deployed_prediction,
                    "policy_capture_frame_id": capture_frame,
                    "policy_available_frame_id": segment_start,
                    "policy_segment_end_frame_id": min(
                        segment_start + hold_frames - 1,
                        frames_per_sequence - 1,
                    ),
                    "policy_source": "current_rgb_completed_async_call",
                }
            )
            segment_count += 1

    return output, {
        "sequence_count": len(grouped),
        "input_rows": sum(len(rows_by_frame) for rows_by_frame in grouped.values()),
        "output_rows": len(output),
        "frames_per_sequence": frames_per_sequence,
        "input_step_frames": input_step_frames,
        "capture_interval_frames": capture_interval_frames,
        "availability_delay_frames": availability_delay_frames,
        "hold_frames": hold_frames,
        "warmup_policy": "radar_only_identity",
        "causal_capture_only": True,
        "invalid_predictions_fail_closed": invalid_predictions,
        "active_policy_segments": segment_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a delayed segment-held RGB condition policy."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=300)
    parser.add_argument("--input-step-frames", type=int, default=5)
    parser.add_argument("--capture-interval-frames", type=int, default=30)
    parser.add_argument("--availability-delay-frames", type=int, default=32)
    parser.add_argument("--hold-frames", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("refusing to overwrite delayed policy outputs")

    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    output, audit = materialize_delayed_policy_rows(
        rows,
        frames_per_sequence=args.frames_per_sequence,
        input_step_frames=args.input_step_frames,
        capture_interval_frames=args.capture_interval_frames,
        availability_delay_frames=args.availability_delay_frames,
        hold_frames=args.hold_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()



