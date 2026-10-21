from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.route_closure import ROUTES, assert_development_only_paths
from radar_bev.route_transitions import (
    delayed_route_for_frame,
    transition_source_frame,
)


FRAME_PATTERN = re.compile(r"(seq_\d{3}).*?frame_(\d{4})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Development-only no-reset route-transition episodes"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sequence-ledger", type=Path, required=True)
    parser.add_argument("--policy-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-frame", type=int, default=150)
    parser.add_argument("--frames-per-episode", type=int, default=300)
    parser.add_argument("--availability-delay-frames", type=int, default=32)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--fixed-route", choices=ROUTES, default="open_outdoor_weak_return")
    return parser.parse_args()


def _prediction_key(row: dict[str, object]) -> tuple[str, int]:
    match = FRAME_PATTERN.search(str(row.get("image", "")).replace("\\", "/"))
    if match is None:
        raise ValueError(f"cannot parse policy image: {row.get('image')}")
    return match.group(1).lower(), int(match.group(2))


def _load_predictions(path: Path) -> dict[tuple[str, int], str]:
    predictions: dict[tuple[str, int], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prediction = row.get("prediction")
        if not isinstance(prediction, dict):
            raise ValueError("policy prediction must be an object")
        route = str(prediction.get("expert_route", "")).strip().lower()
        if route not in ROUTES:
            route = "unknown"
        key = _prediction_key(row)
        if key in predictions:
            raise ValueError(f"duplicate policy prediction: {key}")
        predictions[key] = route
    return predictions


def _load_development_manifest(
    path: Path,
) -> tuple[list[str], dict[str, dict[int, dict[str, str]]]]:
    grouped: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest has no header")
        fields = list(reader.fieldnames)
        for row in reader:
            if row["split"].strip().lower() != "development val":
                continue
            grouped[row["sequence"]][int(row["frame_id"])] = row
    if not grouped:
        raise ValueError("Development manifest is empty")
    return fields, dict(grouped)


def main() -> None:
    args = parse_args()
    assert_development_only_paths(
        args.manifest, args.sequence_ledger, args.policy_predictions, args.output_dir
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if not 1 <= args.boundary_frame < args.frames_per_episode:
        raise ValueError("boundary frame must be inside the episode")
    second_length = args.frames_per_episode - args.boundary_frame
    if second_length > args.boundary_frame:
        raise ValueError("source halves require more second-route frames than first-route frames")
    ledger = json.loads(args.sequence_ledger.read_text(encoding="utf-8"))
    fields, manifest = _load_development_manifest(args.manifest)
    policy = _load_predictions(args.policy_predictions)
    by_route = {
        route: sorted(
            sequence
            for sequence in manifest
            if ledger.get(sequence) == route
            and len(manifest[sequence]) >= args.boundary_frame
        )
        for route in ROUTES
    }
    missing = [route for route, sequences in by_route.items() if not sequences]
    if missing:
        raise ValueError(f"no Development source sequences for routes: {missing}")
    pairs = [(left, right) for left in ROUTES for right in ROUTES if left != right]
    output_manifest: list[dict[str, str]] = []
    schedule_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    for episode_index, (left_route, right_route) in enumerate(pairs):
        left_sequence = by_route[left_route][episode_index % len(by_route[left_route])]
        right_sequence = by_route[right_route][
            (episode_index // len(ROUTES) + episode_index) % len(by_route[right_route])
        ]
        alias = f"route_transition_{episode_index:03d}"
        captures: dict[int, str] = {}
        oracle_captures: dict[int, str] = {}
        for capture in range(0, args.frames_per_episode, args.hold_frames):
            source_half, source_frame = transition_source_frame(
                capture, boundary_frame=args.boundary_frame
            )
            source_sequence = left_sequence if source_half == 0 else right_sequence
            captures[capture] = policy.get((source_sequence, source_frame), "unknown")
            oracle_captures[capture] = left_route if source_half == 0 else right_route
        for frame_id in range(args.frames_per_episode):
            source_half, source_frame = transition_source_frame(
                frame_id, boundary_frame=args.boundary_frame
            )
            source_sequence = left_sequence if source_half == 0 else right_sequence
            source = manifest[source_sequence].get(source_frame)
            if source is None:
                raise ValueError(
                    f"source manifest lacks {source_sequence}:{source_frame}"
                )
            row = dict(source)
            row["sequence"] = alias
            row["frame_id"] = str(frame_id)
            row["split"] = "Development Val"
            output_manifest.append(row)
            learned_route, capture = delayed_route_for_frame(
                frame_id,
                captures,
                availability_delay_frames=args.availability_delay_frames,
                hold_frames=args.hold_frames,
            )
            oracle_delayed, oracle_capture = delayed_route_for_frame(
                frame_id,
                oracle_captures,
                availability_delay_frames=args.availability_delay_frames,
                hold_frames=args.hold_frames,
            )
            if capture != oracle_capture:
                raise AssertionError("learned and oracle schedules must share capture time")
            if capture is not None and capture + args.availability_delay_frames > frame_id:
                raise AssertionError("transition schedule consumed future policy evidence")
            actual_route = left_route if source_half == 0 else right_route
            schedule_rows.append(
                {
                    "sequence": alias,
                    "frame_id": frame_id,
                    "source_sequence": source_sequence,
                    "source_frame_id": source_frame,
                    "actual_route": actual_route,
                    "learned_delayed_route": learned_route,
                    "oracle_delayed_route": oracle_delayed,
                    "oracle_instant_route": actual_route,
                    "best_fixed_route": args.fixed_route,
                    "policy_capture_frame_id": "" if capture is None else capture,
                    "policy_available_delay_frames": args.availability_delay_frames,
                }
            )
        episode_rows.append(
            {
                "sequence": alias,
                "source_route_before": left_route,
                "source_route_after": right_route,
                "source_sequence_before": left_sequence,
                "source_sequence_after": right_sequence,
                "boundary_frame": args.boundary_frame,
                "expected_first_after_route_frame": (
                    args.boundary_frame + args.availability_delay_frames
                ),
                "state_reset_at_boundary": False,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "route_transition_manifest.csv"
    with manifest_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_manifest)
    for name, rows in (
        ("route_transition_schedule.csv", schedule_rows),
        ("route_transition_episodes.csv", episode_rows),
    ):
        path = args.output_dir / name
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report = {
        "protocol": "IF-only Development controlled no-reset route transition",
        "locked_test_used": False,
        "episodes": len(episode_rows),
        "ordered_route_pairs": len(pairs),
        "frames_per_episode": args.frames_per_episode,
        "boundary_frame": args.boundary_frame,
        "availability_delay_frames": args.availability_delay_frames,
        "hold_frames": args.hold_frames,
        "radar_clip_crosses_boundary": True,
        "causal_state_reset_at_boundary": False,
        "source_sha256": {
            "manifest": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "sequence_ledger": hashlib.sha256(args.sequence_ledger.read_bytes()).hexdigest(),
            "policy_predictions": hashlib.sha256(args.policy_predictions.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "route_transition_protocol.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
