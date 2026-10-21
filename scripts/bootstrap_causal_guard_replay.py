#!/usr/bin/env python3
"""Compute a sequence bootstrap directly from final causal guard outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from bootstrap_sequences import bootstrap_sequence_metrics
from rvfusion.causal_guard_replay import (
    CausalGuardReplayFrame,
    GuardedReplayFrame,
    bootstrap_rows_from_guarded_replay,
)


def _as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected True or False, received {value!r}")


def _read_replay(path: Path) -> list[GuardedReplayFrame]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"causal replay is empty: {path}")
    required = {
        "sequence",
        "frame_id",
        "policy_admitted",
        "radar_x_m",
        "radar_y_m",
        "candidate_x_m",
        "target_x_m",
        "target_y_m",
        "score",
        "signed_innovation",
        "visual_accepted",
        "fused_x_m",
        "fused_y_m",
        "innovation_ema",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"causal replay missing columns: {sorted(missing)}")
    replay: list[GuardedReplayFrame] = []
    for row in rows:
        frame = CausalGuardReplayFrame(
            sequence=str(row["sequence"]),
            frame_id=int(row["frame_id"]),
            policy_admitted=_as_bool(row["policy_admitted"]),
            radar_x_m=float(row["radar_x_m"]),
            radar_y_m=float(row["radar_y_m"]),
            candidate_x_m=float(row["candidate_x_m"]),
            target_x_m=float(row["target_x_m"]),
            target_y_m=float(row["target_y_m"]),
            score=float(row["score"]),
            signed_innovation=float(row["signed_innovation"]),
        )
        replay.append(
            GuardedReplayFrame(
                **frame.__dict__,
                visual_accepted=_as_bool(row["visual_accepted"]),
                fused_x_m=float(row["fused_x_m"]),
                fused_y_m=float(row["fused_y_m"]),
                innovation_ema=float(row["innovation_ema"]),
            )
        )
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-replay-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    rows = bootstrap_rows_from_guarded_replay(_read_replay(args.causal_replay_csv))
    result = bootstrap_sequence_metrics(
        rows,
        repetitions=args.repetitions,
        seed=args.seed,
    )
    result["source_causal_replay_csv"] = str(args.causal_replay_csv.resolve())
    result["source_causal_replay_sha256"] = hashlib.sha256(
        args.causal_replay_csv.read_bytes()
    ).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
