#!/usr/bin/env python3
"""Select a causal guard threshold from Train-only frozen evidence replays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from replay_causal_cross_modal_guard import (
    _frames_from_policy_replay_and_evidence,
    _metrics,
)
from rvfusion.causal_guard_replay import replay_guarded_frames
from rvfusion.guard_calibration import CalibrationRow, choose_guard_threshold


def _forbid_locked(*paths: Path) -> None:
    for path in paths:
        normalized = str(path).lower().replace("\\", "/")
        if "locked" in normalized or "test" in normalized:
            raise ValueError(f"Train-only guard calibration rejects path: {path}")


def _require_train_frames(frames, manifest: Path) -> None:
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        split_by_frame = {
            (str(row["sequence"]), int(row["frame_id"])): row["split"]
            for row in csv.DictReader(handle)
        }
    keys = {(frame.sequence, frame.frame_id) for frame in frames}
    missing = keys - set(split_by_frame)
    if missing:
        raise ValueError(f"manifest is missing replay frames: {sorted(missing)[:3]}")
    non_train = [key for key in keys if split_by_frame[key] != "Train"]
    if non_train:
        raise ValueError(
            "guard calibration requires only Train frames; found "
            f"{split_by_frame[non_train[0]]} at {non_train[0]}"
        )


def _replay_metrics(
    frames,
    *,
    ema_alpha: float,
    ema_threshold: float,
) -> tuple[dict[str, float], float]:
    replay = replay_guarded_frames(
        frames,
        ema_alpha=ema_alpha,
        ema_threshold=ema_threshold,
    )
    acceptance = sum(frame.visual_accepted for frame in replay) / len(replay)
    return _metrics(replay, fused=True), acceptance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-causal-replay", type=Path, required=True)
    parser.add_argument("--clean-evidence-audit", type=Path, required=True)
    parser.add_argument("--bias-evidence-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ema-alpha", type=float, default=0.10)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.50, 0.60, 0.70, 0.80, 0.90, 1.00),
    )
    parser.add_argument("--minimum-clean-gain", type=float, default=0.01)
    args = parser.parse_args()

    _forbid_locked(
        args.policy_causal_replay,
        args.clean_evidence_audit,
        args.bias_evidence_audit,
        args.manifest,
        args.output_dir,
    )
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite calibration: {args.output_dir}")
    if not args.thresholds or any(threshold <= 0.0 for threshold in args.thresholds):
        raise ValueError("thresholds must be non-empty positive values")

    clean_frames = _frames_from_policy_replay_and_evidence(
        args.policy_causal_replay,
        args.clean_evidence_audit,
    )
    bias_frames = _frames_from_policy_replay_and_evidence(
        args.policy_causal_replay,
        args.bias_evidence_audit,
    )
    if len(clean_frames) != len(bias_frames):
        raise ValueError("clean and bias replays must have equal frame counts")
    _require_train_frames(clean_frames, args.manifest)
    _require_train_frames(bias_frames, args.manifest)
    radar_clean = _metrics(clean_frames, fused=False)
    radar_bias = _metrics(bias_frames, fused=False)

    rows: list[dict[str, float]] = []
    for threshold in sorted(set(args.thresholds)):
        clean_metrics, clean_acceptance = _replay_metrics(
            clean_frames,
            ema_alpha=args.ema_alpha,
            ema_threshold=threshold,
        )
        bias_metrics, bias_acceptance = _replay_metrics(
            bias_frames,
            ema_alpha=args.ema_alpha,
            ema_threshold=threshold,
        )
        rows.append(
            {
                "threshold": threshold,
                "clean_ap30": clean_metrics["ap_0p3m"],
                "bias_ap30": bias_metrics["ap_0p3m"],
                "clean_p90_m": clean_metrics["p90_location_error_m"],
                "bias_p90_m": bias_metrics["p90_location_error_m"],
                "clean_visual_acceptance": clean_acceptance,
                "bias_visual_acceptance": bias_acceptance,
            }
        )

    selected = choose_guard_threshold(
        [
            CalibrationRow(
                threshold=row["threshold"],
                clean_ap30=row["clean_ap30"],
                biased_ap30=row["bias_ap30"],
            )
            for row in rows
        ],
        radar_clean_ap30=radar_clean["ap_0p3m"],
        radar_biased_ap30=radar_bias["ap_0p3m"],
        minimum_clean_gain=args.minimum_clean_gain,
    )
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "threshold_sweep.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "scope": "Train-only frozen causal guard calibration",
        "ema_alpha": args.ema_alpha,
        "thresholds": [row["threshold"] for row in rows],
        "minimum_clean_gain": args.minimum_clean_gain,
        "radar_clean_ap30": radar_clean["ap_0p3m"],
        "radar_bias_ap30": radar_bias["ap_0p3m"],
        "selected_threshold": None if selected is None else selected.threshold,
        "rows": rows,
        "inputs": {
            "policy_causal_replay": str(args.policy_causal_replay),
            "clean_evidence_audit": str(args.clean_evidence_audit),
            "bias_evidence_audit": str(args.bias_evidence_audit),
            "manifest": str(args.manifest),
        },
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
