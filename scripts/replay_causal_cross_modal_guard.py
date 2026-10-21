"""Replay a frozen RVFusion candidate with a causal consistency guard."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from radar_bev.evaluation import evaluate_single_person_predictions
from rvfusion.causal_guard_replay import (
    CausalGuardReplayFrame,
    override_policy_admission,
    replay_guarded_frames,
)


def _as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected True or False, received {value!r}")


def _read_rows(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"audit is empty: {path}")
    result: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (str(row["sequence"]), int(row["frame_id"]))
        if key in result:
            raise ValueError(f"duplicate audit record: {key}")
        result[key] = row
    return result


def _require_close(
    policy: dict[str, str],
    evidence: dict[str, str],
    key: str,
    *,
    tolerance: float = 1e-6,
) -> None:
    if abs(float(policy[key]) - float(evidence[key])) > tolerance:
        raise ValueError(f"policy/evidence mismatch for {key}")


def _joined_frames(
    policy_rows: dict[tuple[str, int], dict[str, str]],
    evidence_rows: dict[tuple[str, int], dict[str, str]],
) -> list[CausalGuardReplayFrame]:
    if set(policy_rows) != set(evidence_rows):
        raise ValueError("policy and evidence audits must cover identical frames")
    result: list[CausalGuardReplayFrame] = []
    for key in sorted(policy_rows):
        policy = policy_rows[key]
        evidence = evidence_rows[key]
        for field in ("target_x_m", "target_y_m", "radar_x_m", "radar_y_m", "candidate_score"):
            _require_close(policy, evidence, field)
        innovation = float(evidence["cross_modal_innovation"])
        visual_minus_radar = (
            float(evidence["calibrated_visual_x_m"])
            - float(evidence["radar_x_m"])
        )
        signed_innovation = innovation if visual_minus_radar >= 0.0 else -innovation
        result.append(
            CausalGuardReplayFrame(
                sequence=key[0],
                frame_id=key[1],
                policy_admitted=_as_bool(policy["accepted"]),
                radar_x_m=float(policy["radar_x_m"]),
                radar_y_m=float(policy["radar_y_m"]),
                candidate_x_m=float(evidence["candidate_x_m"]),
                target_x_m=float(policy["target_x_m"]),
                target_y_m=float(policy["target_y_m"]),
                score=float(policy["candidate_score"]),
                signed_innovation=signed_innovation,
            )
        )
    return result


def _frames_from_causal_replay(path: Path) -> list[CausalGuardReplayFrame]:
    """Reload a previously frozen causal trace without changing its evidence."""

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
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("causal replay lacks required columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"causal replay is empty: {path}")

    result: list[CausalGuardReplayFrame] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["sequence"]), int(row["frame_id"]))
        if key in seen:
            raise ValueError(f"duplicate causal replay record: {key}")
        seen.add(key)
        result.append(
            CausalGuardReplayFrame(
                sequence=key[0],
                frame_id=key[1],
                policy_admitted=_as_bool(row["policy_admitted"]),
                radar_x_m=float(row["radar_x_m"]),
                radar_y_m=float(row["radar_y_m"]),
                candidate_x_m=float(row["candidate_x_m"]),
                target_x_m=float(row["target_x_m"]),
                target_y_m=float(row["target_y_m"]),
                score=float(row["score"]),
                signed_innovation=float(row["signed_innovation"]),
            )
        )
    return sorted(result, key=lambda frame: (frame.sequence, frame.frame_id))


def _frames_from_policy_replay_and_evidence(
    policy_replay: Path,
    evidence_audit: Path,
) -> list[CausalGuardReplayFrame]:
    """Apply changed visual evidence to the exact frozen policy/radar trace."""

    policy_frames = {
        (frame.sequence, frame.frame_id): frame
        for frame in _frames_from_causal_replay(policy_replay)
    }
    evidence_rows = _read_rows(evidence_audit)
    if set(policy_frames) != set(evidence_rows):
        raise ValueError("policy replay and evidence audit must cover identical frames")

    result: list[CausalGuardReplayFrame] = []
    for key in sorted(policy_frames):
        policy = policy_frames[key]
        evidence = evidence_rows[key]
        for field, policy_value in (
            ("target_x_m", policy.target_x_m),
            ("target_y_m", policy.target_y_m),
            ("radar_x_m", policy.radar_x_m),
            ("radar_y_m", policy.radar_y_m),
            ("candidate_score", policy.score),
        ):
            if abs(float(evidence[field]) - policy_value) > 1e-6:
                raise ValueError(f"policy/evidence mismatch for {field} at {key}")
        innovation = float(evidence["cross_modal_innovation"])
        visual_minus_radar = (
            float(evidence["calibrated_visual_x_m"]) - policy.radar_x_m
        )
        result.append(
            CausalGuardReplayFrame(
                sequence=policy.sequence,
                frame_id=policy.frame_id,
                policy_admitted=policy.policy_admitted,
                radar_x_m=policy.radar_x_m,
                radar_y_m=policy.radar_y_m,
                candidate_x_m=float(evidence["candidate_x_m"]),
                target_x_m=policy.target_x_m,
                target_y_m=policy.target_y_m,
                score=policy.score,
                signed_innovation=(
                    innovation if visual_minus_radar >= 0.0 else -innovation
                ),
            )
        )
    return result


def _metrics(frames: list[CausalGuardReplayFrame], fused: bool) -> dict[str, float]:
    scores = torch.tensor([frame.score for frame in frames])
    prediction = torch.tensor(
        [
            [
                frame.fused_x_m if fused else frame.radar_x_m,
                frame.fused_y_m if fused else frame.radar_y_m,
            ]
            for frame in frames
        ]
    )
    target = torch.tensor(
        [[frame.target_x_m, frame.target_y_m] for frame in frames]
    )
    return evaluate_single_person_predictions(
        scores,
        prediction,
        target,
        distance_thresholds=(0.25, 0.30, 0.50, 1.00),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-audit", type=Path)
    parser.add_argument("--evidence-audit", type=Path)
    parser.add_argument(
        "--causal-replay",
        type=Path,
        help="Frozen replay CSV used only for Development mechanism controls.",
    )
    parser.add_argument(
        "--policy-causal-replay",
        type=Path,
        help=(
            "Frozen policy/radar trace paired with a perturbed visual-evidence "
            "audit for Development-only conflict controls."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ema-alpha", type=float, required=True)
    parser.add_argument("--ema-threshold", type=float, required=True)
    parser.add_argument("--instantaneous-threshold", type=float)
    parser.add_argument(
        "--admission-mode",
        choices=("learned", "always-admit", "always-identity"),
        default="learned",
    )
    parser.add_argument("--warmup-frames", type=int, default=32)
    parser.add_argument(
        "--disable-guard",
        action="store_true",
        help="Development-only mechanism ablation; never a deployed result.",
    )
    args = parser.parse_args()

    if args.causal_replay is not None:
        if args.policy_causal_replay is not None:
            parser.error("--causal-replay cannot be combined with --policy-causal-replay")
        if args.policy_audit is not None or args.evidence_audit is not None:
            parser.error("--causal-replay cannot be combined with audit inputs")
        frames = _frames_from_causal_replay(args.causal_replay)
        input_source = str(args.causal_replay)
    elif args.policy_causal_replay is not None:
        if args.policy_audit is not None or args.evidence_audit is None:
            parser.error(
                "--policy-causal-replay requires --evidence-audit and no --policy-audit"
            )
        frames = _frames_from_policy_replay_and_evidence(
            args.policy_causal_replay,
            args.evidence_audit,
        )
        input_source = {
            "policy_causal_replay": str(args.policy_causal_replay),
            "evidence_audit": str(args.evidence_audit),
        }
    else:
        if args.policy_audit is None or args.evidence_audit is None:
            parser.error("provide both --policy-audit and --evidence-audit")
        policy_rows = _read_rows(args.policy_audit)
        evidence_rows = _read_rows(args.evidence_audit)
        frames = _joined_frames(policy_rows, evidence_rows)
        input_source = {
            "policy_audit": str(args.policy_audit),
            "evidence_audit": str(args.evidence_audit),
        }
    frames = override_policy_admission(
        frames,
        mode=args.admission_mode,
        warmup_frames=args.warmup_frames,
    )
    replay = replay_guarded_frames(
        frames,
        ema_alpha=args.ema_alpha,
        ema_threshold=args.ema_threshold,
        instantaneous_threshold=args.instantaneous_threshold,
        guard_enabled=not args.disable_guard,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "causal_guard_replay.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(replay[0].__dict__))
        writer.writeheader()
        writer.writerows([frame.__dict__ for frame in replay])

    radar_errors = [
        torch.linalg.vector_norm(
            torch.tensor([frame.radar_x_m, frame.radar_y_m])
            - torch.tensor([frame.target_x_m, frame.target_y_m])
        ).item()
        for frame in replay
    ]
    fused_errors = [
        torch.linalg.vector_norm(
            torch.tensor([frame.fused_x_m, frame.fused_y_m])
            - torch.tensor([frame.target_x_m, frame.target_y_m])
        ).item()
        for frame in replay
    ]
    admitted = [frame for frame in replay if frame.visual_accepted]
    epsilon = 1e-6
    improved = sum(
        fused_error + epsilon < radar_error
        for fused_error, radar_error in zip(fused_errors, radar_errors, strict=True)
    )
    harmed = sum(
        fused_error > radar_error + epsilon
        for fused_error, radar_error in zip(fused_errors, radar_errors, strict=True)
    )
    payload = {
        "configuration": {
            "ema_alpha": args.ema_alpha,
            "ema_threshold": args.ema_threshold,
            "instantaneous_threshold": args.instantaneous_threshold,
            "guard_enabled": not args.disable_guard,
            "admission_mode": args.admission_mode,
            "warmup_frames": args.warmup_frames,
            "input_source": input_source,
        },
        "frames": len(replay),
        "radar": _metrics(replay, fused=False),
        "guarded_fusion": _metrics(replay, fused=True),
        "visual_acceptance_rate": len(admitted) / len(replay),
        "identity_fallback_rate": 1.0 - len(admitted) / len(replay),
        "improved_rate": improved / len(replay),
        "harmed_rate": harmed / len(replay),
        "unchanged_rate": 1.0 - (improved + harmed) / len(replay),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
