from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.data import RadarWindowDataset
from radar_bev.evaluation import evaluate_single_person_predictions
from radar_bev.models.backbones import DREAMBEV, extract_dream_bev_state
from radar_bev.route_closure import ROUTES, assert_development_only_paths
from radar_bev.route_transitions import first_stable_route_frame
from rvfusion.adapters import FrozenRadarAdapter
from rvfusion.metric_contract import frozen_radar_confidence, resolved_radar_position
from rvfusion.radar_route_experts import RadarRouteExpertPaths, build_frozen_route_experts
from rvfusion.routed_radar_evidence import RoutedRadarEvidenceAdapter


ARMS = {
    "learned_delayed": "learned_delayed_route",
    "oracle_delayed": "oracle_delayed_route",
    "oracle_instant": "oracle_instant_route",
    "best_fixed": "best_fixed_route",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen radar experts on no-reset route transitions"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-open-indoor-los", type=Path, required=True)
    parser.add_argument("--expert-indoor-clutter", type=Path, required=True)
    parser.add_argument("--expert-open-outdoor-weak-return", type=Path, required=True)
    parser.add_argument("--expert-outdoor-dry-complex", type=Path, required=True)
    parser.add_argument("--expert-outdoor-wet-return", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-frame", type=int, default=150)
    parser.add_argument("--availability-delay-frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_schedule(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    schedule = {(row["sequence"], int(row["frame_id"])): row for row in rows}
    if len(schedule) != len(rows):
        raise ValueError("transition schedule contains duplicate frames")
    return schedule


def _load_adapter(args: argparse.Namespace, device: torch.device) -> RoutedRadarEvidenceAdapter:
    payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    backbone = DREAMBEV()
    backbone.load_state_dict(extract_dream_bev_state(payload), strict=True)
    base = FrozenRadarAdapter(backbone).to(device)
    paths = RadarRouteExpertPaths(
        open_indoor_los=args.expert_open_indoor_los,
        indoor_clutter=args.expert_indoor_clutter,
        open_outdoor_weak_return=args.expert_open_outdoor_weak_return,
        outdoor_dry_complex=args.expert_outdoor_dry_complex,
        outdoor_wet_return=args.expert_outdoor_wet_return,
    )
    experts = build_frozen_route_experts(
        base_checkpoint=args.base_checkpoint,
        paths=paths,
        device=device,
        gg_recovery_scale=1.15,
    )
    return RoutedRadarEvidenceAdapter(base, experts).to(device)


def _metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {"frames": 0.0}
    scores = torch.tensor([float(row["score"]) for row in rows], dtype=torch.float64)
    prediction = torch.tensor(
        [[float(row["prediction_x_m"]), float(row["prediction_y_m"])] for row in rows],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[float(row["target_x_m"]), float(row["target_y_m"])] for row in rows],
        dtype=torch.float64,
    )
    return {
        "frames": float(len(rows)),
        **evaluate_single_person_predictions(
            scores,
            prediction,
            target,
            distance_thresholds=(0.25, 0.30, 0.50),
        ),
    }


def _window_name(frame_id: int, boundary: int, delay: int) -> str:
    if frame_id < boundary:
        return "pre_transition"
    if frame_id < boundary + delay:
        return "transition_pending"
    return "post_transition_stable"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    assert_development_only_paths(args.manifest, args.schedule, args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    schedule = _load_schedule(args.schedule)
    dataset = RadarWindowDataset(
        data_root=args.data_root,
        manifest_path=args.manifest,
        split="Development Val",
        clip_length=16,
        eligibility="later",
        cache_primary_in_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device(args.device)
    adapter = _load_adapter(args, device)
    output_by_arm: dict[str, list[dict[str, object]]] = {}
    for arm, route_field in ARMS.items():
        adapter.reset_causal_state()
        rows: list[dict[str, object]] = []
        with torch.no_grad():
            for batch in loader:
                radar = batch["radar"].to(device, non_blocking=True)
                sequences = tuple(str(value) for value in batch["sequence"])
                frame_ids = tuple(int(value) for value in batch["frame_id"].tolist())
                routes = tuple(schedule[(sequence, frame_id)][route_field] for sequence, frame_id in zip(sequences, frame_ids))
                evidence = adapter(
                    radar,
                    routes,
                    sequences=sequences,
                    frame_ids=frame_ids,
                )
                prediction = resolved_radar_position(evidence).cpu()
                score = frozen_radar_confidence(evidence.logits).cpu()
                target = batch["target_xy_m"].cpu()
                for index, (sequence, frame_id, route) in enumerate(
                    zip(sequences, frame_ids, routes)
                ):
                    meta = schedule[(sequence, frame_id)]
                    rows.append(
                        {
                            "sequence": sequence,
                            "frame_id": frame_id,
                            "window": _window_name(
                                frame_id,
                                args.boundary_frame,
                                args.availability_delay_frames,
                            ),
                            "actual_route": meta["actual_route"],
                            "selected_route": route,
                            "route_correct": route == meta["actual_route"],
                            "source_sequence": meta["source_sequence"],
                            "source_frame_id": int(meta["source_frame_id"]),
                            "prediction_x_m": float(prediction[index, 0]),
                            "prediction_y_m": float(prediction[index, 1]),
                            "target_x_m": float(target[index, 0]),
                            "target_y_m": float(target[index, 1]),
                            "score": float(score[index]),
                        }
                    )
        output_by_arm[arm] = rows
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, object] = {}
    for arm, rows in output_by_arm.items():
        _write_rows(args.output_dir / f"{arm}_predictions.csv", rows)
        grouped = {
            window: [row for row in rows if row["window"] == window]
            for window in (
                "pre_transition",
                "transition_pending",
                "post_transition_stable",
            )
        }
        by_episode: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_episode[str(row["sequence"])].append(row)
        switch_latencies: list[int] = []
        switch_failures = 0
        for sequence in sorted(by_episode):
            all_schedule_routes = {
                frame_id: schedule[(sequence, frame_id)][ARMS[arm]]
                for frame_id in range(300)
            }
            target_route = schedule[(sequence, args.boundary_frame)]["actual_route"]
            first = first_stable_route_frame(
                all_schedule_routes,
                target_route=target_route,
                boundary_frame=args.boundary_frame,
            )
            if first is None:
                switch_failures += 1
            else:
                switch_latencies.append(first - args.boundary_frame)
        summaries[arm] = {
            "overall": _metrics(rows),
            "windows": {window: _metrics(members) for window, members in grouped.items()},
            "route_accuracy": fmean(bool(row["route_correct"]) for row in rows),
            "switch_success_episodes": len(by_episode) - switch_failures,
            "switch_failure_episodes": switch_failures,
            "switch_latency_frames_mean": (
                fmean(switch_latencies) if switch_latencies else None
            ),
            "switch_latency_frames_all": switch_latencies,
        }
    report = {
        "protocol": "IF-only Development no-reset route-transition evaluation",
        "locked_test_used": False,
        "episodes": len({key[0] for key in schedule}),
        "scored_frames": len(next(iter(output_by_arm.values()))),
        "radar_clip_length": 16,
        "boundary_frame": args.boundary_frame,
        "availability_delay_frames": args.availability_delay_frames,
        "causal_state_reset_at_boundary": False,
        "future_policy_access": False,
        "arms": summaries,
        "source_sha256": {
            "manifest": _sha256(args.manifest),
            "schedule": _sha256(args.schedule),
            "base_checkpoint": _sha256(args.base_checkpoint),
            "expert_open_indoor_los": _sha256(args.expert_open_indoor_los),
            "expert_indoor_clutter": _sha256(args.expert_indoor_clutter),
            "expert_open_outdoor_weak_return": _sha256(
                args.expert_open_outdoor_weak_return
            ),
            "expert_outdoor_dry_complex": _sha256(args.expert_outdoor_dry_complex),
            "expert_outdoor_wet_return": _sha256(args.expert_outdoor_wet_return),
        },
    }
    (args.output_dir / "route_transition_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
