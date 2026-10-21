from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.evaluation import evaluate_single_person_predictions
from radar_bev.route_closure import ROUTES, assert_development_only_paths


METRICS = (
    "ap_0p25m",
    "ap_0p3m",
    "ap_0p5m",
    "mean_location_error_m",
    "median_location_error_m",
    "p90_location_error_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed matched universal-expert controls"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--routed-replay", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--sequence-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 2026, 2027])
    return parser.parse_args()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_rows(
    rows: list[dict[str, str]],
    *,
    prediction_fields: tuple[str, str],
    score_field: str,
) -> dict[str, float]:
    scores = torch.tensor([float(row[score_field]) for row in rows], dtype=torch.float64)
    prediction = torch.tensor(
        [[float(row[prediction_fields[0]]), float(row[prediction_fields[1]])] for row in rows],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[float(row["target_x_m"]), float(row["target_y_m"])] for row in rows],
        dtype=torch.float64,
    )
    return evaluate_single_person_predictions(
        scores, prediction, target, distance_thresholds=(0.25, 0.30, 0.50)
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty prediction CSV: {path}")
    return rows


def _route_macro(
    rows: list[dict[str, str]],
    *,
    prediction_fields: tuple[str, str],
    score_field: str,
    ledger: dict[str, str] | None = None,
) -> tuple[float, dict[str, dict[str, float]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        route = row.get("route", "") if ledger is None else ledger.get(row["sequence"], "")
        if route in ROUTES:
            grouped[route].append(row)
    if set(grouped) != set(ROUTES):
        raise ValueError(f"route macro is incomplete: {sorted(grouped)}")
    per_route = {
        route: _metric_rows(
            grouped[route],
            prediction_fields=prediction_fields,
            score_field=score_field,
        )
        for route in ROUTES
    }
    return fmean(per_route[route]["ap_0p3m"] for route in ROUTES), per_route


def _summarize_seed_metrics(seed_reports: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for metric in (*METRICS, "route_macro_ap_0p3m"):
        values = [float(report[metric]) for report in seed_reports]
        summary[metric] = {
            "mean": fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def main() -> None:
    args = parse_args()
    assert_development_only_paths(
        args.run_root,
        args.routed_replay,
        args.identity_audit,
        args.sequence_ledger,
        args.output_dir,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    ledger = json.loads(args.sequence_ledger.read_text(encoding="utf-8"))
    routed_rows = _read_csv(args.routed_replay)
    identity_rows = _read_csv(args.identity_audit)
    routed_metrics = _metric_rows(
        routed_rows,
        prediction_fields=("radar_x_m", "radar_y_m"),
        score_field="score",
    )
    routed_macro, routed_per_route = _route_macro(
        routed_rows,
        prediction_fields=("radar_x_m", "radar_y_m"),
        score_field="score",
        ledger=ledger,
    )
    identity_metrics = _metric_rows(
        identity_rows,
        prediction_fields=("radar_x_m", "radar_y_m"),
        score_field="radar_score",
    )
    identity_macro, identity_per_route = _route_macro(
        identity_rows,
        prediction_fields=("radar_x_m", "radar_y_m"),
        score_field="radar_score",
        ledger=ledger,
    )
    profiles: dict[str, object] = {}
    for profile in ("active", "bank"):
        seed_reports: list[dict[str, object]] = []
        for seed in args.seeds:
            directory = args.run_root / f"universal_{profile}_seed{seed}"
            prediction_path = directory / "best_development_predictions.csv"
            metrics_path = directory / "best_development_metrics.json"
            metadata_path = directory / "run_metadata.json"
            checkpoint_path = directory / "best_universal_expert.pt"
            for path in (prediction_path, metrics_path, metadata_path, checkpoint_path):
                if not path.is_file():
                    raise FileNotFoundError(f"universal control is incomplete: {path}")
            rows = _read_csv(prediction_path)
            keys = [(row["sequence"], int(row["frame_id"])) for row in rows]
            routed_keys = [(row["sequence"], int(row["frame_id"])) for row in routed_rows]
            if keys != routed_keys:
                raise ValueError(f"universal predictions are not aligned: {directory}")
            recomputed = _metric_rows(
                rows,
                prediction_fields=("prediction_x_m", "prediction_y_m"),
                score_field="score",
            )
            recorded = json.loads(metrics_path.read_text(encoding="utf-8"))
            for metric in METRICS:
                tolerance = 1e-9 if metric.startswith("ap_") else 1e-6
                if not math.isclose(
                    float(recomputed[metric]),
                    float(recorded[metric]),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                ):
                    raise ValueError(
                        f"saved prediction metric mismatch for {profile} seed={seed}: {metric}"
                    )
            route_macro, per_route = _route_macro(
                rows,
                prediction_fields=("prediction_x_m", "prediction_y_m"),
                score_field="score",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            seed_reports.append(
                {
                    "seed": seed,
                    **{metric: recomputed[metric] for metric in METRICS},
                    "route_macro_ap_0p3m": route_macro,
                    "per_route": per_route,
                    "trainable_parameters": int(metadata["trainable_parameters"]),
                    "checkpoint_sha256": _sha(checkpoint_path),
                    "prediction_sha256": _sha(prediction_path),
                }
            )
        canonical = next(report for report in seed_reports if report["seed"] == 3407)
        profiles[profile] = {
            "seeds": seed_reports,
            "aggregate": _summarize_seed_metrics(seed_reports),
            "canonical_seed": canonical,
            "canonical_delta_vs_routed_pool_pp": 100.0
            * (float(canonical["ap_0p3m"]) - routed_metrics["ap_0p3m"]),
            "canonical_route_macro_delta_vs_routed_pool_pp": 100.0
            * (float(canonical["route_macro_ap_0p3m"]) - routed_macro),
        }
    report = {
        "protocol": "IF-only Development matched universal-expert controls",
        "locked_test_used": False,
        "identity": {
            **identity_metrics,
            "route_macro_ap_0p3m": identity_macro,
            "per_route": identity_per_route,
        },
        "routed_heterogeneous_pool": {
            **routed_metrics,
            "route_macro_ap_0p3m": routed_macro,
            "per_route": routed_per_route,
            "active_parameters_mean": 72650,
            "bank_parameters": 363248,
        },
        "universal_controls": profiles,
        "source_sha256": {
            "routed_replay": _sha(args.routed_replay),
            "identity_audit": _sha(args.identity_audit),
            "sequence_ledger": _sha(args.sequence_ledger),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "universal_expert_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
