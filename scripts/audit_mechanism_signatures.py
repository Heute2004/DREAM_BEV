from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.mechanism_signatures import (
    SIGNATURE_FIELDS,
    frame_mechanism_signature,
    route_signature_score,
    spearman_correlation,
    zscore_signature_rows,
)
from radar_bev.route_closure import ROUTES, assert_development_only_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Development-only radar mechanism signature audit"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forced-root", type=Path, required=True)
    parser.add_argument("--sequence-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    return parser.parse_args()


def _read_guard_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    output: list[dict[str, object]] = []
    for row in raw:
        output.append(
            {
                **row,
                "frame_id": int(row["frame_id"]),
                **{
                    field: float(row[field])
                    for field in (
                        "radar_x_m",
                        "radar_y_m",
                        "target_x_m",
                        "target_y_m",
                    )
                },
            }
        )
    if not output:
        raise ValueError(f"empty forced-route replay: {path}")
    return output


def _load_manifest_paths(
    manifest: Path, data_root: Path, required_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], tuple[Path, Path]]:
    paths: dict[tuple[str, int], tuple[Path, Path]] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() != "development val":
                continue
            key = (row["sequence"], int(row["frame_id"]))
            if key in required_keys:
                paths[key] = (data_root / row["ra_path"], data_root / row["ra_mti_path"])
    missing = required_keys.difference(paths)
    if missing:
        raise ValueError(f"manifest lacks {len(missing)} scored Development frames")
    return paths


def _euclidean_error(row: dict[str, object]) -> float:
    return math.hypot(
        float(row["radar_x_m"]) - float(row["target_x_m"]),
        float(row["radar_y_m"]) - float(row["target_y_m"]),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_sequences(frame_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in frame_rows:
        grouped[str(row["sequence"])].append(row)
    output: list[dict[str, object]] = []
    for sequence, rows in sorted(grouped.items()):
        output.append(
            {
                "sequence": sequence,
                "route": rows[0]["route"],
                "frames": len(rows),
                **{
                    field: fmean(float(row[field]) for row in rows)
                    for field in SIGNATURE_FIELDS
                },
                "mechanism_score": fmean(float(row["mechanism_score"]) for row in rows),
                "specialist_advantage_m": fmean(
                    float(row["specialist_advantage_m"]) for row in rows
                ),
                "specialist_advantage_vs_best_wrong_m": fmean(
                    float(row["specialist_advantage_vs_best_wrong_m"])
                    for row in rows
                ),
                "hit30_advantage": fmean(float(row["hit30_advantage"]) for row in rows),
            }
        )
    return output


def _macro_f1(target: list[str], prediction: list[str]) -> float:
    scores: list[float] = []
    for route in ROUTES:
        true_positive = sum(t == route and p == route for t, p in zip(target, prediction))
        false_positive = sum(t != route and p == route for t, p in zip(target, prediction))
        false_negative = sum(t == route and p != route for t, p in zip(target, prediction))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        scores.append(
            0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        )
    return fmean(scores)


def _leave_one_sequence_out_centroid(
    sequence_rows: list[dict[str, object]],
) -> dict[str, object]:
    matrix = np.asarray(
        [[float(row[field]) for field in SIGNATURE_FIELDS] for row in sequence_rows],
        dtype=np.float64,
    )
    labels = [str(row["route"]) for row in sequence_rows]
    predictions: list[str] = []
    for held_index in range(len(sequence_rows)):
        train_indices = [index for index in range(len(sequence_rows)) if index != held_index]
        train = matrix[train_indices]
        center = np.median(train, axis=0)
        scale = np.median(np.abs(train - center), axis=0) * 1.4826
        scale = np.where(scale > 1e-9, scale, np.std(train, axis=0))
        scale = np.where(scale > 1e-9, scale, 1.0)
        normalized_train = (train - center) / scale
        normalized_held = (matrix[held_index] - center) / scale
        centroids = {
            route: normalized_train[
                [labels[index] == route for index in train_indices]
            ].mean(axis=0)
            for route in ROUTES
            if any(labels[index] == route for index in train_indices)
        }
        predictions.append(
            min(
                centroids,
                key=lambda route: float(np.linalg.norm(normalized_held - centroids[route])),
            )
        )
    accuracy = fmean(target == predicted for target, predicted in zip(labels, predictions))
    return {
        "protocol": "leave_one_complete_sequence_out_nearest_centroid",
        "features": list(SIGNATURE_FIELDS),
        "accuracy": accuracy,
        "macro_f1": _macro_f1(labels, predictions),
        "predictions": [
            {
                "sequence": row["sequence"],
                "target_route": target,
                "predicted_route": predicted,
            }
            for row, target, predicted in zip(sequence_rows, labels, predictions)
        ],
    }


def _severity_summary(sequence_rows: list[dict[str, object]]) -> dict[str, object]:
    by_route: dict[str, list[dict[str, object]]] = {
        route: [row for row in sequence_rows if row["route"] == route] for route in ROUTES
    }
    route_reports: dict[str, object] = {}
    for route, rows in by_route.items():
        ordered = sorted(rows, key=lambda row: float(row["mechanism_score"]))
        groups = np.array_split(np.asarray(ordered, dtype=object), 3)
        route_reports[route] = {
            "sequences": len(rows),
            "spearman_score_vs_specialist_advantage": spearman_correlation(
                [float(row["mechanism_score"]) for row in rows],
                [float(row["specialist_advantage_m"]) for row in rows],
            ),
            "tertiles": [
                {
                    "level": level,
                    "sequences": len(group),
                    "mean_score": fmean(float(row["mechanism_score"]) for row in group),
                    "mean_specialist_advantage_m": fmean(
                        float(row["specialist_advantage_m"]) for row in group
                    ),
                    "mean_hit30_advantage": fmean(
                        float(row["hit30_advantage"]) for row in group
                    ),
                }
                for level, group in zip(("low", "middle", "high"), groups)
            ],
        }
    return {
        "overall_spearman_score_vs_specialist_advantage": spearman_correlation(
            [float(row["mechanism_score"]) for row in sequence_rows],
            [float(row["specialist_advantage_m"]) for row in sequence_rows],
        ),
        "routes": route_reports,
    }


def main() -> None:
    args = parse_args()
    assert_development_only_paths(
        args.manifest, args.forced_root, args.sequence_ledger, args.output_dir
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    ledger = json.loads(args.sequence_ledger.read_text(encoding="utf-8"))
    forced_rows = {
        route: _read_guard_rows(
            args.forced_root
            / "evaluations"
            / route
            / "guarded"
            / "causal_guard_replay.csv"
        )
        for route in ROUTES
    }
    keys = [(str(row["sequence"]), int(row["frame_id"])) for row in forced_rows[ROUTES[0]]]
    if any(
        [(str(row["sequence"]), int(row["frame_id"])) for row in forced_rows[route]] != keys
        for route in ROUTES[1:]
    ):
        raise ValueError("forced-route outputs are not frame aligned")
    path_map = _load_manifest_paths(args.manifest, args.data_root, set(keys))
    row_maps = {
        route: {
            (str(row["sequence"]), int(row["frame_id"])): row for row in rows
        }
        for route, rows in forced_rows.items()
    }
    raw_signatures: list[dict[str, object]] = []
    previous_peaks: dict[str, tuple[int, int]] = {}
    for sequence, frame_id in keys:
        route = ledger.get(sequence)
        if route not in ROUTES:
            continue
        ra_path, mti_path = path_map[(sequence, frame_id)]
        ra = np.load(ra_path, allow_pickle=False)
        mti = np.load(mti_path, allow_pickle=False)
        signature, peak = frame_mechanism_signature(
            ra, mti, previous_mti_peak=previous_peaks.get(sequence)
        )
        previous_peaks[sequence] = peak
        correct_error = _euclidean_error(row_maps[route][(sequence, frame_id)])
        wrong_errors = [
            _euclidean_error(row_maps[wrong_route][(sequence, frame_id)])
            for wrong_route in ROUTES
            if wrong_route != route
        ]
        raw_signatures.append(
            {
                "sequence": sequence,
                "frame_id": frame_id,
                "route": route,
                **signature,
                "correct_expert_error_m": correct_error,
                "mean_wrong_expert_error_m": fmean(wrong_errors),
                "best_wrong_expert_error_m": min(wrong_errors),
                "specialist_advantage_m": fmean(wrong_errors) - correct_error,
                "specialist_advantage_vs_best_wrong_m": min(wrong_errors) - correct_error,
                "hit30_advantage": float(correct_error <= 0.30)
                - fmean(float(error <= 0.30) for error in wrong_errors),
            }
        )
    standardized = zscore_signature_rows(raw_signatures)
    frame_rows: list[dict[str, object]] = []
    for row, zscores in zip(raw_signatures, standardized):
        route = str(row["route"])
        frame_rows.append(
            {
                **row,
                "mechanism_score": route_signature_score(route, zscores),
            }
        )
    sequence_rows = _aggregate_sequences(frame_rows)
    report = {
        "protocol": "IF-only Development radar-mechanism signature audit",
        "locked_test_used": False,
        "frames": len(frame_rows),
        "sequences": len(sequence_rows),
        "route_counts": {
            route: sum(row["route"] == route for row in sequence_rows) for route in ROUTES
        },
        "signature_route_separability": _leave_one_sequence_out_centroid(sequence_rows),
        "severity_specialist_relationship": _severity_summary(sequence_rows),
        "limitations": [
            "route signatures are descriptive frozen Development diagnostics",
            "small route strata make within-route correlations exploratory",
            "the audit does not claim discovery of exhaustive physical failure mechanisms",
        ],
        "source_sha256": {
            "manifest": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "sequence_ledger": hashlib.sha256(args.sequence_ledger.read_bytes()).hexdigest(),
            **{
                f"forced_{route}": hashlib.sha256(
                    (
                        args.forced_root
                        / "evaluations"
                        / route
                        / "guarded"
                        / "causal_guard_replay.csv"
                    ).read_bytes()
                ).hexdigest()
                for route in ROUTES
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output_dir / "frame_mechanism_signatures.csv", frame_rows)
    _write_csv(args.output_dir / "sequence_mechanism_signatures.csv", sequence_rows)
    (args.output_dir / "mechanism_signature_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
