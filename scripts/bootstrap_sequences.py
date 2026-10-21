from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from radar_bev.evaluation import evaluate_single_person_predictions


FLOAT_FIELDS = (
    "target_x_m",
    "target_y_m",
    "radar_x_m",
    "radar_y_m",
    "candidate_x_m",
    "candidate_y_m",
    "radar_score",
    "candidate_score",
)


def _evaluate(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    target = torch.tensor(
        [[float(row["target_x_m"]), float(row["target_y_m"])] for row in rows],
        dtype=torch.float64,
    )
    radar = torch.tensor(
        [[float(row["radar_x_m"]), float(row["radar_y_m"])] for row in rows],
        dtype=torch.float64,
    )
    fused = torch.tensor(
        [
            [
                (
                    float(row["candidate_x_m"])
                    if bool(row["accepted"])
                    else float(row["radar_x_m"])
                ),
                float(row["radar_y_m"]),
            ]
            for row in rows
        ],
        dtype=torch.float64,
    )
    radar_score = torch.tensor(
        [float(row["radar_score"]) for row in rows], dtype=torch.float64
    )
    fused_score = torch.tensor(
        [float(row["candidate_score"]) for row in rows], dtype=torch.float64
    )
    thresholds = (0.25, 0.30, 0.50)
    radar_metrics = evaluate_single_person_predictions(
        radar_score, radar, target, distance_thresholds=thresholds
    )
    fused_metrics = evaluate_single_person_predictions(
        fused_score, fused, target, distance_thresholds=thresholds
    )
    return {
        "radar_ap_0p3m": radar_metrics["ap_0p3m"],
        "fused_ap_0p3m": fused_metrics["ap_0p3m"],
        "ap_0p3_gain_pp": 100.0
        * (fused_metrics["ap_0p3m"] - radar_metrics["ap_0p3m"]),
        "radar_p90_m": radar_metrics["p90_location_error_m"],
        "fused_p90_m": fused_metrics["p90_location_error_m"],
        "p90_reduction_m": radar_metrics["p90_location_error_m"]
        - fused_metrics["p90_location_error_m"],
        "radar_mean_error_m": radar_metrics["mean_location_error_m"],
        "fused_mean_error_m": fused_metrics["mean_location_error_m"],
        "mean_error_reduction_m": radar_metrics["mean_location_error_m"]
        - fused_metrics["mean_location_error_m"],
    }


def bootstrap_sequence_metrics(
    rows: Sequence[Mapping[str, object]], *, repetitions: int = 10_000, seed: int = 3407
) -> dict[str, object]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    sequences = sorted(grouped)
    if not sequences:
        raise ValueError("at least one complete sequence is required")

    point = _evaluate(rows)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(repetitions):
        sampled_sequences = rng.choice(sequences, size=len(sequences), replace=True)
        sampled_rows: list[Mapping[str, object]] = []
        for sequence in sampled_sequences:
            sampled_rows.extend(grouped[str(sequence)])
        metrics = _evaluate(sampled_rows)
        for key, value in metrics.items():
            draws[key].append(value)

    intervals = {
        key: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for key, values in draws.items()
    }
    return {
        "bootstrap_unit": "complete_sequence",
        "sequence_count": len(sequences),
        "sequences": sequences,
        "repetitions": repetitions,
        "seed": seed,
        "point_estimate": point,
        "confidence_interval_95": intervals,
    }


def read_audit_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("acceptance audit is empty")
    required = {"sequence", "accepted", *FLOAT_FIELDS}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"acceptance audit missing columns: {sorted(missing)}")
    parsed: list[dict[str, object]] = []
    for row in rows:
        parsed.append(
            {
                **row,
                **{field: float(row[field]) for field in FLOAT_FIELDS},
                "accepted": str(row["accepted"]).strip().lower()
                in {"1", "true", "yes"},
            }
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequence-cluster bootstrap for frozen RVFusion outputs."
    )
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = bootstrap_sequence_metrics(
        read_audit_rows(args.audit_csv),
        repetitions=args.repetitions,
        seed=args.seed,
    )
    result["source_audit_csv"] = str(args.audit_csv.resolve())
    result["source_audit_sha256"] = hashlib.sha256(
        args.audit_csv.read_bytes()
    ).hexdigest()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



