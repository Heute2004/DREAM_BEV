from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.route_closure import (
    ROUTES,
    assert_development_only_paths,
    compose_selected_route_rows,
    compose_oracle_rows,
    compose_random_rows,
    compose_static_mixture_rows,
    evaluate_composed_rows,
    paired_route_macro_bootstrap,
    route_macro_ap30,
    validate_route_row_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Development-only route-allocation counterfactual audit"
    )
    parser.add_argument("--forced-root", type=Path, required=True)
    parser.add_argument("--policy-trace", type=Path, required=True)
    parser.add_argument("--identity-csv", type=Path, required=True)
    parser.add_argument("--sequence-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    return parser.parse_args()


def _read_guard_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"empty guard replay: {path}")
    required = {
        "sequence",
        "frame_id",
        "fused_x_m",
        "fused_y_m",
        "target_x_m",
        "target_y_m",
        "score",
    }
    missing = required.difference(raw[0])
    if missing:
        raise ValueError(f"guard replay missing columns {sorted(missing)}: {path}")
    result: list[dict[str, object]] = []
    for row in raw:
        result.append(
            {
                **row,
                "frame_id": int(row["frame_id"]),
                **{
                    field: float(row[field])
                    for field in (
                        "fused_x_m",
                        "fused_y_m",
                        "target_x_m",
                        "target_y_m",
                        "score",
                    )
                },
            }
        )
    return result


def _read_identity_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError(f"empty identity audit: {path}")
    required = {
        "sequence",
        "frame_id",
        "radar_x_m",
        "radar_y_m",
        "target_x_m",
        "target_y_m",
        "radar_score",
    }
    missing = required.difference(raw[0])
    if missing:
        raise ValueError(f"identity audit missing columns {sorted(missing)}: {path}")
    return [
        {
            **row,
            "frame_id": int(row["frame_id"]),
            "fused_x_m": float(row["radar_x_m"]),
            "fused_y_m": float(row["radar_y_m"]),
            "target_x_m": float(row["target_x_m"]),
            "target_y_m": float(row["target_y_m"]),
            "score": float(row["radar_score"]),
        }
        for row in raw
    ]


def _read_policy_selections(path: Path) -> dict[tuple[str, int], str]:
    selections: dict[tuple[str, int], str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        sequence_match = re.search(r"(seq_\d+)", str(record.get("id", "")))
        if sequence_match is None:
            raise ValueError(f"cannot recover sequence from policy record: {record}")
        sequence = sequence_match.group(1)
        route = str(record["prediction"].get("expert_route", "identity"))
        start = int(record["policy_segment_start_frame"])
        end = int(record["policy_segment_end_frame"])
        for frame_id in range(start, end + 1):
            key = (sequence, frame_id)
            if key in selections and selections[key] != route:
                raise ValueError(f"overlapping policy segments disagree for {key}")
            selections[key] = route
    if not selections:
        raise ValueError(f"empty policy trace: {path}")
    return selections


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "sequence",
        "frame_id",
        "selected_route",
        "target_x_m",
        "target_y_m",
        "fused_x_m",
        "fused_y_m",
        "score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _summarize(
    rows: list[dict[str, object]], ledger: dict[str, str]
) -> dict[str, object]:
    return {
        **evaluate_composed_rows(rows),
        "route_macro_ap_0p3m": route_macro_ap30(rows, ledger),
        "frames": len(rows),
        "sequences": len({str(row["sequence"]) for row in rows}),
    }


def main() -> None:
    args = parse_args()
    assert_development_only_paths(
        args.forced_root,
        args.policy_trace,
        args.identity_csv,
        args.sequence_ledger,
        args.output_dir,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    if args.random_seeds < 1:
        raise ValueError("random-seeds must be positive")
    ledger = json.loads(args.sequence_ledger.read_text(encoding="utf-8"))
    route_rows = {
        route: _read_guard_rows(
            args.forced_root
            / "evaluations"
            / route
            / "guarded"
            / "causal_guard_replay.csv"
        )
        for route in ROUTES
    }
    validate_route_row_alignment(route_rows)
    identity = _read_identity_rows(args.identity_csv)
    selections = _read_policy_selections(args.policy_trace)
    route_keys = {
        (str(row["sequence"]), int(row["frame_id"]))
        for row in route_rows[ROUTES[0]]
    }
    missing_policy = sorted(route_keys.difference(selections))
    if missing_policy:
        raise ValueError(f"policy trace misses scored frames: {missing_policy[:5]}")
    learned = compose_selected_route_rows(
        route_rows,
        selected_routes=selections,
        identity_rows=identity,
    )
    ledger_matched = compose_oracle_rows(route_rows, ledger)
    static = compose_static_mixture_rows(route_rows)
    fixed = {route: _summarize(rows, ledger) for route, rows in route_rows.items()}
    best_fixed_route = max(fixed, key=lambda route: fixed[route]["ap_0p3m"])
    random_summaries: list[dict[str, object]] = []
    random_seed_rows: list[dict[str, object]] | None = None
    for seed in range(args.random_seeds):
        rows = compose_random_rows(route_rows, identity_rows=identity, seed=seed)
        if seed == 0:
            random_seed_rows = rows
        random_summaries.append({"seed": seed, **_summarize(rows, ledger)})
    random_ap = np.asarray(
        [float(item["ap_0p3m"]) for item in random_summaries], dtype=np.float64
    )
    methods = {
        "identity": _summarize(identity, ledger),
        "fixed_routes": fixed,
        "best_fixed_route": best_fixed_route,
        "static_uniform_mixture": _summarize(static, ledger),
        "seeded_random_route": {
            "seeds": args.random_seeds,
            "ap_0p3m_mean": float(random_ap.mean()),
            "ap_0p3m_std": float(random_ap.std(ddof=1)) if len(random_ap) > 1 else 0.0,
            "ap_0p3m_min": float(random_ap.min()),
            "ap_0p3m_max": float(random_ap.max()),
            "all_seed_metrics": random_summaries,
        },
        "learned_route": _summarize(learned, ledger),
        "ledger_matched_route": _summarize(ledger_matched, ledger),
    }
    eligible_selections = [
        (key, route) for key, route in selections.items() if key in route_keys and route in ROUTES
    ]
    agreement = sum(ledger.get(key[0]) == route for key, route in eligible_selections)
    methods["learned_route"]["ledger_route_agreement"] = {
        "correct_frames": agreement,
        "eligible_frames": len(eligible_selections),
        "accuracy": agreement / len(eligible_selections),
    }
    inference = {
        "learned_minus_best_fixed_route_macro": paired_route_macro_bootstrap(
            learned,
            route_rows[best_fixed_route],
            ledger,
            repetitions=args.bootstrap_repetitions,
            seed=3407,
        ),
        "learned_minus_static_uniform_route_macro": paired_route_macro_bootstrap(
            learned,
            static,
            ledger,
            repetitions=args.bootstrap_repetitions,
            seed=2026,
        ),
        "ledger_matched_minus_learned_route_macro": paired_route_macro_bootstrap(
            ledger_matched,
            learned,
            ledger,
            repetitions=args.bootstrap_repetitions,
            seed=2027,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "IF-only Development route allocation closure",
        "locked_test_used": False,
        "methods": methods,
        "paired_sequence_inference": inference,
        "source_sha256": {
            "policy_trace": hashlib.sha256(args.policy_trace.read_bytes()).hexdigest(),
            "identity_csv": hashlib.sha256(args.identity_csv.read_bytes()).hexdigest(),
            "sequence_ledger": hashlib.sha256(
                args.sequence_ledger.read_bytes()
            ).hexdigest(),
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
    (args.output_dir / "route_allocation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_rows(args.output_dir / "learned_route_composed.csv", learned)
    _write_rows(args.output_dir / "ledger_matched_composed.csv", ledger_matched)
    _write_rows(args.output_dir / "static_uniform_composed.csv", static)
    assert random_seed_rows is not None
    _write_rows(args.output_dir / "random_seed000_composed.csv", random_seed_rows)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
