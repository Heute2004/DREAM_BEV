from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rvfusion.route_counterfactual import read_policy_trace
from rvfusion.route_matrix import (
    evaluate_audit_by_ledger_route,
    evaluate_audit_by_reference_route,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a route-only RVFusion replay by original policy route."
    )
    audit = parser.add_mutually_exclusive_group(required=True)
    audit.add_argument("--audit-csv", type=Path)
    audit.add_argument("--audit-json", type=Path)
    parser.add_argument("--reference-trace", required=True, type=Path)
    parser.add_argument(
        "--sequence-route-map",
        type=Path,
        help="Frozen sequence-to-route ledger JSON; active frames are grouped by it.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _read_audit(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.audit_csv is not None:
        with args.audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        if rows and {"fused_x_m", "visual_accepted", "score"}.issubset(rows[0]):
            # Guard replays already contain the deployed output. Normalize
            # their field names to the route-matrix evaluator's audit schema.
            return [
                {
                    **row,
                    "radar_score": row["score"],
                    "candidate_x_m": row["fused_x_m"],
                    "candidate_y_m": row["fused_y_m"],
                    "accepted": row["visual_accepted"],
                }
                for row in rows
            ]
        return rows
    loaded = json.loads(args.audit_json.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not all(isinstance(row, dict) for row in loaded):
        raise ValueError("audit JSON must contain a list of audit records")
    return loaded


def main() -> None:
    args = parse_args()
    audit_rows = _read_audit(args)
    reference_trace = read_policy_trace(args.reference_trace)
    if args.sequence_route_map is None:
        metrics = evaluate_audit_by_reference_route(audit_rows, reference_trace)
    else:
        ledger = json.loads(args.sequence_route_map.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict) or not all(
            isinstance(sequence, str) and isinstance(route, str)
            for sequence, route in ledger.items()
        ):
            raise ValueError("sequence route map must be a JSON object of strings")
        metrics = evaluate_audit_by_ledger_route(audit_rows, reference_trace, ledger)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
