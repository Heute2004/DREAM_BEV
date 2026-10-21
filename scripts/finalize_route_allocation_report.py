from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.route_closure import (
    assert_development_only_paths,
    evaluate_composed_rows,
    route_macro_ap30,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add the exact DREAM-BEV identity arm to route-allocation evidence"
    )
    parser.add_argument("--route-report", type=Path, required=True)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--identity-metadata", type=Path, required=True)
    parser.add_argument("--sequence-ledger", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _identity_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    output: list[dict[str, object]] = []
    for row in raw:
        output.append(
            {
                "sequence": row["sequence"],
                "frame_id": int(row["frame_id"]),
                "target_x_m": float(row["target_x_m"]),
                "target_y_m": float(row["target_y_m"]),
                "fused_x_m": float(row["radar_x_m"]),
                "fused_y_m": float(row["radar_y_m"]),
                "score": float(row["radar_score"]),
                "selected_route": "identity",
            }
        )
    if not output:
        raise ValueError("identity audit is empty")
    return output


def main() -> None:
    args = parse_args()
    assert_development_only_paths(
        args.route_report,
        args.identity_audit,
        args.identity_metadata,
        args.sequence_ledger,
        args.output,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    report = json.loads(args.route_report.read_text(encoding="utf-8"))
    metadata = json.loads(args.identity_metadata.read_text(encoding="utf-8"))
    if metadata.get("manifest_sha256") != args.expected_manifest_sha256:
        raise ValueError("identity arm does not use the canonical Development manifest")
    ledger = json.loads(args.sequence_ledger.read_text(encoding="utf-8"))
    rows = _identity_rows(args.identity_audit)
    identity = {
        **evaluate_composed_rows(rows),
        "route_macro_ap_0p3m": route_macro_ap30(rows, ledger),
        "frames": len(rows),
        "sequences": len({str(row["sequence"]) for row in rows}),
        "contract": "raw frozen DREAM-BEV; no expert and no visual fusion",
    }
    report["methods"] = {"identity": identity, **report["methods"]}
    methods = report["methods"]
    learned = float(methods["learned_route"]["ap_0p3m"])
    best_fixed_route = str(methods["best_fixed_route"])
    report["headline_deltas_pp"] = {
        "learned_minus_identity": 100.0 * (learned - float(identity["ap_0p3m"])),
        "learned_minus_best_fixed": 100.0
        * (learned - float(methods["fixed_routes"][best_fixed_route]["ap_0p3m"])),
        "learned_minus_static_uniform": 100.0
        * (learned - float(methods["static_uniform_mixture"]["ap_0p3m"])),
        "learned_minus_random_mean": 100.0
        * (learned - float(methods["seeded_random_route"]["ap_0p3m_mean"])),
        "oracle_minus_learned": 100.0
        * (float(methods["ledger_oracle_route"]["ap_0p3m"]) - learned),
    }
    report["source_sha256"].update(
        {
            "identity_audit": hashlib.sha256(args.identity_audit.read_bytes()).hexdigest(),
            "identity_metadata": hashlib.sha256(
                args.identity_metadata.read_bytes()
            ).hexdigest(),
        }
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
