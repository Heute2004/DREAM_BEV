"""Export frozen numerical records without model inference or test-data access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True)
    used: list[str] = []
    locked_path = "evidence/causal_consistency_protection/final_tau05/locked_metrics_tau05.json"
    locked = read_json(locked_path)
    used.append(locked_path)
    rows = []
    for method, key in (("Expert radar", "radar"), ("Protected RVFusion", "guarded_fusion")):
        row = {"method": method, "frames": locked["frames"]}
        row.update({k: v for k, v in locked[key].items() if isinstance(v, (int, float))})
        rows.append(row)
    write_csv(output / "locked_summary.csv", rows)
    # Retain nested per-sequence records losslessly rather than guessing their schema.
    (output / "locked_per_sequence.json").write_text(
        json.dumps(locked["per_sequence"], indent=2) + "\n", encoding="utf-8"
    )
    cruw = []
    for expert in ("pbmhbr", "gg_srre"):
        source = f"evidence/public_sequence_transfer/cruw/protected/{expert}.json"
        record = read_json(source)
        used.append(source)
        for key in ("expert_radar", "rvfusion_unguarded", "rvfusion_guarded"):
            cruw.append({"expert": expert, "method": key, **record[key]})
    write_csv(output / "cruw_fusion.csv", cruw)
    sweep = "evidence/active_development_contract/stress_curves/visual_conflict_sweep.csv"
    shutil.copy2(ROOT / sweep, output / "visual_conflict_sweep.csv")
    used.append(sweep)
    # The published table is authoritative for inclusion; the broad source record
    # also contains models that are not part of the current paper.
    carrada = "evidence/public_sequence_transfer/carrada/current_paper_rows.json"
    record = read_json(carrada)
    write_csv(output / "carrada.csv", record["rows"])
    used.append(carrada)
    shutil.copytree(ROOT / "paper/tables", output / "paper_tables")
    used.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "paper/tables").glob("*.tex")))
    receipt = {
        "mode": "frozen-record export; no inference, model selection, or test-data reading",
        "paper_pdf_sha256": read_json("contracts/paper_binding.json")["main_pdf_sha256"],
        "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in used},
    }
    (output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output_dir)
    print(f"Exported {len(result['sources'])} source bindings to {args.output_dir}")
