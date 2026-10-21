#!/usr/bin/env python3
"""Materialize Train/Development-only InternVL metadata for RGB condition policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_specs(*, train_records: int, development_records: int) -> dict[str, tuple[str, str, int, bool]]:
    return {
        "train_meta_condition_policy_3expr.json": (
            "condition_policy_train",
            f"train_{train_records}_3expr_condition_policy_actualrgb.jsonl",
            train_records,
            True,
        ),
        "development_meta_condition_policy.json": (
            "condition_policy_development",
            f"development_{development_records}_condition_policy_actualrgb.jsonl",
            development_records,
            False,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    package = args.package_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    annotation_dir = package / "annotations"
    train_matches = list(annotation_dir.glob("train_*_3expr_condition_policy_actualrgb.jsonl"))
    development_matches = list(annotation_dir.glob("development_*_condition_policy_actualrgb.jsonl"))
    locked_matches = list(annotation_dir.glob("locked_test_*_READ_ONLY.jsonl"))
    if len(train_matches) != 1 or len(development_matches) != 1 or len(locked_matches) != 1:
        raise RuntimeError("condition-policy package must contain exactly Train, Development, and sealed Locked Test annotations")
    train_path, development_path = train_matches[0], development_matches[0]
    specs = build_specs(
        train_records=line_count(train_path),
        development_records=line_count(development_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, (name, annotation, length, is_train) in specs.items():
        payload = {
            name: {
                "root": str(data_root),
                "annotation": str(annotation_dir / annotation),
                "length": length,
                "data_augment": False,
                **({"max_dynamic_patch": 12, "repeat_time": 1} if is_train else {}),
            }
        }
        (output_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "status": "PASS",
        "package_root": str(package),
        "data_root": str(data_root),
        "train_annotation": train_path.name,
        "development_annotation": development_path.name,
        "locked_test_annotation": locked_matches[0].name,
        "locked_test_meta_generated": False,
        "locked_test_training_access": False,
        "generated": sorted(specs),
    }
    (output_dir / "condition_policy_meta_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



