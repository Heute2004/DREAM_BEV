#!/usr/bin/env python3
"""Audit CARRADA and freeze the official complete-sequence 18/6/6 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from carrada_dream_bev.data import CARRADABEVGeometry, _targets_from_frame, canonical_frame_files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_statistics(data_root: Path, annotations: dict[str, object]) -> list[dict[str, object]]:
    geometry = CARRADABEVGeometry()
    rows: list[dict[str, object]] = []
    for sequence_id in sorted(annotations):
        sequence_root = data_root / sequence_id
        if not sequence_root.is_dir():
            continue
        ra_directory = sequence_root / "range_angle_numpy"
        rgb_directory = sequence_root / "camera_images"
        ra_frames = set(canonical_frame_files(ra_directory, ".npy"))
        rgb_frames = set(canonical_frame_files(rgb_directory, ".jpg"))
        noncanonical_ra = [path.name for path in ra_directory.glob("*.npy") if len(path.stem) != 6 or not path.stem.isdigit()]
        noncanonical_rgb = [path.name for path in rgb_directory.glob("*.jpg") if len(path.stem) != 6 or not path.stem.isdigit()]
        annotation_frames = {int(value) for value in annotations[sequence_id]}
        synchronized = sorted(ra_frames & rgb_frames & annotation_frames)
        counts = {"empty": 0, "single": 0, "multi": 0, "distractor_frames": 0}
        for frame_id in synchronized:
            targets, distractors = _targets_from_frame(annotations[sequence_id][f"{frame_id:06d}"])
            valid = sum(geometry.contains(target.x_m, target.y_m) for target in targets)
            if valid == 0:
                counts["empty"] += 1
            elif valid == 1:
                counts["single"] += 1
            else:
                counts["multi"] += 1
            counts["distractor_frames"] += int(distractors > 0)
        rows.append(
            {
                "sequence": sequence_id,
                "year": sequence_id[:4],
                "radar_frames": len(ra_frames),
                "rgb_frames": len(rgb_frames),
                "annotation_frames": len(annotation_frames),
                "synchronized_frames": len(synchronized),
                "noncanonical_ra_files": len(noncanonical_ra),
                "noncanonical_rgb_files": len(noncanonical_rgb),
                **counts,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations_path = args.data_root / "annotations_frame_oriented.json"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    rows = sequence_statistics(args.data_root, annotations)
    sequence_reference_path = args.data_root / "data_seq_ref.json"
    sequence_reference = json.loads(sequence_reference_path.read_text(encoding="utf-8"))
    fit = sorted(sequence for sequence, item in sequence_reference.items() if item["split"].lower() == "train")
    development = sorted(sequence for sequence, item in sequence_reference.items() if item["split"].lower() == "validation")
    locked_test = sorted(sequence for sequence, item in sequence_reference.items() if item["split"].lower() == "test")
    payload = {
        "dataset": "CARRADA",
        "purpose": "official complete-sequence Train/Validation/Test external evaluation",
        "selection": "official CARRADA scene split from data_seq_ref.json",
        "fit_sequences": fit,
        "development_sequences": development,
        "locked_test_sequences": locked_test,
        "sequence_statistics": rows,
        "annotations_sha256": sha256(annotations_path),
        "data_seq_ref_sha256": sha256(sequence_reference_path),
        "sequence_overlap": sorted((set(fit) & set(development)) | (set(fit) & set(locked_test)) | (set(development) & set(locked_test))),
        "all_sequence_count": len(rows),
        "fit_sequence_count": len(fit),
        "development_sequence_count": len(development),
        "locked_test_sequence_count": len(locked_test),
        "noncanonical_files": {
            "ra": sum(int(row["noncanonical_ra_files"]) for row in rows),
            "rgb": sum(int(row["noncanonical_rgb_files"]) for row in rows),
        },
    }
    if payload["sequence_overlap"] or len(rows) != 30 or len(fit) != 18 or len(development) != 6 or len(locked_test) != 6:
        raise RuntimeError(f"invalid CARRADA protocol: {payload}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("all_sequence_count", "fit_sequence_count", "development_sequence_count", "locked_test_sequence_count", "development_sequences", "locked_test_sequences")}, indent=2))


if __name__ == "__main__":
    main()
