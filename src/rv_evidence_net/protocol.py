"""Bind person-encoder fitting to the published sequence contract."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

TRAIN_SPLIT = "Train"
DEV_SPLIT = "Development Val"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_base_train_dev_manifest(manifest_path: Path, protocol_path: Path) -> dict[str, str]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if sha256(manifest_path) != protocol["frame_manifest_sha256"]:
        raise ValueError("frame manifest hash does not match the frozen protocol")
    with Path(manifest_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, str] = {}
    for row in rows:
        sequence, split = row["sequence"], row["split"]
        if split not in {TRAIN_SPLIT, DEV_SPLIT, "Locked Test"}:
            raise ValueError(f"unsupported split: {split}")
        if mapping.setdefault(sequence, split) != split:
            raise ValueError(f"sequence crosses splits: {sequence}")
    counts = Counter(mapping.values())
    for split, expected in protocol["splits"].items():
        if counts[split] != expected["sequences"]:
            raise ValueError(f"sequence count mismatch for {split}")
    # Inspecting split metadata is not permission to open test images or labels.
    return {sequence: split for sequence, split in mapping.items() if split in {TRAIN_SPLIT, DEV_SPLIT}}
