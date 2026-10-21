"""Sequence-isolated split validation used by every training entry point."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


TRAIN_SPLIT = "Train"
DEV_SPLIT = "Development Val"
TEST_SPLIT = "Locked Test"


def _require_rows(path: Path) -> Iterable[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    required = {"sequence", "frame_id", "split"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    return rows


def _sequence_splits(rows: Iterable[dict[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        sequence = str(row["sequence"]).strip()
        split = str(row["split"]).strip()
        if not sequence or not split:
            raise ValueError("manifest contains an empty sequence or split")
        previous = mapping.setdefault(sequence, split)
        if previous != split:
            raise ValueError(f"sequence crosses splits: {sequence}")
    return mapping


def validate_training_manifest(path: str | Path, scope: str = "base") -> dict[str, int]:
    """Validate the canonical manifest before a Train/Development-only command."""

    if scope != "base":
        raise ValueError("only the canonical base scope is supported")
    splits = _sequence_splits(_require_rows(Path(path)))
    illegal = {
        sequence: split
        for sequence, split in splits.items()
        if split not in {TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT}
    }
    if illegal:
        raise ValueError(f"manifest includes an unsupported split: {illegal}")
    counts = Counter(splits.values())
    if not counts[TRAIN_SPLIT] or not counts[DEV_SPLIT]:
        raise ValueError("training manifest must include Train and Development Val sequences")
    return {TRAIN_SPLIT: counts[TRAIN_SPLIT], DEV_SPLIT: counts[DEV_SPLIT]}


def validate_locked_test_manifest(path: str | Path, scope: str = "base") -> dict[str, int]:
    """Ensure an evaluation manifest contains only the locked test sequences."""

    if scope != "base":
        raise ValueError("only the canonical base scope is supported")
    splits = _sequence_splits(_require_rows(Path(path)))
    illegal = {sequence: split for sequence, split in splits.items() if split != TEST_SPLIT}
    if illegal:
        raise ValueError(f"locked evaluation manifest includes another split: {illegal}")
    return {TEST_SPLIT: len(splits)}


def validate_training_routing_manifest(path: str | Path, scope: str = "base") -> dict[str, int]:
    """Validate the canonical route manifest before a Train/Development command."""

    if scope != "base":
        raise ValueError("only the canonical base scope is supported")
    sequence_splits: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sequence = str(item.get("sequence_name", item.get("sequence", ""))).strip()
            split = str(item.get("split", "")).strip()
            if not sequence or not split:
                raise ValueError("routing manifest requires sequence_name and split")
            previous = sequence_splits.setdefault(sequence, split)
            if previous != split:
                raise ValueError(f"routing sequence crosses splits: {sequence}")
    if not sequence_splits:
        raise ValueError("routing manifest is empty")
    illegal = {
        sequence: split
        for sequence, split in sequence_splits.items()
        if split not in {TRAIN_SPLIT, DEV_SPLIT, TEST_SPLIT}
    }
    if illegal:
        raise ValueError(f"routing manifest includes an unsupported split: {illegal}")
    counts = Counter(sequence_splits.values())
    return {TRAIN_SPLIT: counts[TRAIN_SPLIT], DEV_SPLIT: counts[DEV_SPLIT]}



