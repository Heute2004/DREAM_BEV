from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from radar_bev.protocol import (
    DEV_SPLIT,
    TEST_SPLIT,
    TRAIN_SPLIT,
    validate_training_manifest,
    validate_training_routing_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = PROJECT_ROOT / "contracts"


def test_canonical_split_contract_is_sequence_isolated_and_hashed() -> None:
    protocol = json.loads((CONTRACT_ROOT / "protocol.json").read_text(encoding="utf-8"))
    manifest = CONTRACT_ROOT / "frame_manifest.csv"
    if not manifest.is_file():
        pytest.skip("private frame-level annotation manifest is available on request")
    sequence_split = CONTRACT_ROOT / "sequence_split.csv"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    assert digest == protocol["frame_manifest_sha256"]
    assert validate_training_manifest(manifest) == {TRAIN_SPLIT: 87, DEV_SPLIT: 34}

    with sequence_split.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    observed = {}
    for row in rows:
        observed.setdefault(row["split"], 0)
        observed[row["split"]] += 1
    assert observed == {TRAIN_SPLIT: 87, DEV_SPLIT: 34, TEST_SPLIT: 8}
    assert {row["sequence"] for row in rows if row["split"] == TEST_SPLIT} == {
        "seq_032",
        "seq_041",
        "seq_052",
        "seq_069",
        "seq_102",
        "seq_118",
        "seq_132",
        "seq_153",
    }


def test_canonical_route_manifest_matches_the_sequence_contract() -> None:
    routing = CONTRACT_ROOT / "routing_manifest.jsonl"
    counts = validate_training_routing_manifest(routing)

    assert counts == {TRAIN_SPLIT: 87, DEV_SPLIT: 34}


def test_fusion_defaults_are_derived_from_the_policy_contract() -> None:
    schedule = json.loads((CONTRACT_ROOT / "policy_schedule.json").read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location(
        "train_fusion_contract_test",
        PROJECT_ROOT / "scripts" / "train_fusion.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.POLICY_AVAILABILITY_DELAY_FRAMES == schedule["policy_availability_delay_frames"]
    assert module.POLICY_HOLD_FRAMES == schedule["policy_hold_frames"]
    assert module.POLICY_UPDATE_HZ == (
        schedule["radar_frame_rate_hz"] / schedule["policy_capture_interval_frames"]
    )
