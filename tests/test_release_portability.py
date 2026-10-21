from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from radar_bev.models import build_model
from rv_evidence_net.protocol import validate_base_train_dev_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_person_encoder_uses_only_train_and_development():
    if not (ROOT / "contracts/frame_manifest.csv").is_file():
        pytest.skip("private frame-level annotation manifest is available on request")
    mapping = validate_base_train_dev_manifest(ROOT / "contracts/frame_manifest.csv", ROOT / "contracts/protocol.json")
    assert Counter(mapping.values()) == {"Train": 87, "Development Val": 34}
    protocol = json.loads((ROOT / "contracts/protocol.json").read_text(encoding="utf-8"))
    assert not set(mapping).intersection(protocol["splits"]["Locked Test"]["members"])


def test_person_encoder_rejects_modified_manifest(tmp_path):
    if not (ROOT / "contracts/frame_manifest.csv").is_file():
        pytest.skip("private frame-level annotation manifest is available on request")
    changed = tmp_path / "frames.csv"
    changed.write_bytes((ROOT / "contracts/frame_manifest.csv").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        validate_base_train_dev_manifest(changed, ROOT / "contracts/protocol.json")


@pytest.mark.parametrize("name,count", [("erodnet_core", 1311137), ("trodnet_core", 1262817)])
def test_paper_core_adapters_keep_parameter_count(name, count):
    model = build_model(name)
    assert sum(parameter.numel() for parameter in model.parameters()) == count


def test_full_sweep_has_expected_source_hash():
    path = ROOT / "evidence/active_development_contract/stress_curves/visual_conflict_sweep.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "7fc743b0016c3931e82efa008be205ecbdfff2b3e6b3a4e0312d02117782ba96"


def test_public_release_excludes_private_frame_artifacts():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    private = json.loads((ROOT / "contracts/on_request_assets.json").read_text(encoding="utf-8"))
    assert len(private["files"]) == 15
    assert all(item["path"] not in manifest["files"] for item in private["files"])
