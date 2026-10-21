from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (PROJECT_ROOT / "contracts" / "paper_result_contract.json").read_text(
        encoding="utf-8"
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_available_paper_result_sources_match_frozen_hashes() -> None:
    for source in CONTRACT["sources"]:
        path = PROJECT_ROOT / source["path"]
        if source.get("distribution") == "on_request" and not path.is_file():
            assert source["path"] == "contracts/frame_manifest.csv"
            assert source["sha256"] == "2c2d65f970cfa3b5b424588b4c9ef6464a865f201ad446c2f977ec7ecf34d152"
            continue
        assert path.is_file(), source["path"]
        assert _sha256(path) == source["sha256"], source["path"]


def test_locked_headline_matches_machine_readable_evidence() -> None:
    metrics = json.loads(
        (
            PROJECT_ROOT
            / "evidence"
            / "causal_consistency_protection"
            / "final_tau05"
            / "locked_metrics_tau05.json"
        ).read_text(encoding="utf-8")
    )
    bootstrap = json.loads(
        (
            PROJECT_ROOT
            / "evidence"
            / "causal_consistency_protection"
            / "final_tau05"
            / "locked_sequence_bootstrap_10000.json"
        ).read_text(encoding="utf-8")
    )
    expected = CONTRACT["locked_result"]
    assert metrics["frames"] == CONTRACT["sequence_contract"]["locked_test_frames"]
    assert metrics["radar"]["ap_0p3m"] == pytest.approx(
        expected["expert_radar_ap_0p30"]
    )
    assert metrics["guarded_fusion"]["ap_0p3m"] == pytest.approx(
        expected["protected_fusion_ap_0p30"]
    )
    assert metrics["radar"]["p90_location_error_m"] == pytest.approx(
        expected["expert_radar_p90_m"]
    )
    assert metrics["guarded_fusion"]["p90_location_error_m"] == pytest.approx(
        expected["protected_fusion_p90_m"]
    )
    assert bootstrap["point_estimate"]["ap_0p3_gain_pp"] == pytest.approx(
        expected["gain_percentage_points"]
    )
    assert bootstrap["confidence_interval_95"]["ap_0p3_gain_pp"] == pytest.approx(
        expected["bootstrap_gain_ci95_percentage_points"]
    )


def test_public_text_uses_forward_coordinate_contract() -> None:
    public_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "REPRODUCIBILITY.md",
        PROJECT_ROOT / "docs" / "PROJECT_LAYOUT.md",
        PROJECT_ROOT / "docs" / "EVIDENCE_INDEX.md",
        PROJECT_ROOT / "scripts" / "build_reproducibility_package.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in public_files)
    assert "range-preserving" not in combined.lower()
    assert "forward-coordinate-preserving" in combined.lower()
