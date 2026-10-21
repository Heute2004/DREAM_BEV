from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_visual_conflict_sweep_dev.sh"
SUMMARIZER = PROJECT_ROOT / "scripts" / "summarize_visual_conflict_sweep.py"


def _load_summarizer():
    spec = importlib.util.spec_from_file_location("visual_conflict_summary", SUMMARIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(*, source: str, ap30: float, acceptance: float, harmed: float) -> dict:
    metric_block = {
        "ap_0p25m": ap30 - 0.1,
        "ap_0p3m": ap30,
        "ap_0p5m": ap30 + 0.1,
        "p90_location_error_m": 0.42,
    }
    return {
        "configuration": {"input_source": source},
        "frames": 9969,
        "radar": {**metric_block, "ap_0p3m": 0.7617},
        "guarded_fusion": metric_block,
        "visual_acceptance_rate": acceptance,
        "identity_fallback_rate": 1.0 - acceptance,
        "improved_rate": 0.2,
        "harmed_rate": harmed,
        "unchanged_rate": 0.8 - harmed,
    }


def _write_pair(root: Path, mode: str, level: str) -> None:
    for method, ap30, admission, harmed in (
        ("guard_off", 0.73, 0.65, 0.31),
        ("protected", 0.77, 0.21, 0.09),
    ):
        target = root / "replay" / mode / level / method
        target.mkdir(parents=True)
        payload = _metrics(
            source=f"development/{mode}/{level}",
            ap30=ap30,
            acceptance=admission,
            harmed=harmed,
        )
        (target / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_runner_declares_exact_development_severity_grid() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'BIAS_LEVELS=(1 2 3 4 5 6)' in source
    assert 'LAG_LEVELS=(2 4 8)' in source
    assert 'DROPOUT_LEVELS=(0.25 0.50 0.75)' in source
    assert 'reference-split "Development Val"' in (
        PROJECT_ROOT / "scripts" / "run_bearing_stress_evaluation_dev.sh"
    ).read_text(encoding="utf-8")
    assert "Locked Test is forbidden" in source
    assert 'EMA_ALPHA="0.10"' in source
    assert 'EMA_THRESHOLD="0.50"' in source
    assert "replay_causal_cross_modal_guard.py" in source
    assert "CLEAN_GUARD_OFF_METRICS" in source
    assert "CLEAN_GUARDED_METRICS" in source


def test_summary_uses_one_canonical_clean_point_for_all_axes(tmp_path: Path) -> None:
    module = _load_summarizer()
    run_root = tmp_path / "run"
    for level in ("1", "2", "3", "4", "5", "6"):
        _write_pair(run_root, "bearing_bias", level)
    for level in ("2", "4", "8"):
        _write_pair(run_root, "temporal_lag", level)
    for level in ("25", "50", "75"):
        _write_pair(run_root, "visual_dropout", level)

    clean_off = tmp_path / "clean_guard_off.json"
    clean_protected = tmp_path / "clean_guarded.json"
    clean_off.write_text(
        json.dumps(_metrics(source="development/clean", ap30=0.7918, acceptance=0.65, harmed=0.27)),
        encoding="utf-8",
    )
    clean_protected.write_text(
        json.dumps(_metrics(source="development/clean", ap30=0.7742, acceptance=0.35, harmed=0.13)),
        encoding="utf-8",
    )

    rows = module.build_summary(
        run_root=run_root,
        clean_guard_off=clean_off,
        clean_protected=clean_protected,
        expected_frames=9969,
    )
    assert len(rows) == 30
    zero_rows = [row for row in rows if row["severity"] == 0.0]
    assert len(zero_rows) == 6
    assert {row["perturbation"] for row in zero_rows} == {
        "bearing_bias",
        "temporal_lag",
        "visual_dropout",
    }
    assert {row["ap_0p30m"] for row in zero_rows if row["method"] == "protected"} == {0.7742}
    assert {row["radar_ap_0p30m"] for row in rows} == {0.7617}


def test_summary_rejects_locked_test_provenance(tmp_path: Path) -> None:
    module = _load_summarizer()
    metrics = _metrics(
        source="/runs/locked_test/forbidden.csv",
        ap30=0.7,
        acceptance=0.2,
        harmed=0.1,
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="locked test"):
        module.load_metrics(path, expected_frames=9969)
