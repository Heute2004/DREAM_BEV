from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_bearing_stress_runner_is_development_only() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_bearing_stress_evaluation_dev.sh"
    source = script.read_text(encoding="utf-8")
    assert "Locked Test is forbidden" in source
    assert 'reference-split "Development Val"' in source
    assert "--evaluation-only" in source
    assert "test ! -e" in source
    assert "CANDIDATE_CHECKPOINT" in source
    assert "bearing_stress_metadata.json" in source


def test_bearing_stress_runner_replays_the_deployed_candidate_contract() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_bearing_stress_evaluation_dev.sh"
    source = script.read_text(encoding="utf-8")
    assert 'scripts/train_fusion.py' in source
    assert '--phase candidate' in source
    assert '--acceptance-mode direct_utility' in source
    assert '--correction-source lattice' in source
    assert '--development-only' in source


def test_bearing_stress_runner_exports_import_path_before_building_cache() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_bearing_stress_evaluation_dev.sh"
    source = script.read_text(encoding="utf-8")
    export = 'export PYTHONPATH="$FUSION_CODE/src:$RADAR_CODE:$FUSION_CODE"'
    builder = '"$PYTHON" "$FUSION_CODE/scripts/build_bearing_stress_cache.py"'
    assert export in source
    assert source.index(export) < source.index(builder)


def test_bearing_stress_runner_keeps_candidate_arguments_in_one_command() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_bearing_stress_evaluation_dev.sh"
    source = script.read_text(encoding="utf-8")
    expected = (
        '  --device "$CUDA_DEVICE" \\\n'
        '  --phase candidate \\\n'
        '  --initial-checkpoint "$CANDIDATE" \\\n'
    )
    assert expected in source


def test_stress_builder_preserves_frozen_cache_manifest(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    source = tmp_path / "bearing_train_dev"
    output = tmp_path / "stress_cache"
    source.mkdir()
    frozen_manifest = {
        "cache_contract": "RVBearingEvidenceCacheV1",
        "manifest_sha256": "frozen-manifest-hash",
    }
    (source / "cache_manifest.json").write_text(
        json.dumps(frozen_manifest) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        source / "seq_022.npz",
        frame_ids=np.array([0], dtype=np.int64),
        theta=np.array([0.1], dtype=np.float32),
        sigma_theta=np.array([0.1], dtype=np.float32),
        valid=np.array([True]),
        reliability=np.array([1.0], dtype=np.float32),
        temporal_feature=np.ones((1, 2), dtype=np.float32),
    )
    environment = os.environ | {
        "PYTHONPATH": str(project_root / "src"),
    }
    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_bearing_stress_cache.py"),
            "--source-cache-root",
            str(source),
            "--output-cache-root",
            str(output),
            "--mode",
            "bearing_bias",
            "--bias-deg",
            "5",
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert (output / "cache_manifest.json").read_bytes() == (
        source / "cache_manifest.json"
    ).read_bytes()
    metadata = json.loads(
        (output / "bearing_stress_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["mode"] == "bearing_bias"
    assert metadata["source_cache_manifest_sha256"]
