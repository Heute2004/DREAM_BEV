from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(path: Path, *, data_root: Path) -> None:
    train_rgb = data_root / "seq_022" / "RGB" / "frame_0000.jpg"
    train_ra = data_root / "seq_022" / "Radar" / "RA" / "frame_0000.npy"
    train_mti = data_root / "seq_022" / "Radar" / "RA-MIT" / "frame_0000.npy"
    for item in (train_rgb, train_ra, train_mti):
        item.parent.mkdir(parents=True, exist_ok=True)
        item.touch()
    fields = [
        "sequence", "frame_id", "split", "sequence_excluded", "rgb_path",
        "ra_path", "ra_mti_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "sequence": "seq_022", "frame_id": "0", "split": "Train",
            "sequence_excluded": "False", "rgb_path": "seq_022/RGB/frame_0000.jpg",
            "ra_path": "seq_022/Radar/RA/frame_0000.npy",
            "ra_mti_path": "seq_022/Radar/RA-MIT/frame_0000.npy",
        })
        writer.writerow({
            "sequence": "seq_032", "frame_id": "0", "split": "Locked Test",
            "sequence_excluded": "False", "rgb_path": "seq_032/RGB/frame_0000.jpg",
            "ra_path": "seq_032/Radar/RA/frame_0000.npy",
            "ra_mti_path": "seq_032/Radar/RA-MIT/frame_0000.npy",
        })


def test_visual_cache_selection_excludes_locked_test(tmp_path: Path) -> None:
    module = _load_script("cache_visual_evidence.py")
    manifest = tmp_path / "manifest.csv"
    data_root = tmp_path / "data"
    _write_manifest(manifest, data_root=data_root)

    selected = module.select_visual_sequences(
        manifest,
        data_root=data_root,
        allowed_splits={"train", "development val"},
    )

    assert tuple(selected) == ("seq_022",)


def test_cache_split_normalizers_accept_shell_safe_development_val_alias() -> None:
    visual = _load_script("cache_visual_evidence.py")
    health = _load_script("cache_radar_health.py")
    bearing = _load_script("export_bearing_cache.py")

    expected = {"train", "development val"}
    values = ["Train", "Development_Val"]

    assert visual.normalize_split_names(values) == expected
    assert health.normalize_split_names(values) == expected
    assert bearing.normalize_split_names(values) == expected


def test_radar_health_selection_excludes_locked_test(tmp_path: Path) -> None:
    module = _load_script("cache_radar_health.py")
    manifest = tmp_path / "manifest.csv"
    data_root = tmp_path / "data"
    _write_manifest(manifest, data_root=data_root)

    selected = module.load_manifest_sequences(
        manifest,
        data_root=data_root,
        allowed_splits={"train", "development val"},
        start_sequence=1,
        end_sequence=999,
    )

    assert tuple(selected) == ("seq_022",)


def test_radar_health_accepts_legacy_state_dict_envelope() -> None:
    from radar_bev.models.backbones import extract_dream_bev_state

    state = {"encoder.weight": object()}

    assert extract_dream_bev_state({"model": state}) is state
    assert extract_dream_bev_state({"model": "dream_bev", "model_state": state}) is state
    assert extract_dream_bev_state({"model": "renamed_checkpoint", "model_state": state}) is state
    with pytest.raises(ValueError, match="does not contain a state dictionary"):
        extract_dream_bev_state({"model": "other_backbone"})


def test_fusion_loader_uses_the_same_checkpoint_envelope_contract() -> None:
    module = _load_script("train_fusion.py")
    from radar_bev.models.backbones import extract_dream_bev_state

    state = {"stem.0.weight": object()}

    assert module.extract_dream_bev_state is extract_dream_bev_state
    assert "radar_family" not in inspect.signature(module.load_radar_adapter).parameters


def test_train_condition_predictions_override_annotation_members(tmp_path: Path) -> None:
    module = _load_script("train_fusion.py")
    predictions = tmp_path / "train_policy.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image": "Data_train/seq_022/RGB/frame_0000.jpg",
                "prediction": {
                    "expert_route": "outdoor_wet_return",
                    "visual_condition": "normal_visibility",
                    "visual_reliability_prior": "usable",
                    "weather_surface_prior": "wet_return",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    timeline, source = module.resolve_train_condition_timeline(
        condition_zip=tmp_path / "unused.zip",
        train_condition_members=["annotations/train.jsonl"],
        train_condition_predictions=predictions,
    )

    assert source == "condition_policy_predictions"
    assert timeline.lookup("seq_022", 0).global_route == "outdoor_wet_return"


def test_train_condition_predictions_do_not_require_unused_annotation_zip(
    tmp_path: Path,
) -> None:
    module = _load_script("train_fusion.py")
    predictions = tmp_path / "train_policy.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "image": "Data_train/seq_022/RGB/frame_0000.jpg",
                "prediction": {
                    "expert_route": "outdoor_wet_return",
                    "visual_condition": "normal_visibility",
                    "visual_reliability_prior": "usable",
                    "weather_surface_prior": "wet_return",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    timeline, source = module.resolve_train_condition_timeline(
        condition_zip=None,
        train_condition_members=["annotations/train.jsonl"],
        train_condition_predictions=predictions,
    )

    assert source == "condition_policy_predictions"
    assert timeline.lookup("seq_022", 0).global_route == "outdoor_wet_return"


def test_annotation_conditions_require_a_zip_when_no_policy_is_provided() -> None:
    module = _load_script("train_fusion.py")

    with pytest.raises(ValueError, match="condition-zip is required"):
        module.resolve_train_condition_timeline(
            condition_zip=None,
            train_condition_members=["annotations/train.jsonl"],
            train_condition_predictions=None,
        )


def test_quantized_residual_checkpoint_allowlist_does_not_require_lattice_heads() -> None:
    module = _load_script("train_fusion.py")
    from rvfusion.quantized_lateral_residual import QuantizedLateralResidual

    allowed = module.checkpoint_allowed_missing_keys(
        QuantizedLateralResidual(radius_m=0.5, step_m=0.025),
        skipped_shape_mismatches=set(),
    )

    assert "utility_head.weight" in allowed
    assert not any(key.startswith("route_utility_heads.") for key in allowed)
    assert not any(
        key.startswith("posterior_route_emission_heads.") for key in allowed
    )
