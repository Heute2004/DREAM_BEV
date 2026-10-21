from __future__ import annotations

import sys

import pytest

from radar_bev.protocol import DEV_SPLIT
from scripts.train_fusion import (
    parse_args,
    validate_development_only_execution,
    validate_posterior_operator_study,
)


def _parse(monkeypatch: pytest.MonkeyPatch, core: str):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fusion.py",
            "--phase", "joint",
            "--data-root", "data",
            "--manifest", "manifest.csv",
            "--visual-cache-root", "visual",
            "--radar-health-cache-root", "health",
            "--condition-zip", "condition.zip",
            "--base-checkpoint", "base.pt",
            "--output-dir", "out",
            "--fusion-core", core,
            "--development-only",
        ],
    )
    return parse_args()


@pytest.mark.parametrize(
    "core",
    ("quantized_lateral_residual", "local_posterior"),
)
def test_posterior_study_entry_accepts_development_arms(
    monkeypatch: pytest.MonkeyPatch,
    core: str,
) -> None:
    args = _parse(monkeypatch, core)
    assert args.fusion_core == core
    assert args.development_only


def test_posterior_study_entry_accepts_measured_peak_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fusion.py",
            "--phase", "joint",
            "--data-root", "data",
            "--manifest", "manifest.csv",
            "--visual-cache-root", "visual",
            "--radar-health-cache-root", "health",
            "--condition-zip", "condition.zip",
            "--base-checkpoint", "base.pt",
            "--output-dir", "out",
            "--candidate-geometry", "measured_peaks",
            "--development-only",
        ],
    )

    args = parse_args()

    assert args.candidate_geometry == "measured_peaks"


def test_policy_prediction_mode_does_not_require_annotation_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_fusion.py",
            "--phase", "candidate",
            "--data-root", "data",
            "--manifest", "manifest.csv",
            "--visual-cache-root", "visual",
            "--radar-health-cache-root", "health",
            "--train-condition-predictions", "train_policy.jsonl",
            "--eval-condition-predictions", "development_policy.jsonl",
            "--base-checkpoint", "base.pt",
            "--output-dir", "out",
            "--evaluation-only",
            "--development-only",
        ],
    )

    args = parse_args()

    assert args.condition_zip is None


def test_development_only_contract_rejects_locked_execution() -> None:
    with pytest.raises(ValueError, match="Development-only"):
        validate_development_only_execution(
            reference_split="Locked Test",
            evaluation_only=False,
            locked_test_release_record=None,
        )
    with pytest.raises(ValueError, match="Development-only"):
        validate_development_only_execution(
            reference_split=DEV_SPLIT,
            evaluation_only=False,
            locked_test_release_record="locked_release.json",
        )


def test_development_only_contract_accepts_development_training() -> None:
    validate_development_only_execution(
        reference_split=DEV_SPLIT,
        evaluation_only=False,
        locked_test_release_record=None,
    )


def test_development_only_contract_accepts_development_reload() -> None:
    validate_development_only_execution(
        reference_split=DEV_SPLIT,
        evaluation_only=True,
        locked_test_release_record=None,
    )


def test_no_op_operator_removal_accepts_only_frozen_development_replay() -> None:
    validate_posterior_operator_study(
        fusion_core="local_posterior",
        initial_checkpoint="posterior.pt",
        evaluation_only=True,
        development_only=True,
        disable_posterior_no_op=True,
    )


@pytest.mark.parametrize(
    ("fusion_core", "initial_checkpoint", "evaluation_only", "development_only"),
    (
        ("quantized_lateral_residual", "posterior.pt", True, True),
        ("local_posterior", None, True, True),
        ("local_posterior", "posterior.pt", False, True),
        ("local_posterior", "posterior.pt", True, False),
    ),
)
def test_no_op_operator_removal_rejects_any_non_frozen_development_path(
    fusion_core: str,
    initial_checkpoint: str | None,
    evaluation_only: bool,
    development_only: bool,
) -> None:
    with pytest.raises(ValueError, match="no-op operator study"):
        validate_posterior_operator_study(
            fusion_core=fusion_core,
            initial_checkpoint=initial_checkpoint,
            evaluation_only=evaluation_only,
            development_only=development_only,
            disable_posterior_no_op=True,
        )
