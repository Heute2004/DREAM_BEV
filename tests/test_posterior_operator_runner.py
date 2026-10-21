from __future__ import annotations

from pathlib import Path


def test_posterior_operator_runner_is_development_only_and_uses_frozen_controls() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_posterior_operator_ablation_dev.sh"
    )

    source = script.read_text(encoding="utf-8")

    assert "Locked Test is forbidden" in source
    assert "--development-only" in source
    assert "--disable-posterior-no-op" in source
    assert "quantized_lateral_residual" in source
    assert "replay_causal_cross_modal_guard.py" in source
    assert "POLICY_CAUSAL_REPLAY" in source
    assert "test ! -e" in source
