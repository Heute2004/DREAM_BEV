from __future__ import annotations

from pathlib import Path


def test_policy_backend_frontier_replays_one_deployed_rvfusion_contract() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_policy_backend_frontier_dev.sh"
    )
    source = script.read_text(encoding="utf-8")
    assert "policy frontier runner refuses Locked Test paths" in source
    assert "VLM_DEV_POLICY" in source
    assert "MOBILENET_DEV_POLICY" in source
    assert "RESNET_DEV_POLICY" in source
    assert 'scripts/train_fusion.py' in source
    assert "--phase candidate" in source
    assert "--acceptance-mode direct_utility" in source
    assert "--correction-source lattice" in source
    assert "--development-only" in source


def test_policy_backend_frontier_replays_the_same_guard_after_evaluation() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_policy_backend_frontier_dev.sh"
    )
    source = script.read_text(encoding="utf-8")
    assert "replay_causal_cross_modal_guard.py" in source
    assert "--ema-alpha 0.10" in source
    assert "--ema-threshold 0.50" in source
    assert "--policy-audit \"${audit}\"" in source
    assert "--evidence-audit \"${audit}\"" in source
    assert "policy_backend_frontier_summary.json" in source
