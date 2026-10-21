from __future__ import annotations

from pathlib import Path


def test_route_counterfactual_runner_requires_frozen_ledger_grouping() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_route_counterfactual_matrix_dev.sh"
    )
    source = script.read_text(encoding="utf-8")
    assert "SEQUENCE_ROUTE_MAP" in source
    assert "--sequence-route-map" in source
    assert "metrics_by_ledger_route.json" in source
    assert "route_counterfactual_matrix_ledger.json" in source
    assert "refuses locked-test paths" in source


def test_route_counterfactual_runner_replays_the_deployed_candidate_contract() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_route_counterfactual_matrix_dev.sh"
    )
    source = script.read_text(encoding="utf-8")
    assert 'scripts/train_fusion.py' in source
    assert '--phase candidate' in source
    assert '--acceptance-mode direct_utility' in source
    assert '--correction-source lattice' in source
    assert '--development-only' in source


def test_route_counterfactual_runner_guards_each_forced_route_output() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_route_counterfactual_matrix_dev.sh"
    )
    source = script.read_text(encoding="utf-8")
    assert "POLICY_CAUSAL_REPLAY" in source
    assert "project_policy_trace_to_evidence.py" in source
    assert "replay_causal_cross_modal_guard.py" in source
    assert "--policy-causal-replay" in source
    assert 'guarded="${output}/guarded"' in source
    assert "causal_guard_replay.csv" in source
