from __future__ import annotations

import copy

import pytest

from rvfusion.route_counterfactual import (
    RouteCounterfactualError,
    force_expert_route,
)


def _record(*, route: str, admission: bool = True) -> dict[str, object]:
    return {
        "id": "causal_policy_seq_082_frame_0032",
        "image": "Data_train/seq_082/RGB/frame_0032.jpg",
        "prediction": {
            "expert_route": route,
            "expert_route_basis": "current_left_rgb_estimate",
            "visual_condition": "normal",
            "visual_reliability_prior": "usable",
            "weather_surface_prior": "dry",
            "rationale_zh": "保留作审计。",
        },
        "policy_capture_frame_id": 0,
        "policy_available_frame_id": 32,
        "policy_segment_start_frame": 32,
        "policy_segment_end_frame": 61,
        "segment_rvfusion_admission": admission,
        "policy_source": "causal_response",
    }


def test_forced_route_changes_only_the_expert_route() -> None:
    original = [_record(route="open_indoor_los"), _record(route="indoor_clutter")]
    snapshot = copy.deepcopy(original)

    forced = force_expert_route(original, "outdoor_wet_return")

    assert original == snapshot
    assert [row["prediction"]["expert_route"] for row in forced] == [
        "outdoor_wet_return",
        "outdoor_wet_return",
    ]
    for before, after in zip(snapshot, forced):
        expected = copy.deepcopy(before)
        expected["prediction"]["expert_route"] = "outdoor_wet_return"
        assert after == expected


def test_forced_route_rejects_unmaterialized_policy_records() -> None:
    record = _record(route="open_indoor_los")
    record.pop("segment_rvfusion_admission")

    with pytest.raises(RouteCounterfactualError, match="materialized"):
        force_expert_route([record], "indoor_clutter")


def test_forced_route_rejects_an_unknown_target_route() -> None:
    with pytest.raises(RouteCounterfactualError, match="unsupported"):
        force_expert_route([_record(route="open_indoor_los")], "not_a_route")


def test_forced_route_preserves_causal_identity_warmup() -> None:
    warmup = _record(route="unknown", admission=False)
    warmup["policy_source"] = "causal_warmup_identity"

    forced = force_expert_route([warmup], "outdoor_wet_return")

    assert forced == [warmup]
