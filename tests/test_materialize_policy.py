from __future__ import annotations

from scripts.materialize_policy import materialize_delayed_policy_rows
from scripts.validate_policy_contract import _policy_admits, validate_deployed_policy_rows
from rvfusion.data import ConditionTimeline


def _prediction(route: str) -> dict[str, str]:
    return {
        "expert_route": route,
        "expert_route_basis": "current_left_rgb_estimate",
        "visual_condition": "normal_visibility",
        "visual_reliability_prior": "usable",
        "weather_surface_prior": "dry",
        "rationale_zh": "当前 RGB 证据支持该策略。",
    }


def test_delayed_policy_uses_identity_warmup_then_full_held_segments(tmp_path) -> None:
    rows = [
        {
            "id": f"policy_{frame:04d}",
            "image": f"Data_train/seq_082/RGB/frame_{frame:04d}.jpg",
            "prediction": _prediction("open_outdoor_weak_return" if frame == 0 else "indoor_clutter"),
        }
        for frame in range(0, 60, 5)
    ]
    output, audit = materialize_delayed_policy_rows(
        rows,
        frames_per_sequence=60,
        input_step_frames=5,
        capture_interval_frames=20,
        availability_delay_frames=32,
        hold_frames=20,
    )

    assert audit["warmup_policy"] == "radar_only_identity"
    assert audit["active_policy_segments"] == 2
    assert [row["policy_available_frame_id"] for row in output] == [0, 32, 52]

    path = tmp_path / "policy.jsonl"
    path.write_text("\n".join(__import__("json").dumps(row) for row in output) + "\n")
    timeline = ConditionTimeline.from_predictions_jsonl([path])
    assert timeline.lookup("seq_082", 31).global_route == "unknown"
    assert timeline.lookup("seq_082", 32).global_route == "open_outdoor_weak_return"
    assert timeline.lookup("seq_082", 51).global_route == "open_outdoor_weak_return"
    assert timeline.lookup("seq_082", 52).global_route == "indoor_clutter"


def test_delayed_policy_fails_closed_for_invalid_prediction() -> None:
    rows = [
        {
            "id": f"policy_{frame:04d}",
            "image": f"Data_train/seq_082/RGB/frame_{frame:04d}.jpg",
            "prediction": {} if frame == 0 else _prediction("indoor_clutter"),
        }
        for frame in range(0, 35, 5)
    ]
    output, audit = materialize_delayed_policy_rows(
        rows,
        frames_per_sequence=35,
        input_step_frames=5,
        capture_interval_frames=20,
        availability_delay_frames=32,
        hold_frames=20,
    )

    assert audit["invalid_predictions_fail_closed"] == 1
    assert output[1]["prediction"]["expert_route"] == "unknown"


def test_rain_or_wet_condition_is_admitted_by_the_deployed_policy() -> None:
    prediction = _prediction("outdoor_wet_return")
    prediction["visual_condition"] = "rain_or_wet"
    prediction["visual_reliability_prior"] = "cautious"

    assert _policy_admits(
        prediction,
        trust_policy="normal_or_rain_nonunreliable",
    )


def test_rain_or_wet_condition_is_not_silently_downgraded_by_soft_policy() -> None:
    prediction = _prediction("outdoor_wet_return")
    prediction["visual_condition"] = "rain_or_wet"
    prediction["visual_reliability_prior"] = "cautious"

    assert _policy_admits(prediction, trust_policy="soft")


def test_deployed_audit_must_match_materialized_segment_gate() -> None:
    policy_rows = [
        {
            "image": "Data_train/seq_082/RGB/frame_0000.jpg",
            "prediction": _prediction("unknown"),
            "policy_available_frame_id": 0,
            "policy_segment_end_frame_id": 31,
        },
        {
            "image": "Data_train/seq_082/RGB/frame_0032.jpg",
            "prediction": _prediction("indoor_clutter"),
            "policy_available_frame_id": 32,
            "policy_segment_end_frame_id": 59,
        },
    ]
    audit_rows = [
        {
            "sequence": "seq_082",
            "frame_id": frame_id,
            "route": "unknown" if frame_id < 32 else "indoor_clutter",
            "accepted": frame_id >= 32,
        }
        for frame_id in range(60)
    ]

    report = validate_deployed_policy_rows(policy_rows, audit_rows)

    assert report["policy_identity_frames"] == 32
    assert report["policy_admitted_frames"] == 28


def test_deployed_audit_rejects_health_driven_framewise_gate() -> None:
    policy_rows = [
        {
            "image": "Data_train/seq_082/RGB/frame_0000.jpg",
            "prediction": _prediction("indoor_clutter"),
            "policy_available_frame_id": 0,
            "policy_segment_end_frame_id": 1,
        }
    ]
    audit_rows = [
        {"sequence": "seq_082", "frame_id": 0, "route": "indoor_clutter", "accepted": True},
        {"sequence": "seq_082", "frame_id": 1, "route": "indoor_clutter", "accepted": False},
    ]

    try:
        validate_deployed_policy_rows(policy_rows, audit_rows)
    except ValueError as error:
        assert "does not match materialized policy" in str(error)
    else:
        raise AssertionError("expected a framewise policy mismatch to fail")
