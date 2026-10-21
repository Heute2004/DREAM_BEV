from __future__ import annotations

from rvfusion.route_matrix import (
    evaluate_audit_by_ledger_route,
    evaluate_audit_by_reference_route,
)


def _trace_record(frame: int, route: str) -> dict[str, object]:
    return {
        "image": f"Data_train/seq_082/RGB/frame_{frame:04d}.jpg",
        "prediction": {"expert_route": route},
        "policy_available_frame_id": frame,
        "segment_rvfusion_admission": route != "unknown",
    }


def _audit_row(
    frame: int,
    *,
    accepted: bool,
    radar_x: float,
    candidate_x: float,
    target_x: float,
) -> dict[str, str]:
    return {
        "sequence": "seq_082",
        "frame_id": str(frame),
        "target_x_m": str(target_x),
        "target_y_m": "2.0",
        "radar_x_m": str(radar_x),
        "radar_y_m": "2.0",
        "candidate_x_m": str(candidate_x),
        "candidate_y_m": "2.0",
        "radar_score": "0.9",
        "accepted": str(accepted),
    }


def test_route_matrix_uses_original_route_and_deployed_decoded_coordinate() -> None:
    reference_trace = [
        _trace_record(0, "unknown"),
        _trace_record(32, "open_indoor_los"),
        _trace_record(62, "indoor_clutter"),
    ]
    audit_rows = [
        _audit_row(
            35,
            accepted=True,
            radar_x=0.6,
            candidate_x=0.1,
            target_x=0.1,
        ),
        _audit_row(
            65,
            accepted=False,
            radar_x=0.2,
            candidate_x=0.8,
            target_x=0.2,
        ),
    ]

    result = evaluate_audit_by_reference_route(audit_rows, reference_trace)

    assert set(result) == {"open_indoor_los", "indoor_clutter"}
    assert result["open_indoor_los"]["frames"] == 1.0
    assert result["open_indoor_los"]["mean_location_error_m"] == 0.0
    assert result["indoor_clutter"]["mean_location_error_m"] == 0.0


def test_route_matrix_does_not_score_identity_warmup_as_a_mechanism() -> None:
    reference_trace = [_trace_record(0, "unknown")]
    audit_rows = [
        _audit_row(
            10,
            accepted=False,
            radar_x=0.2,
            candidate_x=0.7,
            target_x=0.2,
        )
    ]

    assert evaluate_audit_by_reference_route(audit_rows, reference_trace) == {}


def test_route_matrix_can_group_by_frozen_sequence_ledger_not_policy_output() -> None:
    reference_trace = [
        _trace_record(0, "unknown"),
        _trace_record(32, "open_indoor_los"),
    ]
    audit_rows = [
        _audit_row(
            35,
            accepted=True,
            radar_x=0.6,
            candidate_x=0.1,
            target_x=0.1,
        )
    ]

    result = evaluate_audit_by_ledger_route(
        audit_rows,
        reference_trace,
        {"seq_082": "outdoor_wet_return"},
    )

    assert set(result) == {"outdoor_wet_return"}
    assert result["outdoor_wet_return"]["frames"] == 1.0
    assert result["outdoor_wet_return"]["ap_0p3m"] == 1.0
