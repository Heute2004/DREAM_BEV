from __future__ import annotations

from rvfusion.selectivity_audit import summarize_selectivity_rows


def _row(
    *,
    sequence: str = "seq_001",
    assist: bool,
    accepted: bool | None = None,
    radar_x: float,
    candidate_x: float,
    target_x: float,
    selected_index: int = 1,
    fused_x: float | None = None,
) -> dict[str, str]:
    row = {
        "sequence": sequence,
        "frame_id": "0",
        "target_x_m": str(target_x),
        "target_y_m": "1.0",
        "radar_x_m": str(radar_x),
        "radar_y_m": "1.0",
        "candidate_x_m": str(candidate_x),
        "candidate_y_m": "1.0",
        "assist_enabled": str(assist),
        "selected_index": str(selected_index),
    }
    if fused_x is not None:
        row["fused_x_m"] = str(fused_x)
        row["fused_y_m"] = "1.0"
    row["accepted"] = str(assist if accepted is None else accepted)
    return row


def test_selectivity_audit_uses_deployed_accepted_gate_not_health_diagnostic() -> None:
    summary = summarize_selectivity_rows(
        [
            # Health can be false while the deployed compositor still applies vision.
            _row(
                assist=False,
                accepted=True,
                radar_x=0.0,
                candidate_x=0.2,
                target_x=0.2,
            ),
            # Health can be true while the deployed compositor falls back to radar.
            _row(
                assist=True,
                accepted=False,
                radar_x=0.0,
                candidate_x=0.2,
                target_x=0.0,
            ),
        ],
        error_tolerance_m=1e-6,
        noop_tolerance_m=1e-3,
    )

    assert summary["policy_identity"]["frames"] == 1
    assert summary["visual_admitted"]["frames"] == 1
    assert summary["final_outcome"]["improved_frames"] == 1
    assert summary["final_outcome"]["unchanged_frames"] == 1


def test_selectivity_audit_requires_deployed_gate() -> None:
    row = _row(assist=False, radar_x=0.0, candidate_x=0.1, target_x=0.0)
    row.pop("accepted")

    import pytest

    with pytest.raises(KeyError, match="accepted"):
        summarize_selectivity_rows([row])


def test_selectivity_audit_separates_policy_identity_and_local_noop() -> None:
    summary = summarize_selectivity_rows(
        [
            _row(assist=False, radar_x=0.0, candidate_x=0.4, target_x=0.4),
            _row(assist=True, radar_x=0.0, candidate_x=0.2, target_x=0.2),
            _row(assist=True, radar_x=0.0, candidate_x=-0.2, target_x=0.2),
            _row(
                assist=True,
                radar_x=0.0,
                candidate_x=0.0005,
                target_x=0.0,
                selected_index=0,
            ),
        ],
        error_tolerance_m=1e-6,
        noop_tolerance_m=1e-3,
    )

    assert summary["frames"] == 4
    assert summary["policy_identity"]["frames"] == 1
    assert summary["visual_admitted"]["frames"] == 3
    assert summary["visual_admitted"]["selected_noop_frames"] == 1
    assert summary["final_outcome"]["improved_frames"] == 1
    assert summary["final_outcome"]["harmed_frames"] == 2
    assert summary["final_outcome"]["unchanged_frames"] == 1
    assert summary["final_update"]["near_noop_frames"] == 2
    assert summary["final_update"]["max_abs_delta_x_m"] == 0.2


def test_selectivity_audit_reports_per_sequence_counts() -> None:
    summary = summarize_selectivity_rows(
        [
            _row(sequence="seq_001", assist=False, radar_x=0.0, candidate_x=0.1, target_x=0.0),
            _row(sequence="seq_002", assist=True, radar_x=0.0, candidate_x=0.1, target_x=0.1),
        ]
    )

    assert summary["per_sequence"]["seq_001"]["policy_identity_frames"] == 1
    assert summary["per_sequence"]["seq_002"]["improved_frames"] == 1


def test_selectivity_audit_uses_final_composited_prediction_when_available() -> None:
    summary = summarize_selectivity_rows(
        [
            _row(
                assist=True,
                accepted=True,
                radar_x=0.0,
                candidate_x=0.4,
                fused_x=0.0,
                target_x=0.0,
            )
        ],
        error_tolerance_m=1e-6,
        noop_tolerance_m=1e-3,
    )

    assert summary["final_outcome"]["unchanged_frames"] == 1
    assert summary["final_outcome"]["harmed_frames"] == 0
