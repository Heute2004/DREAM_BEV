from pathlib import Path

import pytest

from radar_bev.locked_policy_routes import load_policy_route_schedule


def test_policy_schedule_returns_causal_route_for_each_covered_frame(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            (
                '{"id":"seq_032_0","policy_segment_start_frame":0,"policy_segment_end_frame":31,"prediction":{"expert_route":"unknown"}}',
                '{"id":"seq_032_32","policy_segment_start_frame":32,"policy_segment_end_frame":61,"prediction":{"expert_route":"indoor_clutter"}}',
            )
        ),
        encoding="utf-8",
    )

    schedule = load_policy_route_schedule(trace)

    assert schedule.route_for("seq_032", 0) == "unknown"
    assert schedule.route_for("seq_032", 31) == "unknown"
    assert schedule.route_for("seq_032", 32) == "indoor_clutter"
    assert schedule.route_for("seq_032", 61) == "indoor_clutter"


def test_policy_schedule_rejects_overlapping_segments(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            (
                '{"id":"seq_032_0","policy_segment_start_frame":0,"policy_segment_end_frame":31,"prediction":{"expert_route":"unknown"}}',
                '{"id":"seq_032_20","policy_segment_start_frame":20,"policy_segment_end_frame":61,"prediction":{"expert_route":"indoor_clutter"}}',
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        load_policy_route_schedule(trace)
