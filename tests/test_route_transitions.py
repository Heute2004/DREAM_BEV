from __future__ import annotations

import pytest

from radar_bev.route_transitions import (
    causal_capture_frame,
    delayed_route_for_frame,
    first_stable_route_frame,
    transition_source_frame,
)


def test_transition_source_frame_maps_both_halves() -> None:
    assert transition_source_frame(149, boundary_frame=150) == (0, 149)
    assert transition_source_frame(150, boundary_frame=150) == (1, 0)
    assert transition_source_frame(299, boundary_frame=150) == (1, 149)


def test_causal_capture_respects_32_frame_delay_and_30_frame_hold() -> None:
    assert causal_capture_frame(31) is None
    assert causal_capture_frame(32) == 0
    assert causal_capture_frame(61) == 0
    assert causal_capture_frame(62) == 30
    assert causal_capture_frame(181) == 120
    assert causal_capture_frame(182) == 150


def test_delayed_route_switches_only_after_captured_transition_is_available() -> None:
    captures = {frame: ("route_a" if frame < 150 else "route_b") for frame in range(0, 300, 30)}
    assert delayed_route_for_frame(31, captures) == ("unknown", None)
    assert delayed_route_for_frame(181, captures) == ("route_a", 120)
    assert delayed_route_for_frame(182, captures) == ("route_b", 150)


def test_missing_capture_fails_closed() -> None:
    with pytest.raises(KeyError):
        delayed_route_for_frame(62, {0: "route_a"})


def test_first_stable_route_frame() -> None:
    routes = {frame: ("a" if frame < 182 else "b") for frame in range(300)}
    assert first_stable_route_frame(routes, target_route="b", boundary_frame=150) == 182
