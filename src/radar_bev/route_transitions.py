from __future__ import annotations

from collections.abc import Mapping


def transition_source_frame(
    frame_id: int,
    *,
    boundary_frame: int,
) -> tuple[int, int]:
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if boundary_frame < 1:
        raise ValueError("boundary_frame must be positive")
    if frame_id < boundary_frame:
        return 0, frame_id
    return 1, frame_id - boundary_frame


def causal_capture_frame(
    frame_id: int,
    *,
    availability_delay_frames: int = 32,
    hold_frames: int = 30,
) -> int | None:
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if availability_delay_frames < 1 or hold_frames < 1:
        raise ValueError("delay and hold must be positive")
    if frame_id < availability_delay_frames:
        return None
    return ((frame_id - availability_delay_frames) // hold_frames) * hold_frames


def delayed_route_for_frame(
    frame_id: int,
    capture_routes: Mapping[int, str],
    *,
    availability_delay_frames: int = 32,
    hold_frames: int = 30,
    identity_route: str = "unknown",
) -> tuple[str, int | None]:
    capture = causal_capture_frame(
        frame_id,
        availability_delay_frames=availability_delay_frames,
        hold_frames=hold_frames,
    )
    if capture is None:
        return identity_route, None
    if capture not in capture_routes:
        raise KeyError(f"missing route at causal capture frame {capture}")
    if capture + availability_delay_frames > frame_id:
        raise AssertionError("future policy capture would be used")
    return str(capture_routes[capture]), capture


def first_stable_route_frame(
    frame_routes: Mapping[int, str],
    *,
    target_route: str,
    boundary_frame: int,
) -> int | None:
    ordered = sorted(frame for frame in frame_routes if frame >= boundary_frame)
    for frame in ordered:
        if frame_routes[frame] != target_route:
            continue
        if all(frame_routes[later] == target_route for later in ordered if later >= frame):
            return frame
    return None
