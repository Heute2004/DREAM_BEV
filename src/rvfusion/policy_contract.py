from __future__ import annotations

import math


def expected_slow_policy_records(
    *, sequence_count: int, frames_per_sequence: int, hold_frames: int
) -> int:
    """Return the number of causal slow-policy records for fixed-length sequences."""
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    if frames_per_sequence <= 0:
        raise ValueError("frames_per_sequence must be positive")
    if hold_frames <= 0:
        raise ValueError("hold_frames must be positive")
    return sequence_count * math.ceil(frames_per_sequence / hold_frames)



