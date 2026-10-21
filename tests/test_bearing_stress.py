from __future__ import annotations

import numpy as np
import pytest

from rvfusion.bearing_stress import (
    BearingStressError,
    apply_bearing_bias,
    apply_temporal_lag,
    apply_visual_dropout,
)


def sample_cache() -> dict[str, np.ndarray]:
    return {
        "frame_ids": np.arange(4, dtype=np.int64),
        "theta": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "sigma_theta": np.full(4, 0.1, dtype=np.float32),
        "reliability": np.ones(4, dtype=np.float32),
        "temporal_feature": np.arange(8, dtype=np.float32).reshape(4, 2),
        "valid": np.ones(4, dtype=np.bool_),
    }


def test_bearing_bias_changes_only_valid_theta() -> None:
    cache = sample_cache()
    cache["valid"][1] = False
    stressed, summary = apply_bearing_bias(cache, bias_rad=0.5)
    assert np.allclose(stressed["theta"], [0.6, 0.2, 0.8, 0.9])
    assert np.array_equal(stressed["valid"], cache["valid"])
    assert summary.affected_frames == 3


def test_temporal_lag_never_uses_future_and_invalidates_prefix() -> None:
    cache = sample_cache()
    stressed, summary = apply_temporal_lag(cache, lag_frames=2)
    assert np.array_equal(stressed["valid"], [False, False, True, True])
    assert np.allclose(stressed["theta"][2:], cache["theta"][:-2])
    assert np.array_equal(stressed["temporal_feature"][:2], np.zeros((2, 2)))
    assert summary.affected_frames == 2


def test_dropout_is_deterministic_and_fail_closed() -> None:
    cache = sample_cache()
    stressed, summary = apply_visual_dropout(cache, probability=1.0, seed=7)
    assert not stressed["valid"].any()
    assert np.allclose(stressed["reliability"], 0.0)
    assert np.allclose(stressed["temporal_feature"], 0.0)
    assert summary.affected_frames == 4


def test_invalid_stress_arguments_are_rejected() -> None:
    with pytest.raises(BearingStressError):
        apply_temporal_lag(sample_cache(), lag_frames=0)
    with pytest.raises(BearingStressError):
        apply_visual_dropout(sample_cache(), probability=1.1, seed=1)
