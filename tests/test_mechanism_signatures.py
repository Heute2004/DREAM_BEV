from __future__ import annotations

import numpy as np
import pytest

from radar_bev.mechanism_signatures import (
    frame_mechanism_signature,
    normalized_power_entropy,
    route_signature_score,
    separated_peak_margin_db,
    spearman_correlation,
    zscore_signature_rows,
)


def test_separated_peak_margin_excludes_local_peak_neighbourhood() -> None:
    values = np.zeros((9, 9), dtype=np.float64)
    values[4, 4] = 10.0
    values[4, 5] = 9.0
    values[0, 0] = 6.0
    assert separated_peak_margin_db(values, exclusion_radius=1) == pytest.approx(4.0)


def test_entropy_is_low_for_single_peak_and_high_for_uniform_map() -> None:
    peaked = np.full((8, 8), -100.0)
    peaked[3, 3] = 0.0
    uniform = np.zeros((8, 8))
    assert normalized_power_entropy(peaked) < 0.05
    assert normalized_power_entropy(uniform) == pytest.approx(1.0)


def test_frame_signature_tracks_cross_domain_and_temporal_peak_distance() -> None:
    ra = np.zeros((8, 8))
    mti = np.zeros((8, 8))
    ra[2, 2] = 10.0
    mti[5, 6] = 10.0
    signature, peak = frame_mechanism_signature(
        ra, mti, previous_mti_peak=(5, 3)
    )
    assert peak == (5, 6)
    assert signature["ra_mti_peak_distance_bins"] == pytest.approx(5.0)
    assert signature["mti_temporal_peak_jump_bins"] == pytest.approx(3.0)


def test_route_signature_scores_use_predeclared_directions() -> None:
    fields = {
        "ra_peak_to_background_db": -1.0,
        "mti_peak_to_background_db": -1.0,
        "ra_separated_peak_margin_db": -1.0,
        "mti_separated_peak_margin_db": -1.0,
        "ra_normalized_entropy": 1.0,
        "mti_normalized_entropy": 1.0,
        "ra_mti_peak_distance_bins": 1.0,
        "ra_mti_correlation": -1.0,
        "mti_temporal_peak_jump_bins": 1.0,
        "ra_diffuse_fraction_6db": 1.0,
        "mti_diffuse_fraction_6db": 1.0,
    }
    assert route_signature_score("indoor_clutter", fields) > 0
    assert route_signature_score("open_outdoor_weak_return", fields) > 0
    assert route_signature_score("outdoor_wet_return", fields) > 0


def test_robust_standardization_and_spearman() -> None:
    rows = []
    for index in range(5):
        rows.append(
            {
                "ra_peak_to_background_db": index,
                "mti_peak_to_background_db": index,
                "ra_separated_peak_margin_db": index,
                "mti_separated_peak_margin_db": index,
                "ra_normalized_entropy": index,
                "mti_normalized_entropy": index,
                "ra_mti_peak_distance_bins": index,
                "ra_mti_correlation": index,
                "mti_temporal_peak_jump_bins": index,
                "ra_diffuse_fraction_6db": index,
                "mti_diffuse_fraction_6db": index,
            }
        )
    standardized = zscore_signature_rows(rows)
    assert standardized[2]["ra_peak_to_background_db"] == pytest.approx(0.0)
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
