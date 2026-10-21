from rvfusion.guard_calibration import CalibrationRow, choose_guard_threshold


def test_selects_smallest_threshold_that_preserves_clean_gain_and_biased_floor() -> None:
    selected = choose_guard_threshold(
        [
            CalibrationRow(0.5, clean_ap30=0.86, biased_ap30=0.69),
            CalibrationRow(0.8, clean_ap30=0.88, biased_ap30=0.72),
            CalibrationRow(1.0, clean_ap30=0.90, biased_ap30=0.74),
        ],
        radar_clean_ap30=0.86,
        radar_biased_ap30=0.72,
        minimum_clean_gain=0.01,
    )

    assert selected is not None
    assert selected.threshold == 0.8


def test_returns_none_when_no_threshold_satisfies_the_predeclared_contract() -> None:
    selected = choose_guard_threshold(
        [
            CalibrationRow(0.5, clean_ap30=0.85, biased_ap30=0.74),
            CalibrationRow(0.8, clean_ap30=0.90, biased_ap30=0.71),
        ],
        radar_clean_ap30=0.86,
        radar_biased_ap30=0.72,
        minimum_clean_gain=0.01,
    )

    assert selected is None
