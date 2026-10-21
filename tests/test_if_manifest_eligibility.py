from __future__ import annotations

import pytest

from radar_bev.data import training_eligibility_from_row as radar_eligible
from rvfusion.data import training_eligibility_from_row as fusion_eligible


@pytest.mark.parametrize("resolver", (radar_eligible, fusion_eligible))
def test_if_manifest_uses_later_stage_eligibility_when_canonical_flag_is_absent(
    resolver,
) -> None:
    assert resolver({"m3_later_stage_eligible": "True"})
    assert not resolver({"m3_later_stage_eligible": "False"})


@pytest.mark.parametrize("resolver", (radar_eligible, fusion_eligible))
def test_canonical_eligibility_takes_precedence_over_legacy_flag(resolver) -> None:
    assert not resolver(
        {
            "radar_training_eligible": "False",
            "m3_later_stage_eligible": "True",
        }
    )


@pytest.mark.parametrize("resolver", (radar_eligible, fusion_eligible))
def test_manifest_without_any_eligibility_flag_is_rejected(resolver) -> None:
    with pytest.raises(KeyError, match="eligible"):
        resolver({})
