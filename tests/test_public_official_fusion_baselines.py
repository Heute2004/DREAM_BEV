import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from rvfusion.contracts import RadarEvidence, VisualEvidence
from rvfusion.public_official_fusion import (
    CRNOfficialFusionBaseline,
    PUBLIC_BASELINE_SPECS,
    RCBEVDetOfficialFusionBaseline,
    RCROSNetOfficialFusionBaseline,
    _register_crn_positional_encoding,
    validate_public_source,
)


def _evidence(batch: int = 2) -> tuple[RadarEvidence, VisualEvidence]:
    radar = RadarEvidence(
        polar_feature=torch.randn(batch, 48, 128, 128),
        bev_feature=torch.randn(batch, 128, 50, 59),
        logits=torch.randn(batch, 1, 200, 234),
    )
    visual = VisualEvidence(
        boxes=torch.tensor(
            [[[[40.0, 20.0, 120.0, 180.0]], [[50.0, 20.0, 130.0, 180.0]]]]
        ).expand(batch, -1, -1, -1),
        raw_scores=torch.ones(batch, 2, 1),
        raw_tokens=torch.randn(batch, 2, 1, 128),
        temporal_tokens=torch.randn(batch, 2, 1, 128),
        temporal_support=torch.ones(batch, 2, 1),
        valid_mask=torch.ones(batch, 2, 1, dtype=torch.bool),
    )
    return radar, visual


def test_public_specs_pin_official_repositories_and_commits() -> None:
    assert PUBLIC_BASELINE_SPECS["rcrosnet"].repository_url == (
        "https://github.com/Zhuanglong2/RC-ROSNet.git"
    )
    assert PUBLIC_BASELINE_SPECS["rcrosnet"].commit == (
        "8ff57d49efa90934dca2032adf7f6625b5ad9248"
    )
    assert PUBLIC_BASELINE_SPECS["crn"].commit == (
        "5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d"
    )
    assert PUBLIC_BASELINE_SPECS["rcbevdet"].commit == (
        "15c83ccdd5a8cfd3b7c0390eacd8664cf842d513"
    )


def test_public_source_validation_rejects_missing_repository(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="official source"):
        validate_public_source("rcrosnet", tmp_path / "missing")


def test_crn_loader_registers_official_mmdet_positional_encoding(
    tmp_path: Path, monkeypatch
) -> None:
    module_name = "_dream_bev_crn_mmdet_positional_encoding"
    source = tmp_path / "mmdet" / "models" / "utils" / "positional_encoding.py"
    source.parent.mkdir(parents=True)
    source.write_text("REGISTERED = True\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    first = _register_crn_positional_encoding(tmp_path)
    second = _register_crn_positional_encoding(tmp_path)

    assert first.REGISTERED is True
    assert second is first


class _RCROSCore(nn.Module):
    def forward(self, radar: torch.Tensor, image_to_ra: torch.Tensor) -> torch.Tensor:
        assert tuple(radar.shape[1:]) == (256, 50, 59)
        assert tuple(image_to_ra.shape[1:]) == (3, 50, 59)
        return radar


def test_rcrosnet_wrapper_preserves_single_person_bev_contract() -> None:
    radar, visual = _evidence()
    model = RCROSNetOfficialFusionBaseline(official_core=_RCROSCore())

    output = model(radar, visual)

    assert tuple(output.fused_logits.shape) == (2, 1, 200, 234)
    assert tuple(output.delta_logits.shape) == (2, 1, 200, 234)
    assert output.metadata["official_model"] == "RC-ROSNet"
    assert output.metadata["radar_anchor"] == "frozen_five_expert"
    assert output.metadata["official_fusion_core_modified"] is False


class _CRNCore(nn.Module):
    def forward(self, features: torch.Tensor):
        assert tuple(features.shape[1:]) == (1, 256, 50, 59)
        return features[:, 0], None


class _RCBEVDetCore(nn.Module):
    def forward(self, radar: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        assert tuple(radar.shape[1:]) == (256, 50, 59)
        assert tuple(visual.shape[1:]) == (256, 50, 59)
        return radar + visual


@pytest.mark.parametrize(
    ("model", "official_name"),
    (
        (CRNOfficialFusionBaseline(official_core=_CRNCore()), "CRN"),
        (
            RCBEVDetOfficialFusionBaseline(official_core=_RCBEVDetCore()),
            "RCBEVDet",
        ),
    ),
)
def test_other_official_wrappers_preserve_single_person_bev_contract(
    model: nn.Module, official_name: str
) -> None:
    radar, visual = _evidence()

    output = model(radar, visual)

    assert tuple(output.fused_logits.shape) == (2, 1, 200, 234)
    assert output.metadata["official_model"] == official_name
    assert output.metadata["radar_anchor"] == "frozen_five_expert"
