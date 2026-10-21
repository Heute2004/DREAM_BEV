from __future__ import annotations

import torch

from rvfusion.contracts import (
    ConditionPrior,
    RadarEvidence,
    TemporalBearingEvidence,
)
from rvfusion.quantized_lateral_residual import QuantizedLateralResidual
from rvfusion.quantized_lateral_loss import quantized_lateral_residual_loss
from rvfusion.radar_health import PersistentRadarState


def _inputs(batch: int = 2):
    logits = torch.zeros(batch, 1, 200, 234)
    logits[:, 0, 115, 40] = 5.0
    radar = RadarEvidence(
        polar_feature=torch.randn(batch, 48, 128, 128),
        bev_feature=torch.randn(batch, 128, 50, 59),
        logits=logits,
    )
    bearing = TemporalBearingEvidence(
        theta=torch.full((batch, 1), 0.2),
        sigma_theta=torch.full((batch, 1), 0.04),
        temporal_feature=torch.randn(batch, 96),
        reliability=torch.ones(batch, 1),
        valid=torch.ones(batch, dtype=torch.bool),
    )
    condition = ConditionPrior(
        global_route=("indoor_clutter",) * batch,
        visual_condition=("normal_visibility",) * batch,
        visual_reliability=("usable",) * batch,
        weather_surface=("indoor",) * batch,
    )
    state = PersistentRadarState(
        assist_enabled=torch.zeros(batch, dtype=torch.bool),
        uncertainty_strength=torch.zeros(batch),
    )
    return radar, bearing, condition, state


def test_quantized_lateral_residual_preserves_forward_coordinate_and_uses_fixed_bins() -> None:
    model = QuantizedLateralResidual(radius_m=0.5, step_m=0.025)
    radar, bearing, condition, state = _inputs()

    output = model(radar, bearing, condition, state)

    assert output.candidate_x_m.shape == (2, 41)
    assert torch.equal(output.decoded_xy_m[:, 1], output.radar_xy_m[:, 1])
    assert torch.all(
        (output.decoded_xy_m[:, 0] - output.radar_xy_m[:, 0]).abs() <= 0.5
    )
    assert not hasattr(model, "candidate_attention")


def test_quantized_lateral_residual_honors_identity_override() -> None:
    model = QuantizedLateralResidual(radius_m=0.5, step_m=0.025)
    radar, bearing, condition, state = _inputs()

    output = model(
        radar,
        bearing,
        condition,
        state,
        acceptance_override=torch.zeros(2, 1),
    )

    assert torch.equal(output.decoded_xy_m, output.radar_xy_m)
    assert not output.accepted.any()


def test_quantized_loss_supervises_bin_and_within_bin_residual() -> None:
    model = QuantizedLateralResidual(radius_m=0.5, step_m=0.025)
    radar, bearing, condition, state = _inputs()
    output = model(radar, bearing, condition, state)
    target = output.radar_xy_m.detach().clone()
    target[:, 0] += 0.137

    loss, terms = quantized_lateral_residual_loss(
        output,
        target,
        torch.ones(target.shape[0]),
        radius_m=0.5,
        step_m=0.025,
    )
    loss.backward()

    assert set(terms) == {"quantized_classification", "quantized_residual", "quantized_ap30_margin"}
    assert model.class_head.weight.grad is not None
    assert model.within_bin_head.weight.grad is not None
