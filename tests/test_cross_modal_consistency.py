from __future__ import annotations

import torch
import pytest

from rvfusion.contracts import (
    ConditionPrior,
    RadarEvidence,
    TemporalBearingEvidence,
)
from rvfusion.cross_modal_consistency import (
    CausalSignedInnovationGuard,
    cross_modal_innovation,
    radar_lateral_sigma,
    should_accept_visual_evidence,
)
from rvfusion.lattice_fusion import (
    DenseLatticeRVFusion,
    apply_causal_cross_modal_fallback,
)
from rvfusion.radar_health import PersistentRadarState


def _radar_slice(*, peak_index: int, secondary_index: int | None = None) -> torch.Tensor:
    probability = torch.full((1, 1, 200, 234), 1e-9)
    probability[0, 0, peak_index, 40] = 0.90
    if secondary_index is not None:
        probability[0, 0, secondary_index, 40] = 0.85
    return probability / probability.sum()


def test_consistent_visual_evidence_passes_against_a_sharp_radar_slice() -> None:
    sigma_radar = radar_lateral_sigma(
        _radar_slice(peak_index=100),
        torch.tensor([[0.0, 4.0]]),
        radius_m=0.5,
    )
    innovation = cross_modal_innovation(
        visual_x_m=torch.tensor([[0.06]]),
        visual_sigma_x_m=torch.tensor([[0.08]]),
        radar_x_m=torch.tensor([[0.0]]),
        radar_sigma_x_m=sigma_radar,
    )

    assert sigma_radar.item() >= 0.05
    assert should_accept_visual_evidence(innovation, threshold=3.0).item()


def test_large_visual_bias_is_rejected_against_a_sharp_radar_slice() -> None:
    sigma_radar = radar_lateral_sigma(
        _radar_slice(peak_index=100),
        torch.tensor([[0.0, 4.0]]),
        radius_m=0.5,
    )
    innovation = cross_modal_innovation(
        visual_x_m=torch.tensor([[0.45]]),
        visual_sigma_x_m=torch.tensor([[0.08]]),
        radar_x_m=torch.tensor([[0.0]]),
        radar_sigma_x_m=sigma_radar,
    )

    assert innovation.item() > 3.0
    assert not should_accept_visual_evidence(innovation, threshold=3.0).item()


def test_causal_guard_allows_transient_disagreement_but_rejects_persistent_bias() -> None:
    guard = CausalSignedInnovationGuard(
        ema_alpha=0.1,
        ema_threshold=0.8,
    )
    eligible = torch.ones(5, dtype=torch.bool)
    passed = guard.step(
        sequences=("seq_a",) * 5,
        frame_ids=(0, 1, 2, 3, 4),
        signed_innovation=torch.tensor([[0.3], [-0.2], [3.0], [3.0], [3.0]]),
        eligible=eligible,
    )

    assert passed[:4].all()
    assert not passed[4].item()


def test_causal_guard_tracks_sequences_independently_and_rejects_reordered_frames() -> None:
    guard = CausalSignedInnovationGuard(
        ema_alpha=0.5,
        ema_threshold=0.8,
    )
    passed = guard.step(
        sequences=("seq_a", "seq_b", "seq_a", "seq_b"),
        frame_ids=(0, 0, 1, 1),
        signed_innovation=torch.tensor([[2.0], [-2.0], [2.0], [-2.0]]),
        eligible=torch.ones(4, dtype=torch.bool),
    )

    assert torch.equal(passed, torch.tensor([False, False, False, False]))
    with pytest.raises(ValueError, match="strictly increasing"):
        guard.step(
            sequences=("seq_a",),
            frame_ids=(1,),
            signed_innovation=torch.tensor([[0.1]]),
            eligible=torch.ones(1, dtype=torch.bool),
        )


def test_broad_radar_evidence_increases_tolerance_without_disabling_guard() -> None:
    sharp_sigma = radar_lateral_sigma(
        _radar_slice(peak_index=100),
        torch.tensor([[0.0, 4.0]]),
        radius_m=0.5,
    )
    broad_sigma = radar_lateral_sigma(
        _radar_slice(peak_index=100, secondary_index=105),
        torch.tensor([[0.0, 4.0]]),
        radius_m=0.5,
    )
    sharp_innovation = cross_modal_innovation(
        visual_x_m=torch.tensor([[0.26]]),
        visual_sigma_x_m=torch.tensor([[0.08]]),
        radar_x_m=torch.tensor([[0.0]]),
        radar_sigma_x_m=sharp_sigma,
    )
    broad_innovation = cross_modal_innovation(
        visual_x_m=torch.tensor([[0.26]]),
        visual_sigma_x_m=torch.tensor([[0.08]]),
        radar_x_m=torch.tensor([[0.0]]),
        radar_sigma_x_m=broad_sigma,
    )

    assert broad_sigma.item() > sharp_sigma.item()
    assert broad_innovation.item() < sharp_innovation.item()
    assert should_accept_visual_evidence(broad_innovation, threshold=3.0).item()


def test_segment_admitted_visual_correction_falls_back_on_large_innovation() -> None:
    logits = torch.full((1, 1, 200, 234), -8.0)
    logits[0, 0, 100, 40] = 8.0
    radar = RadarEvidence(
        polar_feature=torch.randn(1, 48, 128, 128),
        bev_feature=torch.randn(1, 128, 50, 59),
        logits=logits,
    )
    bearing = TemporalBearingEvidence(
        theta=torch.tensor([[0.12]]),
        sigma_theta=torch.tensor([[0.01]]),
        temporal_feature=torch.randn(1, 96),
        reliability=torch.ones(1, 1),
        valid=torch.ones(1, dtype=torch.bool),
    )
    condition = ConditionPrior(
        global_route=("indoor_clutter",),
        visual_condition=("normal_visibility",),
        visual_reliability=("usable",),
        weather_surface=("indoor",),
    )
    state = PersistentRadarState(
        assist_enabled=torch.ones(1, dtype=torch.bool),
        uncertainty_strength=torch.ones(1),
    )
    model = DenseLatticeRVFusion(assist_policy="vlm_segment")

    output = model(
        radar,
        bearing,
        condition,
        state,
        cross_modal_guard_z_threshold=3.0,
    )

    assert not output.accepted.item()
    assert torch.equal(output.decoded_xy_m, output.radar_xy_m)
    assert torch.equal(output.fused_probability, output.radar_probability)


def test_causal_guard_composes_exact_radar_identity_without_changing_range() -> None:
    logits = torch.full((1, 1, 200, 234), -8.0)
    logits[0, 0, 100, 40] = 8.0
    radar = RadarEvidence(
        polar_feature=torch.randn(1, 48, 128, 128),
        bev_feature=torch.randn(1, 128, 50, 59),
        logits=logits,
    )
    bearing = TemporalBearingEvidence(
        theta=torch.tensor([[0.02]]),
        sigma_theta=torch.tensor([[0.01]]),
        temporal_feature=torch.randn(1, 96),
        reliability=torch.ones(1, 1),
        valid=torch.ones(1, dtype=torch.bool),
    )
    condition = ConditionPrior(
        global_route=("indoor_clutter",),
        visual_condition=("normal_visibility",),
        visual_reliability=("usable",),
        weather_surface=("indoor",),
    )
    state = PersistentRadarState(
        assist_enabled=torch.ones(1, dtype=torch.bool),
        uncertainty_strength=torch.ones(1),
    )
    output = DenseLatticeRVFusion(assist_policy="vlm_segment")(
        radar,
        bearing,
        condition,
        state,
    )
    guarded = apply_causal_cross_modal_fallback(
        output,
        torch.zeros(1, dtype=torch.bool),
    )

    assert not guarded.accepted.item()
    assert torch.equal(guarded.decoded_xy_m, guarded.radar_xy_m)
    assert torch.equal(guarded.fused_probability, guarded.radar_probability)
