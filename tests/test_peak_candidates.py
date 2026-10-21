from __future__ import annotations

import torch

from rvfusion.contracts import (
    ConditionPrior,
    RadarEvidence,
    TemporalBearingEvidence,
)
from rvfusion.lattice_fusion import DenseLatticeRVFusion
from rvfusion.peak_candidates import (
    SOURCE_DIRECT,
    SOURCE_NOOP,
    SOURCE_PEAK,
    SOURCE_VISUAL,
    build_peak_candidate_set,
    build_peak_x_lattice,
)
from rvfusion.radar_health import PersistentRadarState


def _probability_with_two_local_peaks() -> torch.Tensor:
    probability = torch.full((1, 1, 200, 234), 1e-8)
    probability[0, 0, 102, 40] = 0.40
    probability[0, 0, 98, 40] = 0.30
    probability[0, 0, 100, 40] = 0.20
    return probability / probability.sum()


def test_peak_candidates_keep_noop_measured_peaks_and_proposals() -> None:
    candidates = build_peak_candidate_set(
        radar_probability=_probability_with_two_local_peaks(),
        radar_xy_m=torch.tensor([[0.0, 4.0]]),
        visual_x_m=torch.tensor([[0.30]]),
        direct_x_m=torch.tensor([[-0.30]]),
        radius_m=0.5,
        top_k=2,
    )

    assert candidates.source_id[0, 0].item() == SOURCE_NOOP
    assert (candidates.source_id == SOURCE_PEAK).sum().item() == 2
    assert (candidates.source_id == SOURCE_VISUAL).any()
    assert (candidates.source_id == SOURCE_DIRECT).any()
    assert torch.equal(candidates.y_m, torch.full_like(candidates.y_m, 4.0))


def test_peak_candidates_deduplicate_proposals_but_preserve_noop() -> None:
    candidates = build_peak_candidate_set(
        radar_probability=_probability_with_two_local_peaks(),
        radar_xy_m=torch.tensor([[0.0, 4.0]]),
        visual_x_m=torch.tensor([[0.20]]),
        direct_x_m=torch.tensor([[0.20]]),
        radius_m=0.5,
        top_k=2,
    )

    assert candidates.valid_mask[0, 0]
    assert candidates.valid_mask.sum().item() == 4


def test_peak_lattice_retains_radar_range_contract() -> None:
    lattice = build_peak_x_lattice(
        radar_probability=_probability_with_two_local_peaks(),
        radar_xy_m=torch.tensor([[0.0, 4.0]]),
        visual_x_m=torch.tensor([[0.30]]),
        direct_x_m=torch.tensor([[-0.30]]),
        visual_sigma_x_m=torch.tensor([[0.10]]),
        radius_m=0.5,
        top_k=2,
    )

    assert lattice.no_op_index == 0
    assert lattice.valid_mask.sum().item() == 5
    assert lattice.source_id[0, 0].item() == SOURCE_NOOP


def test_peak_posterior_runs_the_common_fusion_contract() -> None:
    logits = torch.full((2, 1, 200, 234), -6.0)
    logits[:, 0, 100, 40] = 6.0
    logits[:, 0, 102, 40] = 5.0
    radar = RadarEvidence(
        polar_feature=torch.randn(2, 48, 128, 128),
        bev_feature=torch.randn(2, 128, 50, 59),
        logits=logits,
    )
    bearing = TemporalBearingEvidence(
        theta=torch.full((2, 1), 0.03),
        sigma_theta=torch.full((2, 1), 0.04),
        temporal_feature=torch.randn(2, 64),
        reliability=torch.ones(2, 1),
        valid=torch.ones(2, dtype=torch.bool),
    )
    condition = ConditionPrior(
        global_route=("indoor_clutter", "outdoor_dry_complex"),
        visual_condition=("normal_visibility",) * 2,
        visual_reliability=("usable",) * 2,
        weather_surface=("indoor", "dry"),
    )
    state = PersistentRadarState(
        assist_enabled=torch.ones(2, dtype=torch.bool),
        uncertainty_strength=torch.ones(2),
    )

    output = DenseLatticeRVFusion(candidate_geometry="measured_peaks")(
        radar, bearing, condition, state
    )

    assert output.candidate_x_m.shape == (2, 7)
    assert torch.equal(output.decoded_xy_m[:, 1], output.radar_xy_m[:, 1])
