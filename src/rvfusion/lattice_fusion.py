from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from .conditions import ROUTES, ConditionEncoder
from .contracts import ConditionPrior, RadarEvidence, TemporalBearingEvidence
from .cross_modal_consistency import (
    cross_modal_innovation,
    radar_lateral_sigma,
    should_accept_visual_evidence,
)
from .local_lattice import (
    build_local_x_lattice,
    build_visual_blend_x_lattice,
)
from .peak_candidates import build_peak_x_lattice
from .metric_contract import resolved_radar_position
from .radar_health import PersistentRadarState


ROUTE_BLEND_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
ROUTE_BLEND_DENSE_FRACTIONS = tuple(index / 20.0 for index in range(21))
ROUTE_BLEND_TARGET = {
    "open_indoor_los": 0.0,
    "indoor_clutter": 0.50,
    "open_outdoor_weak_return": 0.10,
    "outdoor_dry_complex": 0.50,
    "outdoor_wet_return": 0.25,
    "outdoor_adverse": 0.25,
    "corner_nlos": 0.0,
    "unknown": 0.0,
}


def _route_blend_candidate_prior(
    routes: Sequence[str],
    *,
    blend_fractions: tuple[float, ...],
    prior_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    prior = torch.full(
        (len(routes), len(blend_fractions)),
        -float(prior_scale),
        device=device,
        dtype=dtype,
    )
    prior[:, 0] = -20.0
    for row, route in enumerate(routes):
        target = ROUTE_BLEND_TARGET.get(
            str(route).strip().lower(),
            0.0,
        )
        if target == 0.0:
            prior[row, 1:] = -3.5
            continue
        target_index = blend_fractions.index(target)
        prior[row, target_index] = 3.0 * float(prior_scale)
    return prior


def _bearing_likelihood_candidate_prior(
    visual_log_likelihood: Tensor,
    *,
    reliability: Tensor,
    trust_cap: Tensor,
    prior_scale: float,
) -> Tensor:
    """Convert the causal visual bearing distribution into candidate evidence."""
    if visual_log_likelihood.ndim != 2:
        raise ValueError("visual_log_likelihood must be [batch,candidates]")
    batch = visual_log_likelihood.shape[0]
    for name, value in (("reliability", reliability), ("trust_cap", trust_cap)):
        if tuple(value.shape) != (batch, 1):
            raise ValueError(f"{name} must be [batch,1]")
    strength = reliability.clamp(0.0, 1.0) * trust_cap.clamp(0.0, 1.0)
    return (
        float(prior_scale)
        * strength
        * visual_log_likelihood.clamp(-20.0, 0.0)
    )


@dataclass(frozen=True)
class DenseLatticeFusionOutput:
    radar_probability: Tensor
    fused_probability: Tensor
    radar_xy_m: Tensor
    decoded_xy_m: Tensor
    correction_x_m: Tensor
    correction_index: Tensor
    correction_non_noop: Tensor
    correction_source: str
    continuous_x_m: Tensor
    continuous_offset_m: Tensor
    candidate_x_m: Tensor
    candidate_residual_m: Tensor
    candidate_valid_mask: Tensor
    emission_logits: Tensor
    candidate_presence_logit: Tensor
    candidate_presence_probability: Tensor
    candidate_utility_logit: Tensor
    candidate_utility_probability: Tensor
    posterior: Tensor
    selected_index: Tensor
    predicted_gain_logit: Tensor
    predicted_harm_logit: Tensor
    predicted_utility_logit: Tensor
    predicted_gain: Tensor
    predicted_harm: Tensor
    predicted_utility_probability: Tensor
    effective_acceptance: Tensor
    accepted: Tensor
    identity_mask: Tensor
    visual_x_m: Tensor
    calibrated_visual_x_m: Tensor
    candidate_log_sigma_m: Tensor
    radar_sigma_x_m: Tensor
    cross_modal_innovation: Tensor
    cross_modal_guard_passed: Tensor

    def validate(self) -> None:
        batch = self.radar_probability.shape[0]
        map_shape = (batch, 1, 200, 234)
        if tuple(self.radar_probability.shape) != map_shape:
            raise ValueError("radar_probability must be [batch,1,200,234]")
        if tuple(self.fused_probability.shape) != map_shape:
            raise ValueError("fused_probability must be [batch,1,200,234]")
        candidate_shape = self.candidate_x_m.shape
        if candidate_shape[0] != batch:
            raise ValueError("candidate_x_m must start with batch")
        for name, value in (
            ("candidate_residual_m", self.candidate_residual_m),
            ("candidate_valid_mask", self.candidate_valid_mask),
            ("emission_logits", self.emission_logits),
            ("candidate_utility_logit", self.candidate_utility_logit),
            (
                "candidate_utility_probability",
                self.candidate_utility_probability,
            ),
            ("posterior", self.posterior),
            ("candidate_log_sigma_m", self.candidate_log_sigma_m),
        ):
            if tuple(value.shape) != tuple(candidate_shape):
                raise ValueError(f"{name} must match candidate_x_m")
        if self.candidate_valid_mask.dtype != torch.bool:
            raise ValueError("candidate_valid_mask must be boolean")
        if tuple(self.radar_xy_m.shape) != (batch, 2):
            raise ValueError("radar_xy_m must be [batch,2]")
        if tuple(self.decoded_xy_m.shape) != (batch, 2):
            raise ValueError("decoded_xy_m must be [batch,2]")
        if tuple(self.correction_x_m.shape) != (batch, 1):
            raise ValueError("correction_x_m must be [batch,1]")
        if tuple(self.correction_index.shape) != (batch,):
            raise ValueError("correction_index must be [batch]")
        if (
            tuple(self.correction_non_noop.shape) != (batch,)
            or self.correction_non_noop.dtype != torch.bool
        ):
            raise ValueError("correction_non_noop must be boolean [batch]")
        if self.correction_source not in {"lattice", "continuous"}:
            raise ValueError(
                "correction_source must be lattice or continuous"
            )
        if tuple(self.continuous_x_m.shape) != (batch, 1):
            raise ValueError("continuous_x_m must be [batch,1]")
        if tuple(self.continuous_offset_m.shape) != (batch, 1):
            raise ValueError("continuous_offset_m must be [batch,1]")
        if tuple(self.selected_index.shape) != (batch,):
            raise ValueError("selected_index must be [batch]")
        for name, value in (
            ("predicted_gain", self.predicted_gain),
            ("predicted_harm", self.predicted_harm),
            ("predicted_gain_logit", self.predicted_gain_logit),
            ("predicted_harm_logit", self.predicted_harm_logit),
            ("predicted_utility_logit", self.predicted_utility_logit),
            (
                "predicted_utility_probability",
                self.predicted_utility_probability,
            ),
            ("candidate_presence_logit", self.candidate_presence_logit),
            (
                "candidate_presence_probability",
                self.candidate_presence_probability,
            ),
            ("effective_acceptance", self.effective_acceptance),
            ("visual_x_m", self.visual_x_m),
            ("calibrated_visual_x_m", self.calibrated_visual_x_m),
            ("radar_sigma_x_m", self.radar_sigma_x_m),
            ("cross_modal_innovation", self.cross_modal_innovation),
        ):
            if tuple(value.shape) != (batch, 1):
                raise ValueError(f"{name} must be [batch,1]")
        if tuple(self.accepted.shape) != (batch,) or self.accepted.dtype != torch.bool:
            raise ValueError("accepted must be boolean [batch]")
        if tuple(self.identity_mask.shape) != (batch,) or self.identity_mask.dtype != torch.bool:
            raise ValueError("identity_mask must be boolean [batch]")
        if (
            tuple(self.cross_modal_guard_passed.shape) != (batch,)
            or self.cross_modal_guard_passed.dtype != torch.bool
        ):
            raise ValueError(
                "cross_modal_guard_passed must be boolean [batch]"
            )
        if not torch.equal(self.decoded_xy_m[:, 1], self.radar_xy_m[:, 1]):
            raise ValueError("DenseLatticeRVFusion must copy radar y exactly")
        for name, value in (
            ("radar_probability", self.radar_probability),
            ("fused_probability", self.fused_probability),
            ("radar_xy_m", self.radar_xy_m),
            ("decoded_xy_m", self.decoded_xy_m),
            ("correction_x_m", self.correction_x_m),
            ("continuous_x_m", self.continuous_x_m),
            ("continuous_offset_m", self.continuous_offset_m),
            ("candidate_x_m", self.candidate_x_m),
            ("candidate_residual_m", self.candidate_residual_m),
            ("emission_logits", self.emission_logits),
            ("candidate_presence_logit", self.candidate_presence_logit),
            (
                "candidate_presence_probability",
                self.candidate_presence_probability,
            ),
            ("posterior", self.posterior),
            ("candidate_utility_logit", self.candidate_utility_logit),
            (
                "candidate_utility_probability",
                self.candidate_utility_probability,
            ),
            ("predicted_gain", self.predicted_gain),
            ("predicted_harm", self.predicted_harm),
            ("predicted_gain_logit", self.predicted_gain_logit),
            ("predicted_harm_logit", self.predicted_harm_logit),
            ("predicted_utility_logit", self.predicted_utility_logit),
            (
                "predicted_utility_probability",
                self.predicted_utility_probability,
            ),
            ("effective_acceptance", self.effective_acceptance),
            ("candidate_log_sigma_m", self.candidate_log_sigma_m),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")


def apply_causal_cross_modal_fallback(
    output: DenseLatticeFusionOutput,
    causal_guard_passed: Tensor,
) -> DenseLatticeFusionOutput:
    """Compose a frame-rate causal guard with the frozen segment policy.

    Segment admission remains the slow policy authority. The fast guard can
    only revoke an admitted visual correction; revocation restores both the
    decoded position and confidence map to the radar anchor exactly.
    """

    batch = output.radar_probability.shape[0]
    if tuple(causal_guard_passed.shape) != (batch,):
        raise ValueError("causal_guard_passed must be [batch]")
    if causal_guard_passed.dtype != torch.bool:
        raise ValueError("causal_guard_passed must be boolean")
    accepted = output.accepted & causal_guard_passed
    identity_mask = output.identity_mask | ~causal_guard_passed
    decoded_xy_m = torch.where(
        accepted[:, None],
        output.decoded_xy_m,
        output.radar_xy_m,
    )
    fused_probability = torch.where(
        accepted[:, None, None, None],
        output.fused_probability,
        output.radar_probability,
    )
    guarded = replace(
        output,
        fused_probability=fused_probability,
        decoded_xy_m=decoded_xy_m,
        accepted=accepted,
        identity_mask=identity_mask,
        cross_modal_guard_passed=(
            output.cross_modal_guard_passed & causal_guard_passed
        ),
    )
    guarded.validate()
    return guarded


def _decode_radar_peak(probability: Tensor) -> Tensor:
    flat_index = probability.flatten(1).argmax(dim=1)
    x_index = torch.div(flat_index, 234, rounding_mode="floor")
    y_index = flat_index.remainder(234)
    x_m = -10.0 + x_index.to(probability.dtype) * 0.1
    y_m = y_index.to(probability.dtype) * 0.1
    return torch.stack((x_m, y_m), dim=1)


def _canonical_bev_x_coordinates(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return -10.0 + torch.arange(200, device=device, dtype=dtype) * 0.1


def _sample_bev_feature(feature: Tensor, x_m: Tensor, y_m: Tensor) -> Tensor:
    x_normalized = 2.0 * (x_m + 10.0) / 19.9 - 1.0
    y_normalized = 2.0 * y_m / 23.3 - 1.0
    grid = torch.stack((y_normalized, x_normalized), dim=-1).unsqueeze(2)
    sampled = functional.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, :, :, 0].transpose(1, 2)


def _sample_polar_feature(feature: Tensor, x_m: Tensor, y_m: Tensor) -> Tensor:
    range_m = torch.sqrt(x_m.square() + y_m.square()).clamp_min(1e-6)
    range_index = range_m / 23.4 * 128.0
    range_normalized = 2.0 * range_index / 127.0 - 1.0
    sin_azimuth = (x_m / range_m).clamp(-1.0, 1.0)
    azimuth_index = (sin_azimuth + 1.0) * 64.0
    azimuth_normalized = 2.0 * azimuth_index / 127.0 - 1.0
    grid = torch.stack((azimuth_normalized, range_normalized), dim=-1).unsqueeze(2)
    sampled = functional.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, :, :, 0].transpose(1, 2)


class DenseLatticeRVFusion(nn.Module):
    """Radar-y-preserving RGB-assisted inference on a dense local x lattice."""

    def __init__(
        self,
        *,
        radius_m: float = 0.5,
        lattice_step_m: float = 0.025,
        token_dim: int = 96,
        acceptance_threshold: float = 0.5,
        acceptance_mode: str = "factorized",
        correction_source: str = "lattice",
        candidate_geometry: str = "dense_local",
        candidate_prior_mode: str = "none",
        candidate_utility_source: str = "token",
        candidate_prior_scale: float = 1.0,
        posterior_emission_mode: str = "shared",
        posterior_temperature: float = 1.0,
        continuous_mode: str = "free_offset",
        continuous_context_mode: str = "scalar",
        posterior_offset_scale: float = 0.25,
        visual_trust_policy: str = "soft",
        assist_policy: str = "radar_uncertainty",
    ) -> None:
        super().__init__()
        if radius_m <= 0.0 or lattice_step_m <= 0.0:
            raise ValueError("radius_m and lattice_step_m must be positive")
        if token_dim < 32 or token_dim % 2:
            raise ValueError("token_dim must be an even integer at least 32")
        if acceptance_mode not in {
            "factorized",
            "direct_utility",
            "candidate_utility",
        }:
            raise ValueError(
                "acceptance_mode must be factorized, direct_utility, "
                "or candidate_utility"
            )
        if correction_source not in {"lattice", "continuous"}:
            raise ValueError(
                "correction_source must be lattice or continuous"
            )
        if candidate_geometry not in {
            "dense_local",
            "measured_peaks",
            "visual_blend",
            "visual_blend_route",
            "visual_blend_route_dense",
        }:
            raise ValueError(
                "candidate_geometry must be dense_local, measured_peaks, "
                "visual_blend, visual_blend_route, or visual_blend_route_dense"
            )
        if candidate_prior_mode not in {
            "none",
            "route_blend",
            "bearing_likelihood",
        }:
            raise ValueError(
                "candidate_prior_mode must be none, route_blend, or "
                "bearing_likelihood"
            )
        if candidate_utility_source not in {"token", "scalar"}:
            raise ValueError(
                "candidate_utility_source must be token or scalar"
            )
        if candidate_prior_scale <= 0.0:
            raise ValueError("candidate_prior_scale must be positive")
        if posterior_emission_mode not in {
            "shared",
            "route_residual",
            "route_finetune",
        }:
            raise ValueError(
                "posterior_emission_mode must be shared, route_residual, "
                "or route_finetune"
            )
        if posterior_temperature <= 0.0:
            raise ValueError("posterior_temperature must be positive")
        if assist_policy not in {"radar_uncertainty", "vlm_segment"}:
            raise ValueError(
                "assist_policy must be radar_uncertainty or vlm_segment"
            )
        if continuous_mode not in {
            "free_offset",
            "posterior_mean",
            "route_blend_residual",
            "route_blend_offset_residual",
        }:
            raise ValueError(
                "continuous_mode must be free_offset, posterior_mean, "
                "route_blend_residual, or route_blend_offset_residual"
            )
        if continuous_context_mode not in {"scalar", "rich"}:
            raise ValueError(
                "continuous_context_mode must be scalar or rich"
            )
        if not 0.0 < posterior_offset_scale <= 1.0:
            raise ValueError(
                "posterior_offset_scale must be in (0,1]"
            )
        if (
            continuous_mode == "route_blend_residual"
            and (
                candidate_prior_mode != "route_blend"
                or candidate_geometry
                not in {
                    "visual_blend_route",
                    "visual_blend_route_dense",
                }
            )
        ):
            raise ValueError(
                "route_blend_residual requires route_blend prior and "
                "route-aware visual blend geometry"
            )
        if (
            candidate_prior_mode == "route_blend"
            and candidate_geometry
            not in {"visual_blend_route", "visual_blend_route_dense"}
        ):
            raise ValueError(
                "candidate_prior_mode route_blend requires "
                "a route-aware visual blend geometry"
            )
        self.radius_m = float(radius_m)
        self.lattice_step_m = float(lattice_step_m)
        self.acceptance_threshold = float(acceptance_threshold)
        self.acceptance_mode = acceptance_mode
        self.correction_source = correction_source
        self.candidate_geometry = candidate_geometry
        self.candidate_prior_mode = candidate_prior_mode
        self.candidate_utility_source = candidate_utility_source
        self.candidate_prior_scale = float(candidate_prior_scale)
        self.posterior_emission_mode = posterior_emission_mode
        self.posterior_temperature = float(posterior_temperature)
        self.continuous_mode = continuous_mode
        self.continuous_context_mode = continuous_context_mode
        self.posterior_offset_scale = float(
            posterior_offset_scale
        )
        self.assist_policy = assist_policy
        self.condition_encoder = ConditionEncoder(
            embedding_dim=32,
            trust_policy=visual_trust_policy,
        )
        self.bev_projection = nn.Sequential(
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
        )
        self.polar_projection = nn.Sequential(
            nn.Linear(48, 16),
            nn.SiLU(),
            nn.LayerNorm(16),
        )
        self.visual_projection = nn.Sequential(
            nn.LazyLinear(24),
            nn.SiLU(),
            nn.LayerNorm(24),
        )
        self.visual_calibrator = nn.Sequential(
            nn.LazyLinear(48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )
        nn.init.zeros_(self.visual_calibrator[-1].weight)
        nn.init.zeros_(self.visual_calibrator[-1].bias)
        self.scalar_projection = nn.Sequential(
            nn.LazyLinear(24),
            nn.SiLU(),
            nn.LayerNorm(24),
        )
        self.token_projection = nn.Sequential(
            nn.LazyLinear(token_dim),
            nn.SiLU(),
            nn.LayerNorm(token_dim),
        )
        self.candidate_attention = nn.MultiheadAttention(
            token_dim,
            num_heads=2,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.SiLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(token_dim)
        self.emission_head = nn.Linear(token_dim, 1)
        if candidate_prior_mode == "route_blend":
            nn.init.zeros_(self.emission_head.weight)
            nn.init.zeros_(self.emission_head.bias)
        self.posterior_route_emission_heads = nn.ModuleList(
            nn.Linear(token_dim, 1) for _ in ROUTES
        )
        for head in self.posterior_route_emission_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.candidate_presence_head = nn.Linear(token_dim * 2, 1)
        nn.init.zeros_(self.candidate_presence_head.weight)
        nn.init.constant_(self.candidate_presence_head.bias, -1.5)
        self.residual_head = nn.Linear(token_dim, 1)
        self.log_sigma_head = nn.Linear(token_dim, 1)
        self.acceptance_context = nn.Sequential(
            nn.LazyLinear(64),
            nn.SiLU(),
            nn.LayerNorm(64),
        )
        self.gain_head = nn.Linear(64, 1)
        self.harm_head = nn.Linear(64, 1)
        self.utility_head = nn.Linear(64, 1)
        nn.init.zeros_(self.utility_head.weight)
        nn.init.constant_(self.utility_head.bias, -1.5)
        self.selected_utility_context = nn.Sequential(
            nn.LazyLinear(64),
            nn.SiLU(),
            nn.LayerNorm(64),
        )
        self.selected_utility_head = nn.Linear(64, 1)
        nn.init.zeros_(self.selected_utility_head.weight)
        nn.init.zeros_(self.selected_utility_head.bias)
        self.route_utility_heads = nn.ModuleList(
            nn.Linear(64, 1) for _ in ROUTES
        )
        for head in self.route_utility_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.candidate_utility_head = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.candidate_utility_head[-1].weight)
        nn.init.constant_(self.candidate_utility_head[-1].bias, -1.5)
        self.candidate_scalar_utility_head = nn.Sequential(
            nn.Linear(9, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.candidate_scalar_utility_head[-1].weight)
        nn.init.constant_(
            self.candidate_scalar_utility_head[-1].bias,
            -1.5,
        )
        self.continuous_location_context = nn.Sequential(
            nn.LazyLinear(64),
            nn.SiLU(),
            nn.LayerNorm(64),
        )
        self.continuous_offset_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.continuous_offset_head[-1].weight)
        nn.init.zeros_(self.continuous_offset_head[-1].bias)
        residual_input = (
            nn.LazyLinear(24)
            if continuous_mode
            in {"route_blend_residual", "route_blend_offset_residual"}
            else nn.Linear(1, 24)
        )
        self.route_blend_residual_head = nn.Sequential(
            residual_input,
            nn.SiLU(),
            nn.LayerNorm(24),
            nn.Linear(24, 1),
        )
        nn.init.zeros_(self.route_blend_residual_head[-1].weight)
        nn.init.zeros_(self.route_blend_residual_head[-1].bias)

    @staticmethod
    def _validate_state(state: PersistentRadarState, batch: int) -> None:
        if tuple(state.assist_enabled.shape) != (batch,):
            raise ValueError("assist_enabled must be [batch]")
        if state.assist_enabled.dtype != torch.bool:
            raise ValueError("assist_enabled must be boolean")
        if tuple(state.uncertainty_strength.shape) != (batch,):
            raise ValueError("uncertainty_strength must be [batch]")

    def forward(
        self,
        radar: RadarEvidence,
        bearing: TemporalBearingEvidence,
        condition: ConditionPrior,
        uncertainty_state: PersistentRadarState,
        *,
        visual_anchor_geometry: Tensor | None = None,
        visual_trust_cap_override: Tensor | None = None,
        previous_correction_m: Tensor | None = None,
        acceptance_override: Tensor | None = None,
        cross_modal_guard_z_threshold: float | None = None,
        disable_rgb: bool = False,
        disable_temporal_visual: bool = False,
        disable_no_op: bool = False,
    ) -> DenseLatticeFusionOutput:
        radar.validate()
        batch = radar.logits.shape[0]
        bearing.validate(batch)
        condition.validate(batch)
        self._validate_state(uncertainty_state, batch)
        if (
            cross_modal_guard_z_threshold is not None
            and cross_modal_guard_z_threshold <= 0.0
        ):
            raise ValueError(
                "cross_modal_guard_z_threshold must be positive when set"
            )
        if visual_anchor_geometry is None:
            visual_anchor_geometry = radar.logits.new_zeros(batch, 6)
        if tuple(visual_anchor_geometry.shape) != (batch, 6):
            raise ValueError("visual_anchor_geometry must be [batch,6]")
        if not torch.isfinite(visual_anchor_geometry).all():
            raise ValueError("visual_anchor_geometry must be finite")
        visual_anchor_geometry = visual_anchor_geometry.float()
        probability = radar.logits.float().flatten(1).softmax(dim=1).view_as(
            radar.logits
        )
        radar_xy_m = resolved_radar_position(radar)
        radar_theta = torch.atan2(
            radar_xy_m[:, 0:1],
            radar_xy_m[:, 1:2].clamp_min(1e-6),
        )
        effective_theta = radar_theta if disable_rgb else bearing.theta.float()
        effective_sigma_theta = (
            torch.full_like(effective_theta, 0.10)
            if disable_rgb
            else bearing.sigma_theta.float()
        )
        effective_reliability = (
            torch.ones_like(bearing.reliability.float())
            if disable_rgb
            else bearing.reliability.float()
        )
        effective_bearing_valid = (
            torch.ones_like(bearing.valid)
            if disable_rgb
            else bearing.valid
        )
        effective_temporal_feature = (
            torch.zeros_like(bearing.temporal_feature.float())
            if disable_rgb or disable_temporal_visual
            else bearing.temporal_feature.float()
        )
        if disable_rgb:
            visual_anchor_geometry = torch.zeros_like(visual_anchor_geometry)
        condition_feature, trust_cap = self.condition_encoder(condition)
        if visual_trust_cap_override is not None:
            if tuple(visual_trust_cap_override.shape) != (batch, 1):
                raise ValueError("visual_trust_cap_override must be [batch,1]")
            trust_cap = visual_trust_cap_override.to(
                device=trust_cap.device,
                dtype=trust_cap.dtype,
            ).clamp(0.0, 1.0)
        visual_feature = self.visual_projection(effective_temporal_feature)
        calibration_context = torch.cat(
            (
                visual_feature,
                condition_feature,
                effective_theta,
                effective_sigma_theta,
                effective_reliability,
                visual_anchor_geometry,
                radar_xy_m.float() / 20.0,
                uncertainty_state.uncertainty_strength[:, None].float(),
            ),
            dim=1,
        )
        visual_x_m = radar_xy_m[:, 1:2] * torch.tan(effective_theta)
        calibration_bias_m = self.radius_m * torch.tanh(
            self.visual_calibrator(calibration_context)
        )
        calibrated_visual_x_m = visual_x_m + calibration_bias_m
        cosine = torch.cos(effective_theta).abs().clamp_min(0.2)
        visual_sigma_x_m = (
            radar_xy_m[:, 1:2]
            * effective_sigma_theta
            / cosine.square()
            + 0.05
        ).clamp(self.lattice_step_m, self.radius_m)
        radar_sigma_x_m = radar_lateral_sigma(
            probability,
            radar_xy_m,
            radius_m=self.radius_m,
        )
        innovation = cross_modal_innovation(
            visual_x_m=calibrated_visual_x_m,
            visual_sigma_x_m=visual_sigma_x_m,
            radar_x_m=radar_xy_m[:, 0:1],
            radar_sigma_x_m=radar_sigma_x_m,
        )
        if cross_modal_guard_z_threshold is None:
            cross_modal_guard_passed = torch.ones(
                batch,
                device=radar.logits.device,
                dtype=torch.bool,
            )
        else:
            cross_modal_guard_passed = (
                effective_bearing_valid
                & (effective_reliability[:, 0] > 0.0)
                & should_accept_visual_evidence(
                    innovation,
                    threshold=cross_modal_guard_z_threshold,
                )[:, 0]
            )
        if self.candidate_geometry == "visual_blend_route_dense":
            lattice = build_visual_blend_x_lattice(
                probability,
                radar_xy_m,
                calibrated_visual_x_m,
                visual_sigma_x_m,
                radius_m=self.radius_m,
                blend_fractions=ROUTE_BLEND_DENSE_FRACTIONS,
            )
        elif self.candidate_geometry == "visual_blend_route":
            lattice = build_visual_blend_x_lattice(
                probability,
                radar_xy_m,
                calibrated_visual_x_m,
                visual_sigma_x_m,
                radius_m=self.radius_m,
                blend_fractions=ROUTE_BLEND_FRACTIONS,
            )
        elif self.candidate_geometry == "visual_blend":
            lattice = build_visual_blend_x_lattice(
                probability,
                radar_xy_m,
                calibrated_visual_x_m,
                visual_sigma_x_m,
                radius_m=self.radius_m,
            )
        elif self.candidate_geometry == "measured_peaks":
            direct_x_m = radar_xy_m[:, 0:1] + 0.5 * (
                calibrated_visual_x_m - radar_xy_m[:, 0:1]
            ).clamp(-self.radius_m, self.radius_m)
            lattice = build_peak_x_lattice(
                radar_probability=probability,
                radar_xy_m=radar_xy_m,
                visual_x_m=calibrated_visual_x_m,
                direct_x_m=direct_x_m,
                visual_sigma_x_m=visual_sigma_x_m,
                radius_m=self.radius_m,
                top_k=4,
            )
        else:
            lattice = build_local_x_lattice(
                probability,
                radar_xy_m,
                calibrated_visual_x_m,
                visual_sigma_x_m,
                radius_m=self.radius_m,
                step_m=self.lattice_step_m,
            )
        candidate_y_m = radar_xy_m[:, 1:2].expand_as(lattice.x_m)
        bev_feature = self.bev_projection(
            _sample_bev_feature(
                radar.bev_feature.float(),
                lattice.x_m,
                candidate_y_m,
            )
        )
        polar_feature = self.polar_projection(
            _sample_polar_feature(
                radar.polar_feature.float(),
                lattice.x_m,
                candidate_y_m,
            )
        )
        visual_token = visual_feature[:, None].expand(
            -1,
            lattice.x_m.shape[1],
            -1,
        )
        condition_token = condition_feature[:, None].expand(
            -1,
            lattice.x_m.shape[1],
            -1,
        )
        no_op_support = lattice.radar_support[:, 0:1].clamp_min(1e-12)
        support_log_ratio = (
            lattice.radar_support.clamp_min(1e-12).log()
            - no_op_support.log()
        )
        normalized_visual_innovation = (
            lattice.x_m - calibrated_visual_x_m
        ) / visual_sigma_x_m
        scalars = torch.stack(
            (
                lattice.relative_x_m / self.radius_m,
                lattice.relative_x_m.abs() / self.radius_m,
                support_log_ratio.clamp(-20.0, 20.0),
                lattice.visual_log_likelihood.clamp(-20.0, 0.0),
                normalized_visual_innovation.clamp(-10.0, 10.0),
                effective_reliability.expand_as(lattice.x_m),
                trust_cap.expand_as(lattice.x_m),
                uncertainty_state.uncertainty_strength[:, None].expand_as(
                    lattice.x_m
                ),
                (lattice.source_id == 0).to(lattice.x_m.dtype),
            ),
            dim=-1,
        )
        scalar_feature = self.scalar_projection(scalars)
        token = self.token_projection(
            torch.cat(
                (
                    bev_feature,
                    polar_feature,
                    visual_token,
                    condition_token,
                    scalar_feature,
                ),
                dim=-1,
            )
        )
        attended, _ = self.candidate_attention(
            token,
            token,
            token,
            key_padding_mask=~lattice.valid_mask,
            need_weights=False,
        )
        token = self.attention_norm(token + attended)
        token = self.feed_forward_norm(token + self.feed_forward(token))
        emission_logits = self.emission_head(token).squeeze(-1)
        if self.posterior_emission_mode in {
            "route_residual",
            "route_finetune",
        }:
            route_lookup = {
                route: index for index, route in enumerate(ROUTES)
            }
            unknown_route = route_lookup["unknown"]
            route_index = torch.tensor(
                [
                    route_lookup.get(
                        str(route).strip().lower(),
                        unknown_route,
                    )
                    for route in condition.global_route
                ],
                device=emission_logits.device,
                dtype=torch.long,
            )
            route_emission_logits = torch.stack(
                [
                    head(token).squeeze(-1)
                    for head in self.posterior_route_emission_heads
                ],
                dim=1,
            )
            emission_logits = emission_logits + route_emission_logits[
                torch.arange(batch, device=emission_logits.device),
                route_index,
            ]
        if previous_correction_m is not None:
            if tuple(previous_correction_m.shape) != (batch, 1):
                raise ValueError("previous_correction_m must be [batch,1]")
            transition = functional.smooth_l1_loss(
                lattice.relative_x_m,
                previous_correction_m.expand_as(lattice.relative_x_m),
                reduction="none",
                beta=0.05,
            )
            emission_logits = emission_logits - 2.0 * transition
        if (
            self.candidate_prior_mode == "route_blend"
            and self.acceptance_mode != "candidate_utility"
        ):
            emission_logits = (
                emission_logits
                + _route_blend_candidate_prior(
                    condition.global_route,
                    blend_fractions=(
                        ROUTE_BLEND_DENSE_FRACTIONS
                        if self.candidate_geometry
                        == "visual_blend_route_dense"
                        else ROUTE_BLEND_FRACTIONS
                    ),
                    prior_scale=self.candidate_prior_scale,
                    device=emission_logits.device,
                    dtype=emission_logits.dtype,
                )
            )
        elif (
            self.candidate_prior_mode == "bearing_likelihood"
            and self.acceptance_mode != "candidate_utility"
        ):
            emission_logits = emission_logits + _bearing_likelihood_candidate_prior(
                lattice.visual_log_likelihood,
                reliability=effective_reliability,
                trust_cap=trust_cap,
                prior_scale=self.candidate_prior_scale,
            )
        emission_logits = emission_logits.masked_fill(
            ~lattice.valid_mask,
            -1e4,
        )
        non_noop_valid = lattice.valid_mask[:, 1:].to(token.dtype)
        candidate_summary = (
            token[:, 1:] * non_noop_valid.unsqueeze(-1)
        ).sum(dim=1) / non_noop_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        candidate_presence_logit = self.candidate_presence_head(
            torch.cat((token[:, 0], candidate_summary), dim=1)
        ).float()
        if disable_no_op:
            candidate_presence_logit = torch.full_like(
                candidate_presence_logit,
                20.0,
            )
        candidate_presence_probability = torch.sigmoid(
            candidate_presence_logit
        )
        conditional_non_noop = (
            emission_logits[:, 1:] / self.posterior_temperature
        ).softmax(dim=1)
        posterior = torch.cat(
            (
                1.0 - candidate_presence_probability,
                candidate_presence_probability * conditional_non_noop,
            ),
            dim=1,
        )
        candidate_residual_m = 0.5 * self.lattice_step_m * torch.tanh(
            self.residual_head(token).squeeze(-1)
        )
        candidate_residual_m = candidate_residual_m.clone()
        candidate_residual_m[:, 0] = 0.0
        candidate_x_m = (
            lattice.x_m + candidate_residual_m
        ).clamp(-10.0, 9.9)
        candidate_x_m = torch.max(
            torch.min(
                candidate_x_m,
                radar_xy_m[:, 0:1] + self.radius_m,
            ),
            radar_xy_m[:, 0:1] - self.radius_m,
        )
        candidate_x_m = candidate_x_m.clone()
        candidate_x_m[:, 0] = radar_xy_m[:, 0]
        candidate_log_sigma_m = self.log_sigma_head(token).squeeze(-1).clamp(
            -5.0,
            0.0,
        )
        if (
            self.candidate_utility_source == "scalar"
            or self.candidate_prior_mode == "route_blend"
        ):
            candidate_utility_logit = (
                self.candidate_scalar_utility_head(scalars)
                .squeeze(-1)
                .float()
            )
        else:
            candidate_utility_logit = (
                self.candidate_utility_head(token).squeeze(-1).float()
            )
        if self.candidate_prior_mode == "route_blend":
            candidate_utility_logit = (
                candidate_utility_logit
                + _route_blend_candidate_prior(
                    condition.global_route,
                    blend_fractions=(
                        ROUTE_BLEND_DENSE_FRACTIONS
                        if self.candidate_geometry
                        == "visual_blend_route_dense"
                        else ROUTE_BLEND_FRACTIONS
                    ),
                    prior_scale=self.candidate_prior_scale,
                    device=candidate_utility_logit.device,
                    dtype=candidate_utility_logit.dtype,
                )
            )
        elif self.candidate_prior_mode == "bearing_likelihood":
            candidate_utility_logit = (
                candidate_utility_logit
                + _bearing_likelihood_candidate_prior(
                    lattice.visual_log_likelihood,
                    reliability=effective_reliability,
                    trust_cap=trust_cap,
                    prior_scale=self.candidate_prior_scale,
                )
            )
        candidate_utility_logit = candidate_utility_logit.masked_fill(
            ~lattice.valid_mask,
            -20.0,
        )
        candidate_utility_logit = candidate_utility_logit.clone()
        candidate_utility_logit[:, 0] = -20.0
        candidate_utility_probability = torch.sigmoid(
            candidate_utility_logit
        )
        selected_non_noop = emission_logits[:, 1:].argmax(dim=1) + 1
        if self.acceptance_mode == "candidate_utility":
            selected_index = (
                candidate_utility_logit[:, 1:].argmax(dim=1) + 1
            )
        elif self.acceptance_mode == "direct_utility":
            selected_index = selected_non_noop
        else:
            selected_index = torch.where(
                candidate_presence_probability[:, 0] >= 0.5,
                selected_non_noop,
                torch.zeros_like(selected_non_noop),
            )
        row_index = torch.arange(batch, device=radar.logits.device)
        selected_x_m = candidate_x_m[row_index, selected_index]
        posterior_feature = (posterior.unsqueeze(-1) * token).sum(dim=1)
        continuous_location_feature = self.continuous_location_context(
            torch.cat(
                (
                    posterior_feature,
                    candidate_summary,
                    visual_feature,
                    condition_feature,
                    effective_reliability,
                    visual_anchor_geometry,
                    radar_xy_m.float() / 20.0,
                    uncertainty_state.uncertainty_strength[:, None].float(),
                ),
                dim=1,
            )
        )
        if self.continuous_mode == "posterior_mean":
            continuous_offset_m = (
                (posterior * candidate_x_m).sum(dim=1, keepdim=True)
                - radar_xy_m[:, 0:1]
            )
        elif self.continuous_mode in {
            "route_blend_residual",
            "route_blend_offset_residual",
        }:
            route_lookup = {
                route: index for index, route in enumerate(ROUTES)
            }
            unknown_route = route_lookup["unknown"]
            route_index = torch.tensor(
                [
                    route_lookup.get(
                        str(route).strip().lower(),
                        unknown_route,
                    )
                    for route in condition.global_route
                ],
                device=radar_xy_m.device,
                dtype=torch.long,
            )
            route_one_hot = functional.one_hot(
                route_index,
                num_classes=len(ROUTES),
            ).to(radar_xy_m.dtype)
            route_alpha = torch.tensor(
                [
                    ROUTE_BLEND_TARGET.get(
                        str(route).strip().lower(),
                        0.0,
                    )
                    for route in condition.global_route
                ],
                device=radar_xy_m.device,
                dtype=radar_xy_m.dtype,
            )[:, None]
            visual_delta_m = (
                calibrated_visual_x_m - radar_xy_m[:, 0:1]
            ).clamp(-self.radius_m, self.radius_m)
            route_blend_feature = torch.cat(
                (
                    route_one_hot.float(),
                    radar_xy_m.float()
                    / torch.tensor(
                        [10.0, 23.4],
                        device=radar_xy_m.device,
                        dtype=radar_xy_m.dtype,
                    ),
                    visual_delta_m.float() / self.radius_m,
                    visual_sigma_x_m.float() / self.radius_m,
                    bearing.theta.float(),
                    bearing.sigma_theta.float(),
                    effective_reliability.float(),
                    trust_cap.float(),
                    uncertainty_state.uncertainty_strength[:, None].float(),
                    support_log_ratio.float(),
                    lattice.visual_log_likelihood.clamp(-20.0, 0.0).float()
                    / 20.0,
                ),
                dim=1,
            )
            if self.continuous_context_mode == "rich":
                route_blend_feature = torch.cat(
                    (
                        route_blend_feature,
                        continuous_location_feature,
                    ),
                    dim=1,
                )
            residual_value = torch.tanh(
                self.route_blend_residual_head(route_blend_feature)
            )
            if self.continuous_mode == "route_blend_residual":
                blend_delta = self.posterior_offset_scale * residual_value
                blend_alpha = (route_alpha + blend_delta).clamp(0.0, 1.0)
                continuous_offset_m = blend_alpha * visual_delta_m
            else:
                base_offset_m = route_alpha * visual_delta_m
                residual_offset_m = (
                    self.radius_m
                    * self.posterior_offset_scale
                    * residual_value
                )
                continuous_offset_m = (
                    base_offset_m + residual_offset_m
                ).clamp(-self.radius_m, self.radius_m)
            continuous_offset_m = torch.where(
                route_alpha > 0.0,
                continuous_offset_m,
                torch.zeros_like(continuous_offset_m),
            )
        else:
            continuous_offset_m = self.radius_m * torch.tanh(
                self.continuous_offset_head(continuous_location_feature)
            )
        continuous_x_m = (
            radar_xy_m[:, 0:1] + continuous_offset_m
        ).clamp(-10.0, 9.9)
        if self.correction_source == "continuous":
            correction_x_m = continuous_x_m
            correction_distance = (
                candidate_x_m - correction_x_m
            ).abs().masked_fill(~lattice.valid_mask, float("inf"))
            correction_index = correction_distance.argmin(dim=1)
            correction_non_noop = (
                correction_x_m[:, 0] - radar_xy_m[:, 0]
            ).abs() >= 0.5 * self.lattice_step_m
        else:
            correction_x_m = selected_x_m[:, None]
            correction_index = selected_index
            correction_non_noop = selected_index != 0
        top_values = posterior.topk(k=2, dim=1).values
        acceptance_feature = self.acceptance_context(
            torch.cat(
                (
                    posterior_feature,
                    top_values,
                    condition_feature,
                    visual_feature,
                    effective_reliability,
                visual_anchor_geometry,
                trust_cap,
                uncertainty_state.uncertainty_strength[:, None].float(),
                ),
                dim=1,
            )
        )
        predicted_gain_logit = self.gain_head(acceptance_feature).float()
        predicted_harm_logit = self.harm_head(acceptance_feature).float()
        selected_token = token[row_index, correction_index]
        selected_posterior = posterior[
            row_index,
            correction_index,
        ][:, None]
        non_noop_probability = emission_logits[:, 1:].softmax(dim=1)
        non_noop_top = non_noop_probability.topk(k=2, dim=1).values
        non_noop_margin = non_noop_top[:, 0:1] - non_noop_top[:, 1:2]
        non_noop_entropy = -(
            non_noop_probability
            * non_noop_probability.clamp_min(1e-12).log()
        ).sum(dim=1, keepdim=True)
        non_noop_entropy = non_noop_entropy / torch.log(
            torch.tensor(
                float(non_noop_probability.shape[1]),
                device=non_noop_entropy.device,
                dtype=non_noop_entropy.dtype,
            )
        )
        selected_sigma_m = candidate_log_sigma_m[
            row_index,
            correction_index,
        ][:, None].exp()
        selected_diagnostics = torch.cat(
            (
                (correction_x_m - radar_xy_m[:, 0:1])
                / self.radius_m,
                (correction_x_m - calibrated_visual_x_m)
                / self.radius_m,
                (calibrated_visual_x_m - radar_xy_m[:, 0:1])
                / self.radius_m,
                selected_posterior,
                non_noop_margin,
                non_noop_entropy,
                selected_sigma_m / self.radius_m,
                candidate_presence_probability,
                effective_reliability,
                trust_cap,
                uncertainty_state.uncertainty_strength[:, None].float(),
            ),
            dim=1,
        )
        selected_utility_feature = self.selected_utility_context(
            torch.cat(
                (
                    selected_token,
                    acceptance_feature,
                    selected_diagnostics,
                ),
                dim=1,
            )
        )
        route_utility_logits = torch.cat(
            [
                head(selected_utility_feature)
                for head in self.route_utility_heads
            ],
            dim=1,
        )
        route_lookup = {route: index for index, route in enumerate(ROUTES)}
        unknown_route = route_lookup["unknown"]
        route_index = torch.tensor(
            [
                route_lookup.get(
                    str(route).strip().lower(),
                    unknown_route,
                )
                for route in condition.global_route
            ],
            device=route_utility_logits.device,
            dtype=torch.long,
        )
        route_utility_logit = route_utility_logits.gather(
            1,
            route_index[:, None],
        )
        predicted_utility_logit = (
            self.utility_head(acceptance_feature)
            + self.selected_utility_head(selected_utility_feature)
            + route_utility_logit
        ).float()
        predicted_gain = torch.sigmoid(predicted_gain_logit)
        predicted_harm = torch.sigmoid(predicted_harm_logit)
        predicted_utility_probability = torch.sigmoid(
            predicted_utility_logit
        )
        if self.acceptance_mode == "candidate_utility":
            selected_candidate_utility = (
                candidate_utility_probability[
                    row_index,
                    correction_index,
                ][:, None]
            )
            effective_acceptance = (
                selected_candidate_utility * trust_cap
            ).clamp(0.0, 1.0)
        elif self.acceptance_mode == "direct_utility":
            effective_acceptance = (
                predicted_utility_probability * trust_cap
            ).clamp(0.0, 1.0)
        else:
            effective_acceptance = (
                predicted_gain
                * (1.0 - predicted_harm)
                * effective_reliability
                * trust_cap
                * uncertainty_state.uncertainty_strength[:, None].float()
            ).clamp(0.0, 1.0)
        if acceptance_override is not None:
            if tuple(acceptance_override.shape) != (batch, 1):
                raise ValueError("acceptance_override must be [batch,1]")
            effective_acceptance = acceptance_override.to(
                device=effective_acceptance.device,
                dtype=effective_acceptance.dtype,
            )
        route_allowed = torch.tensor(
            [
                str(route).strip().lower() != "corner_nlos"
                for route in condition.global_route
            ],
            device=radar.logits.device,
            dtype=torch.bool,
        )
        if self.assist_policy == "vlm_segment":
            # The slow policy is the only authority that enables visual assistance.
            # Once it enables a segment, every frame uses the continuous posterior;
            # frame-local bearing validity, no-op selection, and acceptance scores are
            # diagnostics rather than hidden frame-wise fusion switches.
            identity_mask = (
                ~route_allowed
                | (trust_cap[:, 0] <= 0.0)
                | ~cross_modal_guard_passed
            )
            accepted = ~identity_mask
        else:
            identity_mask = (
                ~effective_bearing_valid
                | ~route_allowed
                | (trust_cap[:, 0] <= 0.0)
            )
            identity_mask = (
                identity_mask
                | ~uncertainty_state.assist_enabled
                | (uncertainty_state.uncertainty_strength <= 0.0)
                | ~cross_modal_guard_passed
            )
            accepted = (
                ~identity_mask
                & correction_non_noop
                & (effective_acceptance[:, 0] >= self.acceptance_threshold)
            )
        decoded_x_m = torch.where(
            accepted,
            correction_x_m[:, 0],
            radar_xy_m[:, 0],
        )
        decoded_xy_m = torch.stack((decoded_x_m, radar_xy_m[:, 1]), dim=1)

        x_coordinate = _canonical_bev_x_coordinates(
            device=radar.logits.device,
            dtype=probability.dtype,
        )
        x_density = torch.exp(
            -0.5
            * (
                (
                    x_coordinate[None, None, :]
                    - candidate_x_m[:, :, None]
                )
                / 0.05
            ).square()
        )
        if self.correction_source == "continuous":
            x_density = torch.exp(
                -0.5
                * (
                    (
                        x_coordinate[None, :]
                        - correction_x_m
                    )
                    / 0.05
                ).square()
            )
        elif self.acceptance_mode == "candidate_utility":
            x_density = x_density[row_index, selected_index]
        else:
            x_density = (posterior[:, :, None] * x_density).sum(dim=1)
        x_density = x_density / x_density.sum(dim=1, keepdim=True).clamp_min(
            1e-12
        )
        y_marginal = probability.sum(dim=2, keepdim=True)
        candidate_probability = (
            x_density[:, None, :, None] * y_marginal
        )
        fused_probability = torch.where(
            accepted[:, None, None, None],
            candidate_probability,
            probability,
        )
        output = DenseLatticeFusionOutput(
            radar_probability=probability,
            fused_probability=fused_probability,
            radar_xy_m=radar_xy_m,
            decoded_xy_m=decoded_xy_m,
            correction_x_m=correction_x_m,
            correction_index=correction_index,
            correction_non_noop=correction_non_noop,
            correction_source=self.correction_source,
            continuous_x_m=continuous_x_m,
            continuous_offset_m=continuous_offset_m,
            candidate_x_m=candidate_x_m,
            candidate_residual_m=candidate_residual_m,
            candidate_valid_mask=lattice.valid_mask,
            emission_logits=emission_logits,
            candidate_presence_logit=candidate_presence_logit,
            candidate_presence_probability=candidate_presence_probability,
            candidate_utility_logit=candidate_utility_logit,
            candidate_utility_probability=candidate_utility_probability,
            posterior=posterior,
            selected_index=selected_index,
            predicted_gain_logit=predicted_gain_logit,
            predicted_harm_logit=predicted_harm_logit,
            predicted_utility_logit=predicted_utility_logit,
            predicted_gain=predicted_gain,
            predicted_harm=predicted_harm,
            predicted_utility_probability=predicted_utility_probability,
            effective_acceptance=effective_acceptance,
            accepted=accepted,
            identity_mask=identity_mask,
            visual_x_m=visual_x_m,
            calibrated_visual_x_m=calibrated_visual_x_m,
            candidate_log_sigma_m=candidate_log_sigma_m,
            radar_sigma_x_m=radar_sigma_x_m,
            cross_modal_innovation=innovation,
            cross_modal_guard_passed=cross_modal_guard_passed,
        )
        output.validate()
        return output



