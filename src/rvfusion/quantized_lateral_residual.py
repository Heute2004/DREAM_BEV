from __future__ import annotations

"""Shared-context quantized lateral residual control.

This control intentionally has no candidate-local feature crops or candidate
attention. It predicts a fixed lateral-bin class and a bounded within-bin
offset from shared anchor context only.
"""

import torch
from torch import Tensor, nn

from .conditions import ConditionEncoder
from .contracts import ConditionPrior, RadarEvidence, TemporalBearingEvidence
from .cross_modal_consistency import (
    cross_modal_innovation,
    radar_lateral_sigma,
    should_accept_visual_evidence,
)
from .lattice_fusion import (
    DenseLatticeFusionOutput,
    _sample_bev_feature,
    _sample_polar_feature,
)
from .metric_contract import resolved_radar_position
from .radar_health import PersistentRadarState


def fixed_lateral_offsets(
    *,
    radius_m: float,
    step_m: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    steps = int(round(float(radius_m) / float(step_m)))
    if radius_m <= 0.0 or step_m <= 0.0 or steps < 1:
        raise ValueError("radius_m and step_m must define positive fixed bins")
    if abs(steps * float(step_m) - float(radius_m)) > 1e-7:
        raise ValueError("radius_m must be an integer multiple of step_m")
    negative = torch.arange(-steps, 0, device=device, dtype=dtype)
    positive = torch.arange(1, steps + 1, device=device, dtype=dtype)
    return torch.cat((torch.zeros(1, device=device, dtype=dtype), negative, positive)) * float(step_m)


class QuantizedLateralResidual(nn.Module):
    """Predict a discrete x offset and one bounded residual per fixed bin."""

    is_quantized_lateral_residual = True

    def __init__(
        self,
        *,
        radius_m: float = 0.5,
        step_m: float = 0.025,
        trust_policy: str = "normal_or_rain_nonunreliable",
    ) -> None:
        super().__init__()
        if radius_m <= 0.0 or step_m <= 0.0:
            raise ValueError("radius_m and step_m must be positive")
        steps = int(round(float(radius_m) / float(step_m)))
        if steps < 1 or abs(steps * float(step_m) - float(radius_m)) > 1e-7:
            raise ValueError("radius_m must be an integer multiple of step_m")
        self.radius_m = float(radius_m)
        self.step_m = float(step_m)
        self.candidate_count = 1 + 2 * steps
        self.condition_encoder = ConditionEncoder(
            embedding_dim=32,
            trust_policy=trust_policy,
        )
        self.bev_projection = nn.Sequential(
            nn.Linear(128, 32), nn.SiLU(), nn.LayerNorm(32)
        )
        self.polar_projection = nn.Sequential(
            nn.Linear(48, 16), nn.SiLU(), nn.LayerNorm(16)
        )
        self.visual_projection = nn.Sequential(
            nn.LazyLinear(24), nn.SiLU(), nn.LayerNorm(24)
        )
        self.scalar_projection = nn.Sequential(
            nn.Linear(7, 16), nn.SiLU(), nn.LayerNorm(16)
        )
        self.context = nn.Sequential(
            nn.LazyLinear(64), nn.SiLU(), nn.LayerNorm(64)
        )
        self.class_head = nn.Linear(64, self.candidate_count)
        self.within_bin_head = nn.Linear(64, self.candidate_count)
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)
        nn.init.zeros_(self.within_bin_head.weight)
        nn.init.zeros_(self.within_bin_head.bias)

    @staticmethod
    def _route_allowed(condition: ConditionPrior, device: torch.device) -> Tensor:
        return torch.tensor(
            [str(route).strip().lower() != "corner_nlos" for route in condition.global_route],
            device=device,
            dtype=torch.bool,
        )

    def forward(
        self,
        radar: RadarEvidence,
        bearing: TemporalBearingEvidence,
        condition: ConditionPrior,
        uncertainty_state: PersistentRadarState,
        *,
        visual_anchor_geometry: Tensor | None = None,
        visual_trust_cap_override: Tensor | None = None,
        acceptance_override: Tensor | None = None,
        cross_modal_guard_z_threshold: float | None = None,
        **_: object,
    ) -> DenseLatticeFusionOutput:
        radar.validate()
        batch = radar.logits.shape[0]
        bearing.validate(batch)
        condition.validate(batch)
        if visual_anchor_geometry is None:
            visual_anchor_geometry = radar.logits.new_zeros(batch, 6)
        if tuple(visual_anchor_geometry.shape) != (batch, 6):
            raise ValueError("visual_anchor_geometry must be [batch,6]")
        probability = radar.logits.float().flatten(1).softmax(dim=1).view_as(radar.logits)
        radar_xy_m = resolved_radar_position(radar)
        condition_feature, trust_cap = self.condition_encoder(condition)
        if visual_trust_cap_override is not None:
            if tuple(visual_trust_cap_override.shape) != (batch, 1):
                raise ValueError("visual_trust_cap_override must be [batch,1]")
            trust_cap = visual_trust_cap_override.to(trust_cap).clamp(0.0, 1.0)
        bev_at_anchor = _sample_bev_feature(
            radar.bev_feature.float(), radar_xy_m[:, 0:1], radar_xy_m[:, 1:2]
        )[:, 0]
        polar_at_anchor = _sample_polar_feature(
            radar.polar_feature.float(), radar_xy_m[:, 0:1], radar_xy_m[:, 1:2]
        )[:, 0]
        visual_feature = self.visual_projection(bearing.temporal_feature.float())
        visual_x_m = radar_xy_m[:, 1:2] * torch.tan(bearing.theta.float())
        visual_sigma_x_m = (
            radar_xy_m[:, 1:2]
            * bearing.sigma_theta.float()
            / torch.cos(bearing.theta.float()).abs().clamp_min(0.2).square()
            + 0.05
        ).clamp(self.step_m, self.radius_m)
        radar_sigma_x_m = radar_lateral_sigma(
            probability,
            radar_xy_m,
            radius_m=self.radius_m,
        )
        innovation = cross_modal_innovation(
            visual_x_m=visual_x_m,
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
                bearing.valid
                & (bearing.reliability[:, 0] > 0.0)
                & should_accept_visual_evidence(
                    innovation,
                    threshold=cross_modal_guard_z_threshold,
                )[:, 0]
            )
        scalar_feature = self.scalar_projection(
            torch.cat(
                (
                    radar_xy_m.float() / radar_xy_m.new_tensor([10.0, 23.4]),
                    bearing.theta.float(),
                    bearing.sigma_theta.float(),
                    bearing.reliability.float(),
                    trust_cap.float(),
                    uncertainty_state.uncertainty_strength[:, None].float(),
                ),
                dim=1,
            )
        )
        context = self.context(
            torch.cat(
                (
                    self.bev_projection(bev_at_anchor),
                    self.polar_projection(polar_at_anchor),
                    visual_feature,
                    condition_feature,
                    scalar_feature,
                    visual_anchor_geometry.float(),
                    (visual_x_m - radar_xy_m[:, 0:1]) / self.radius_m,
                ),
                dim=1,
            )
        )
        emission_logits = self.class_head(context)
        within_bin_residual_m = 0.5 * self.step_m * torch.tanh(
            self.within_bin_head(context)
        )
        offsets = fixed_lateral_offsets(
            radius_m=self.radius_m,
            step_m=self.step_m,
            device=radar_xy_m.device,
            dtype=radar_xy_m.dtype,
        )
        candidate_x_m = radar_xy_m[:, 0:1] + offsets[None]
        selected_index = emission_logits.argmax(dim=1)
        row = torch.arange(batch, device=radar_xy_m.device)
        selected_x_m = candidate_x_m[row, selected_index][:, None] + within_bin_residual_m[row, selected_index][:, None]
        selected_x_m = selected_x_m.clamp(
            radar_xy_m[:, 0:1] - self.radius_m,
            radar_xy_m[:, 0:1] + self.radius_m,
        ).clamp(-10.0, 9.9)
        route_allowed = self._route_allowed(condition, radar.logits.device)
        identity_mask = (
            ~route_allowed
            | (trust_cap[:, 0] <= 0.0)
            | ~cross_modal_guard_passed
        )
        if acceptance_override is not None:
            if tuple(acceptance_override.shape) != (batch, 1):
                raise ValueError("acceptance_override must be [batch,1]")
            identity_mask = identity_mask | (acceptance_override[:, 0] <= 0.0)
        accepted = ~identity_mask
        decoded_x_m = torch.where(accepted, selected_x_m[:, 0], radar_xy_m[:, 0])
        decoded_xy_m = torch.stack((decoded_x_m, radar_xy_m[:, 1]), dim=1)
        posterior = emission_logits.softmax(dim=1)
        correction_non_noop = (selected_index != 0) & accepted
        correction_index = torch.where(
            correction_non_noop, selected_index, torch.zeros_like(selected_index)
        )
        zero = torch.zeros(batch, 1, device=radar.logits.device, dtype=radar.logits.dtype)
        output = DenseLatticeFusionOutput(
            radar_probability=probability,
            fused_probability=probability,
            radar_xy_m=radar_xy_m,
            decoded_xy_m=decoded_xy_m,
            correction_x_m=selected_x_m,
            correction_index=correction_index,
            correction_non_noop=correction_non_noop,
            correction_source="lattice",
            continuous_x_m=selected_x_m,
            continuous_offset_m=selected_x_m - radar_xy_m[:, 0:1],
            candidate_x_m=candidate_x_m,
            candidate_residual_m=within_bin_residual_m,
            candidate_valid_mask=torch.ones_like(candidate_x_m, dtype=torch.bool),
            emission_logits=emission_logits,
            candidate_presence_logit=zero,
            candidate_presence_probability=1.0 - posterior[:, 0:1],
            candidate_utility_logit=torch.zeros_like(emission_logits),
            candidate_utility_probability=torch.zeros_like(emission_logits),
            posterior=posterior,
            selected_index=selected_index,
            predicted_gain_logit=zero,
            predicted_harm_logit=zero,
            predicted_utility_logit=zero,
            predicted_gain=torch.sigmoid(zero),
            predicted_harm=torch.sigmoid(zero),
            predicted_utility_probability=torch.sigmoid(zero),
            effective_acceptance=accepted[:, None].to(dtype=radar.logits.dtype),
            accepted=accepted,
            identity_mask=identity_mask,
            visual_x_m=visual_x_m,
            calibrated_visual_x_m=visual_x_m,
            candidate_log_sigma_m=torch.zeros_like(emission_logits),
            radar_sigma_x_m=radar_sigma_x_m,
            cross_modal_innovation=innovation,
            cross_modal_guard_passed=cross_modal_guard_passed,
        )
        output.validate()
        return output
