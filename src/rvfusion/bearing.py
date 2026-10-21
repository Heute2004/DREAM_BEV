from __future__ import annotations

import torch
from torch import Tensor, nn

from .contracts import TemporalBearingEvidence, VisualEvidence


def top_edge_anchor_geometry(
    boxes: Tensor,
    *,
    image_width: float = 640.0,
    image_height: float = 640.0,
    boundary_tolerance_px: float = 1.0,
) -> Tensor:
    """Return normalized top-edge anchor and truncation cues for each box."""
    if boxes.ndim < 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must end with [x1,y1,x2,y2]")
    if image_width <= 0.0 or image_height <= 0.0:
        raise ValueError("image dimensions must be positive")
    anchor_u = 0.5 * (boxes[..., 0] + boxes[..., 2]) / float(image_width)
    anchor_v = boxes[..., 1] / float(image_height)
    width = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0) / float(image_width)
    height = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0) / float(image_height)
    top_clipped = (boxes[..., 1] <= float(boundary_tolerance_px)).to(boxes.dtype)
    side_clipped = (
        (boxes[..., 0] <= float(boundary_tolerance_px))
        | (boxes[..., 2] >= float(image_width) - float(boundary_tolerance_px))
    ).to(boxes.dtype)
    return torch.stack(
        (anchor_u, anchor_v, width, height, top_clipped, side_clipped),
        dim=-1,
    )


def aggregate_top_edge_anchor_geometry(
    visual: VisualEvidence,
    *,
    image_width: float = 640.0,
    image_height: float = 640.0,
) -> Tensor:
    """Hold the latest highest-score valid top-edge anchor as [batch,6]."""
    visual.validate()
    geometry = top_edge_anchor_geometry(
        visual.boxes,
        image_width=image_width,
        image_height=image_height,
    )
    frame_valid = visual.valid_mask.any(dim=2)
    masked_scores = visual.raw_scores.masked_fill(~visual.valid_mask, -1e4)
    selected_index = masked_scores.argmax(dim=2)
    frame_geometry = geometry.gather(
        2,
        selected_index[..., None, None].expand(-1, -1, 1, geometry.shape[-1]),
    ).squeeze(2)
    held = torch.zeros_like(frame_geometry[:, 0])
    ever_valid = torch.zeros(
        visual.boxes.shape[0],
        device=visual.boxes.device,
        dtype=torch.bool,
    )
    for time_index in range(visual.boxes.shape[1]):
        current_valid = frame_valid[:, time_index]
        held = torch.where(
            current_valid[:, None],
            frame_geometry[:, time_index],
            held,
        )
        ever_valid = ever_valid | current_valid
    return held * ever_valid.to(held.dtype).unsqueeze(1)


class CausalVisualBearingEncoder(nn.Module):
    def __init__(
        self,
        token_dim: int = 128,
        hidden_dim: int = 64,
        reference_image_size: float = 640.0,
        minimum_sigma_rad: float = 0.01,
        base_sigma_span_rad: float = 0.04,
        maximum_delta_rad: float = 0.20,
        initial_half_fov_rad: float = 0.75,
        max_missing_age: int = 3,
        missing_age_tau: float = 2.0,
        age_sigma_growth: float = 0.02,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reference_image_size = float(reference_image_size)
        self.minimum_sigma_rad = float(minimum_sigma_rad)
        self.base_sigma_span_rad = float(base_sigma_span_rad)
        self.maximum_delta_rad = float(maximum_delta_rad)
        self.initial_half_fov_rad = float(initial_half_fov_rad)
        self.max_missing_age = int(max_missing_age)
        self.missing_age_tau = float(missing_age_tau)
        self.age_sigma_growth = float(age_sigma_growth)
        geometry_dim = 6
        input_dim = 2 * token_dim + geometry_dim
        self.proposal_score = nn.Linear(input_dim, 1)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.theta_delta_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)
        self.reliability_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.theta_delta_head.weight)
        nn.init.zeros_(self.theta_delta_head.bias)

    def _aggregate_proposals(
        self, visual: VisualEvidence
    ) -> tuple[Tensor, Tensor, Tensor]:
        boxes = visual.boxes / self.reference_image_size
        # The canonical image anchor is the top-edge midpoint. Its horizontal
        # coordinate defines bearing; vertical position is not interpreted as range.
        anchor_u = 0.5 * (boxes[..., 0] + boxes[..., 2])
        width = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0)
        height = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0)
        geometry = torch.stack(
            (
                anchor_u,
                width,
                height,
                visual.raw_scores,
                visual.temporal_support,
                visual.valid_mask.to(boxes.dtype),
            ),
            dim=-1,
        )
        proposal_input = torch.cat(
            (visual.raw_tokens, visual.temporal_tokens, geometry),
            dim=-1,
        )
        score = self.proposal_score(proposal_input).squeeze(-1)
        score = score.masked_fill(~visual.valid_mask, -1e4)
        frame_valid = visual.valid_mask.any(dim=2)
        weights = score.softmax(dim=2) * visual.valid_mask.to(score.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        aggregate = (proposal_input * weights.unsqueeze(-1)).sum(dim=2)
        anchor_u_norm = (anchor_u * weights).sum(dim=2)
        return aggregate, anchor_u_norm, frame_valid

    @staticmethod
    def _causal_hold(
        value: Tensor, frame_valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, time_steps = value.shape
        held = torch.zeros_like(value)
        missing_age = torch.full_like(value, float(time_steps + 1))
        ever_valid = torch.zeros(batch, device=value.device, dtype=torch.bool)
        last_value = torch.zeros(batch, device=value.device, dtype=value.dtype)
        age = torch.zeros(batch, device=value.device, dtype=value.dtype)
        for time_index in range(time_steps):
            current_valid = frame_valid[:, time_index]
            last_value = torch.where(current_valid, value[:, time_index], last_value)
            age = torch.where(current_valid, torch.zeros_like(age), age + 1.0)
            ever_valid = ever_valid | current_valid
            held[:, time_index] = last_value
            missing_age[:, time_index] = torch.where(
                ever_valid, age, torch.full_like(age, float(time_steps + 1))
            )
        return held, missing_age, ever_valid

    def forward(
        self,
        visual: VisualEvidence,
        *,
        output_time_index: int | None = None,
    ) -> TemporalBearingEvidence:
        visual.validate()
        batch, time_steps = visual.boxes.shape[:2]
        if output_time_index is None:
            output_time_index = time_steps - 1
        if not 0 <= output_time_index < time_steps:
            raise ValueError("output_time_index is outside the visual clip")

        aggregate, anchor_u_norm, frame_valid = self._aggregate_proposals(visual)
        base_theta = self.initial_half_fov_rad * (
            2.0 * anchor_u_norm - 1.0
        )
        held_theta, missing_age, _ = self._causal_hold(base_theta, frame_valid)
        missing_flag = (~frame_valid).to(aggregate.dtype).unsqueeze(-1)
        temporal_input = self.input_projection(torch.cat((aggregate, missing_flag), dim=-1))
        temporal_feature, _ = self.temporal(temporal_input)
        selected = temporal_feature[:, output_time_index]
        selected_age = missing_age[:, output_time_index : output_time_index + 1]
        selected_theta = held_theta[:, output_time_index : output_time_index + 1]
        delta = self.maximum_delta_rad * torch.tanh(self.theta_delta_head(selected))
        theta = selected_theta + delta
        sigma = (
            self.minimum_sigma_rad
            + self.base_sigma_span_rad * torch.sigmoid(self.log_sigma_head(selected))
            + selected_age * self.age_sigma_growth
        )
        reliability = torch.sigmoid(self.reliability_head(selected))
        reliability = reliability * torch.exp(-selected_age / self.missing_age_tau)
        valid = (
            frame_valid[:, : output_time_index + 1].any(dim=1)
            & (selected_age[:, 0] <= self.max_missing_age)
        )
        reliability = reliability * valid.to(reliability.dtype).unsqueeze(1)
        output = TemporalBearingEvidence(
            theta=theta,
            sigma_theta=sigma,
            temporal_feature=selected,
            reliability=reliability,
            valid=valid,
        )
        output.validate(batch)
        return output



