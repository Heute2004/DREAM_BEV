from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional

from .contracts import RVBearingOutput


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.output_norm = nn.BatchNorm2d(channels)

    def forward(self, values: Tensor) -> Tensor:
        residual = values
        values = self.depthwise(values)
        values = self.norm(values)
        values = self.pointwise(values)
        return functional.silu(self.output_norm(values) + residual)


class RGBFrameEncoder(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        if len(channels) != 4 or any(value < 8 for value in channels):
            raise ValueError("encoder_channels must contain four values >= 8")
        stages: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(
                    3,
                    channels[0],
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm2d(channels[0]),
                nn.SiLU(),
                DepthwiseResidualBlock(channels[0]),
            )
        ]
        for input_channels, output_channels in zip(
            channels[:-1],
            channels[1:],
            strict=True,
        ):
            stages.append(
                nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels),
                    nn.SiLU(),
                    DepthwiseResidualBlock(output_channels),
                    DepthwiseResidualBlock(output_channels),
                )
            )
        self.stages = nn.Sequential(*stages)

    def forward(self, images: Tensor) -> Tensor:
        return self.stages(images)


class VerticalAttentionPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, features: Tensor) -> Tensor:
        weights = self.attention(features).softmax(dim=2)
        return (features * weights).sum(dim=2)


class RVBearingNet(nn.Module):
    """Causal RGB model that predicts horizontal bearing evidence only."""

    def __init__(
        self,
        *,
        angle_bins: int = 181,
        half_fov_rad: float = 0.80,
        stripe_dim: int = 96,
        temporal_dim: int = 96,
        encoder_channels: Sequence[int] = (32, 48, 80, 128),
        minimum_sigma_rad: float = math.radians(0.25),
        max_missing_age: int = 3,
        missing_age_tau: float = 2.0,
    ) -> None:
        super().__init__()
        if angle_bins < 3 or angle_bins % 2 == 0:
            raise ValueError("angle_bins must be an odd integer >= 3")
        if half_fov_rad <= 0.0:
            raise ValueError("half_fov_rad must be positive")
        if stripe_dim < 8 or temporal_dim < 8:
            raise ValueError("stripe_dim and temporal_dim must be >= 8")
        self.angle_bins = int(angle_bins)
        self.half_fov_rad = float(half_fov_rad)
        self.temporal_dim = int(temporal_dim)
        self.minimum_sigma_rad = float(minimum_sigma_rad)
        self.max_missing_age = int(max_missing_age)
        self.missing_age_tau = float(missing_age_tau)

        self.encoder = RGBFrameEncoder(encoder_channels)
        self.vertical_pool = VerticalAttentionPool(encoder_channels[-1])
        self.stripe_projection = nn.Sequential(
            nn.Conv1d(encoder_channels[-1], stripe_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(stripe_dim),
            nn.SiLU(),
            nn.Conv1d(
                stripe_dim,
                stripe_dim,
                kernel_size=3,
                padding=1,
                groups=stripe_dim,
                bias=False,
            ),
            nn.BatchNorm1d(stripe_dim),
            nn.SiLU(),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(stripe_dim, temporal_dim),
            nn.SiLU(),
            nn.LayerNorm(temporal_dim),
        )
        self.temporal_cell = nn.GRUCell(temporal_dim, temporal_dim)
        self.film = nn.Linear(temporal_dim, 2 * stripe_dim)
        self.bearing_head = nn.Sequential(
            nn.Conv1d(stripe_dim, stripe_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(stripe_dim, 1, kernel_size=1),
        )
        self.person_u_head = nn.Sequential(
            nn.Conv1d(stripe_dim, stripe_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(stripe_dim, 1, kernel_size=1),
        )
        self.uncertainty_floor_head = nn.Linear(temporal_dim, 1)
        self.reliability_head = nn.Sequential(
            nn.Linear(temporal_dim + 2, temporal_dim),
            nn.SiLU(),
            nn.Linear(temporal_dim, 1),
        )
        self.register_buffer(
            "angle_centers",
            torch.linspace(
                -self.half_fov_rad,
                self.half_fov_rad,
                self.angle_bins,
            ),
            persistent=True,
        )

    def _encode_frames(self, images: Tensor) -> Tensor:
        batch, time_steps, channels, height, width = images.shape
        flattened = images.reshape(batch * time_steps, channels, height, width)
        stripe = self.vertical_pool(self.encoder(flattened))
        stripe = self.stripe_projection(stripe)
        return stripe.reshape(
            batch,
            time_steps,
            stripe.shape[1],
            stripe.shape[2],
        )

    def _causal_state_sequence(
        self,
        images: Tensor,
        frame_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, time_steps = images.shape[:2]
        stripes = self._encode_frames(images)
        tokens = self.token_projection(stripes.mean(dim=-1))
        hidden = images.new_zeros(batch, self.temporal_dim)
        held_stripe = images.new_zeros(
            batch,
            stripes.shape[2],
            stripes.shape[3],
        )
        ever_valid = torch.zeros(batch, device=images.device, dtype=torch.bool)
        age = images.new_full((batch,), float(time_steps + 1))
        hidden_history: list[Tensor] = []
        stripe_history: list[Tensor] = []
        age_history: list[Tensor] = []
        valid_history: list[Tensor] = []
        for time_index in range(time_steps):
            current_valid = frame_valid[:, time_index]
            proposal = self.temporal_cell(tokens[:, time_index], hidden)
            hidden = torch.where(current_valid[:, None], proposal, hidden)
            held_stripe = torch.where(
                current_valid[:, None, None],
                stripes[:, time_index],
                held_stripe,
            )
            age = torch.where(current_valid, torch.zeros_like(age), age + 1.0)
            ever_valid = ever_valid | current_valid
            hidden_history.append(hidden)
            stripe_history.append(held_stripe)
            age_history.append(age)
            valid_history.append(
                ever_valid & (age <= self.max_missing_age)
            )
        return (
            torch.stack(hidden_history, dim=1),
            torch.stack(stripe_history, dim=1),
            torch.stack(age_history, dim=1),
            torch.stack(valid_history, dim=1),
        )

    def _decode_state(
        self,
        selected_hidden: Tensor,
        selected_stripe: Tensor,
        selected_age: Tensor,
        selected_valid: Tensor,
    ) -> RVBearingOutput:
        batch = selected_hidden.shape[0]
        scale, shift = self.film(selected_hidden).chunk(2, dim=1)
        modulated = selected_stripe * (
            1.0 + 0.25 * torch.tanh(scale).unsqueeze(-1)
        ) + 0.25 * torch.tanh(shift).unsqueeze(-1)
        dense = functional.interpolate(
            modulated,
            size=self.angle_bins,
            mode="linear",
            align_corners=True,
        )
        bearing_logits = self.bearing_head(dense).squeeze(1)
        person_u_logits = self.person_u_head(dense).squeeze(1)
        probability = bearing_logits.softmax(dim=1)
        centers = self.angle_centers.to(
            device=probability.device,
            dtype=probability.dtype,
        )
        theta = (probability * centers[None]).sum(dim=1, keepdim=True)
        variance = (
            probability
            * (centers[None] - theta).square()
        ).sum(dim=1, keepdim=True)
        learned_floor = functional.softplus(
            self.uncertainty_floor_head(selected_hidden)
        )
        sigma = variance.clamp_min(1e-8).sqrt()
        sigma_theta = (
            sigma.square()
            + (self.minimum_sigma_rad + 0.02 * learned_floor).square()
        ).sqrt()
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True) / math.log(self.angle_bins)
        peak = probability.amax(dim=1, keepdim=True)
        reliability_input = torch.cat(
            (selected_hidden, peak, entropy),
            dim=1,
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_input))
        reliability = reliability * torch.exp(
            -selected_age[:, None] / self.missing_age_tau
        )
        reliability = reliability * selected_valid[:, None].to(
            reliability.dtype
        )
        output = RVBearingOutput(
            bearing_logits=bearing_logits,
            theta=theta,
            sigma_theta=sigma_theta,
            reliability=reliability,
            temporal_feature=selected_hidden,
            person_u_logits=person_u_logits,
            valid=selected_valid,
        )
        output.validate(
            batch_size=batch,
            angle_bins=self.angle_bins,
            temporal_dim=self.temporal_dim,
        )
        return output

    def _validated_frame_mask(
        self,
        images: Tensor,
        frame_valid: Tensor | None,
    ) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(
                "images must have shape [batch,time,3,height,width]"
            )
        batch, time_steps = images.shape[:2]
        if time_steps < 1:
            raise ValueError("the causal clip must contain at least one frame")
        if frame_valid is None:
            return torch.ones(
                batch,
                time_steps,
                dtype=torch.bool,
                device=images.device,
            )
        if tuple(frame_valid.shape) != (batch, time_steps):
            raise ValueError("frame_valid must have shape [batch,time]")
        return frame_valid

    def forward(
        self,
        images: Tensor,
        *,
        frame_valid: Tensor | None = None,
        output_time_index: int | None = None,
    ) -> RVBearingOutput:
        frame_valid = self._validated_frame_mask(images, frame_valid)
        time_steps = images.shape[1]
        if output_time_index is None:
            output_time_index = time_steps - 1
        if not 0 <= output_time_index < time_steps:
            raise ValueError("output_time_index is outside the supplied clip")
        hidden, stripe, age, valid = self._causal_state_sequence(
            images,
            frame_valid,
        )
        return self._decode_state(
            hidden[:, output_time_index],
            stripe[:, output_time_index],
            age[:, output_time_index],
            valid[:, output_time_index],
        )

    def forward_pair(
        self,
        images: Tensor,
        *,
        frame_valid: Tensor | None = None,
    ) -> tuple[RVBearingOutput, RVBearingOutput]:
        frame_valid = self._validated_frame_mask(images, frame_valid)
        if images.shape[1] < 2:
            raise ValueError("forward_pair requires at least two frames")
        hidden, stripe, age, valid = self._causal_state_sequence(
            images,
            frame_valid,
        )
        previous = self._decode_state(
            hidden[:, -2],
            stripe[:, -2],
            age[:, -2],
            valid[:, -2],
        )
        current = self._decode_state(
            hidden[:, -1],
            stripe[:, -1],
            age[:, -1],
            valid[:, -1],
        )
        return previous, current

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )



