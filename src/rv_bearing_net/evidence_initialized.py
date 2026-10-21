from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .model import RVBearingNet, VerticalAttentionPool


class EvidencePyramidEncoder(nn.Module):
    """Reuse the trained S1 person-evidence pyramid as a horizontal encoder."""

    def __init__(self, width: float = 1.25) -> None:
        super().__init__()
        try:
            from rv_evidence_net.model import (
                LiteFeaturePyramid,
                LightweightSpatialEncoder,
                _channels,
            )
        except ImportError as error:
            raise ImportError(
                "EvidenceInitializedRVBearingNet requires sibling "
                "the local rv_evidence_net package on PYTHONPATH"
            ) from error
        self.encoder = LightweightSpatialEncoder(width)
        self.output_channels = _channels(192, width, minimum=48)
        self.neck = LiteFeaturePyramid(
            self.encoder.output_channels,
            self.output_channels,
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.neck(self.encoder(images))[0]


class EvidenceInitializedRVBearingNet(RVBearingNet):
    def __init__(
        self,
        *,
        evidence_width: float = 1.25,
        angle_bins: int = 181,
        half_fov_rad: float = 0.80,
        stripe_dim: int = 128,
        temporal_dim: int = 128,
    ) -> None:
        super().__init__(
            angle_bins=angle_bins,
            half_fov_rad=half_fov_rad,
            stripe_dim=stripe_dim,
            temporal_dim=temporal_dim,
        )
        self.evidence_width = float(evidence_width)
        self._evidence_frozen = False
        self.encoder = EvidencePyramidEncoder(self.evidence_width)
        channels = self.encoder.output_channels
        self.vertical_pool = VerticalAttentionPool(channels)
        self.stripe_projection = nn.Sequential(
            nn.Conv1d(channels, stripe_dim, kernel_size=1, bias=False),
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

    def load_evidence_state(
        self,
        payload: Mapping[str, object],
    ) -> None:
        raw_state = payload.get("model", payload)
        if not isinstance(raw_state, Mapping):
            raise TypeError("evidence checkpoint must contain a model state")
        state = {
            str(name): value
            for name, value in raw_state.items()
            if isinstance(value, Tensor)
        }
        encoder_state = {
            name.removeprefix("encoder."): value
            for name, value in state.items()
            if name.startswith("encoder.")
        }
        neck_state = {
            name.removeprefix("neck."): value
            for name, value in state.items()
            if name.startswith("neck.")
        }
        if not encoder_state or not neck_state:
            raise ValueError(
                "evidence checkpoint lacks encoder or neck parameters"
            )
        self.encoder.encoder.load_state_dict(encoder_state, strict=True)
        self.encoder.neck.load_state_dict(neck_state, strict=True)

    def freeze_evidence_backbone(self, frozen: bool) -> None:
        self._evidence_frozen = bool(frozen)
        self.encoder.requires_grad_(not frozen)
        if frozen:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._evidence_frozen:
            self.encoder.eval()
        return self

    def backbone_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.encoder.parameters())



