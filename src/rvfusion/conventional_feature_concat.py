from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from .contracts import RadarEvidence, VisualEvidence


@dataclass(frozen=True)
class ConventionalFusionOutput:
    fused_logits: Tensor
    radar_logits: Tensor
    delta_logits: Tensor
    visual_summary: Tensor


class ConventionalBEVFeatureConcat(nn.Module):
    """Matched conventional radar/RGB BEV feature-concatenation baseline.

    This is deliberately a plain comparator: radar and visual features are
    projected to the shared BEV grid, concatenated, and decoded by a small
    convolutional head. It has no VLM route prior, candidate lattice,
    forward-coordinate-preserving constraint, temporal memory, or identity gate.
    """

    def __init__(self, visual_dim: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden_dim),
        )
        self.radar_projection = nn.Sequential(
            nn.Conv2d(128, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim * 2 + 1, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        # Start from the radar prediction, but let the ordinary decoder learn
        # an unconstrained correction in both BEV axes.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _pool_visual(visual: VisualEvidence) -> Tensor:
        visual.validate()
        weights = (
            visual.valid_mask.to(visual.raw_tokens.dtype)
            * visual.raw_scores.clamp(0.0, 1.0)
            * visual.temporal_support.clamp(0.0, 1.0)
        )
        tokens = 0.5 * (visual.raw_tokens + visual.temporal_tokens)
        weights = weights.unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=(1, 2))
        return pooled / weights.sum(dim=(1, 2)).clamp_min(1e-6)

    def forward(self, radar: RadarEvidence, visual: VisualEvidence) -> ConventionalFusionOutput:
        radar.validate()
        visual.validate()
        if radar.logits.shape[0] != visual.boxes.shape[0]:
            raise ValueError("radar and visual batch sizes must match")
        radar_bev = self.radar_projection(radar.bev_feature)
        visual_summary = self.visual_projection(self._pool_visual(visual))
        visual_bev = visual_summary[:, :, None, None].expand_as(radar_bev)
        radar_logit_low = functional.interpolate(
            radar.logits, size=radar_bev.shape[-2:], mode="bilinear", align_corners=False
        )
        features = torch.cat((radar_bev, visual_bev, radar_logit_low), dim=1)
        delta_low = self.decoder(features)
        delta = functional.interpolate(
            delta_low, size=radar.logits.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = radar.logits + delta
        return ConventionalFusionOutput(
            fused_logits=fused,
            radar_logits=radar.logits,
            delta_logits=delta,
            visual_summary=visual_summary,
        )



