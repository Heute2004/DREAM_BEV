from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import RadarEvidence, VisualEvidence


@dataclass(frozen=True)
class ExternalFusionOutput:
    fused_logits: Tensor
    radar_logits: Tensor
    delta_logits: Tensor
    metadata: dict[str, object]


def visual_evidence_to_bev_ray_map(
    visual: VisualEvidence,
    *,
    spatial_size: tuple[int, int] = (50, 59),
    image_width: float = 480.0,
    ray_sigma_cells: float = 1.5,
) -> Tensor:
    """Project frozen RGB evidence to a range-agnostic BEV ray map.

    The student evidence supplies image-space bearing and temporal tokens, but
    no depth. Each proposal therefore becomes a soft vertical ray in the BEV
    grid. This is the common input adapter for the public fusion controls; it
    deliberately does not use depth or ground-truth coordinates.
    """
    visual.validate()
    height, width = spatial_size
    boxes = visual.boxes[:, -1]
    scores = visual.raw_scores[:, -1].clamp(0.0, 1.0)
    support = visual.temporal_support[:, -1].clamp(0.0, 1.0)
    valid = visual.valid_mask[:, -1].to(scores.dtype)
    weights = scores * support * valid
    tokens = 0.5 * (visual.raw_tokens[:, -1] + visual.temporal_tokens[:, -1])
    centers = ((boxes[..., 0] + boxes[..., 2]) * 0.5 / image_width)
    centers = centers.clamp(0.0, 1.0) * float(width - 1)
    columns = torch.arange(width, device=visual.boxes.device, dtype=centers.dtype)
    basis = torch.exp(-0.5 * ((columns[None, None] - centers[..., None]) / ray_sigma_cells) ** 2)
    weighted_basis = basis * weights[..., None]
    numerator = torch.einsum("bkc,bkw->bcw", tokens, weighted_basis)
    denominator = weighted_basis.sum(dim=1).clamp_min(1e-6)
    ray = numerator / denominator[:, None]
    return ray.unsqueeze(2).expand(-1, -1, height, -1)


class _OutputMixin:
    @staticmethod
    def _output(radar: RadarEvidence, delta: Tensor, name: str) -> ExternalFusionOutput:
        delta = F.interpolate(delta, size=radar.logits.shape[-2:], mode="bilinear", align_corners=False)
        return ExternalFusionOutput(
            fused_logits=radar.logits + delta,
            radar_logits=radar.logits,
            delta_logits=delta,
            metadata={
                "baseline_name": name,
                "range_preserving_constraint": False,
                "depth_input": False,
                "future_frame_input": False,
            },
        )


class RCMFusionTaskAdapter(nn.Module, _OutputMixin):
    """Task adapter of the published RCM-Fusion feature/gating idea.

    It retains the radar-guided BEV update and adaptive multimodal gate, but
    replaces the original automotive point-cloud/3D-box stack with the locked
    2D human-localization contract used by this project.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.radar_projection = nn.Sequential(
            nn.Conv2d(128, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.visual_projection = nn.Sequential(
            nn.Conv2d(128, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.query = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.key = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.value = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, radar: RadarEvidence, visual: VisualEvidence) -> ExternalFusionOutput:
        radar.validate()
        visual_map = visual_evidence_to_bev_ray_map(visual, spatial_size=radar.bev_feature.shape[-2:])
        radar_map = self.radar_projection(radar.bev_feature)
        visual_map = self.visual_projection(visual_map)
        query = self.query(radar_map)
        key = self.key(visual_map)
        value = self.value(visual_map)
        affinity = (query * key).sum(dim=1, keepdim=True) / (query.shape[1] ** 0.5)
        affinity = affinity.sigmoid()
        gate = self.gate(torch.cat((radar_map, visual_map, (radar_map - visual_map).abs()), dim=1))
        fused = radar_map + gate * affinity * value
        radar_low = F.interpolate(radar.logits, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        return self._output(radar, self.decoder(torch.cat((fused, radar_low), dim=1)), "RCM-Fusion-task-adapted")


class RCBEVDetTaskAdapter(nn.Module, _OutputMixin):
    """Task adapter of RCBEVDet's dual-stream BEV cross-attention idea."""

    def __init__(self, hidden_dim: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.radar_stream = nn.Sequential(
            nn.Conv2d(128, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.visual_stream = nn.Sequential(
            nn.Conv2d(128, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def forward(self, radar: RadarEvidence, visual: VisualEvidence) -> ExternalFusionOutput:
        radar.validate()
        visual_map = visual_evidence_to_bev_ray_map(visual, spatial_size=radar.bev_feature.shape[-2:])
        radar_map = F.avg_pool2d(self.radar_stream(radar.bev_feature), 2)
        visual_map = F.avg_pool2d(self.visual_stream(visual_map), 2)
        batch, channels, height, width = radar_map.shape
        query = radar_map.flatten(2).transpose(1, 2)
        memory = visual_map.flatten(2).transpose(1, 2)
        attended, _ = self.cross_attention(query, memory, memory, need_weights=False)
        attended = self.norm(attended + query).transpose(1, 2).reshape(batch, channels, height, width)
        gate = self.spatial_gate(torch.cat((radar_map, attended), dim=1))
        fused = radar_map + gate * attended
        radar_low = F.interpolate(radar.logits, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        return self._output(radar, self.decoder(torch.cat((fused, radar_low), dim=1)), "RCBEVDet-task-adapted")




