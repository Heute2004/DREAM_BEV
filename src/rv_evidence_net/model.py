from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torchvision.ops import roi_align

from .contracts import SpatialFrameOutput, SpatialTemporalOutput
from .temporal import CausalMambaEvidenceRecovery


def _channels(value: int, width: float, minimum: int = 16) -> int:
    return max(minimum, int(round(value * width / 8.0) * 8))


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, groups: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 2) -> None:
        super().__init__()
        hidden = max(in_channels * expansion, out_channels)
        self.expand = ConvNormAct(in_channels, hidden, 1)
        self.depthwise = ConvNormAct(hidden, hidden, 3, stride=stride, groups=hidden)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.project(self.depthwise(self.expand(inputs))) + self.skip(inputs))


class LightweightSpatialEncoder(nn.Module):
    """Simple task-specific convolutional encoder, intentionally not a claimed novelty."""

    def __init__(self, width: float) -> None:
        super().__init__()
        stem_channels = _channels(48, width)
        stage2_channels = _channels(96, width)
        stage3_channels = _channels(160, width)
        stage4_channels = _channels(256, width)
        stage5_channels = _channels(320, width)
        self.output_channels = (stage3_channels, stage4_channels, stage5_channels)

        self.stem = ConvNormAct(3, stem_channels, 3, stride=2)
        self.stage2 = nn.Sequential(
            DepthwiseResidualBlock(stem_channels, stage2_channels, stride=2),
            DepthwiseResidualBlock(stage2_channels, stage2_channels),
        )
        self.stage3 = nn.Sequential(
            DepthwiseResidualBlock(stage2_channels, stage3_channels, stride=2),
            DepthwiseResidualBlock(stage3_channels, stage3_channels),
        )
        self.stage4 = nn.Sequential(
            DepthwiseResidualBlock(stage3_channels, stage4_channels, stride=2),
            DepthwiseResidualBlock(stage4_channels, stage4_channels),
        )
        self.stage5 = nn.Sequential(
            DepthwiseResidualBlock(stage4_channels, stage5_channels, stride=2),
            DepthwiseResidualBlock(stage5_channels, stage5_channels),
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self.stem(images)
        features = self.stage2(features)
        p3 = self.stage3(features)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        return p3, p4, p5


class LiteFeaturePyramid(nn.Module):
    def __init__(self, input_channels: tuple[int, int, int], output_channels: int) -> None:
        super().__init__()
        self.laterals = nn.ModuleList(nn.Conv2d(channels, output_channels, 1) for channels in input_channels)
        self.smooth = nn.ModuleList(
            nn.Sequential(
                ConvNormAct(output_channels, output_channels, 3, groups=output_channels),
                ConvNormAct(output_channels, output_channels, 1),
            )
            for _ in range(3)
        )

    def forward(self, pyramid: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        p3, p4, p5 = (lateral(feature) for lateral, feature in zip(self.laterals, pyramid, strict=True))
        p4 = p4 + functional.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p3 = p3 + functional.interpolate(p4, size=p3.shape[-2:], mode="nearest")
        return tuple(smooth(feature) for smooth, feature in zip(self.smooth, (p3, p4, p5), strict=True))


class AnchorFreePersonHead(nn.Module):
    """One-class center-based prediction head used at strides 8, 16, and 32."""

    def __init__(self, channels: int, quality_head: bool = False) -> None:
        super().__init__()
        self.tower = nn.Sequential(
            ConvNormAct(channels, channels, 3),
            ConvNormAct(channels, channels, 3),
        )
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.size = nn.Conv2d(channels, 2, 1)
        self.offset = nn.Conv2d(channels, 2, 1)
        self.quality = nn.Conv2d(channels, 1, 1) if quality_head else None
        nn.init.constant_(self.objectness.bias, -4.6)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        shared = self.tower(features)
        output = {
            "objectness": self.objectness(shared),
            "size": self.size(shared),
            "offset": self.offset(shared),
        }
        if self.quality is not None:
            output["quality"] = self.quality(shared)
        return output


class ProposalConditionedEvidenceAdapter(nn.Module):
    """Extract a memory-compatible token from P3/P4/P5 at a supplied proposal."""

    def __init__(self, feature_channels: int, token_dim: int, output_size: int = 3) -> None:
        super().__init__()
        flattened = feature_channels * output_size * output_size
        self.output_size = output_size
        self.per_level = nn.ModuleList(
            nn.Sequential(nn.Flatten(), nn.Linear(flattened, token_dim), nn.SiLU()) for _ in range(3)
        )
        self.fuse = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(self, pyramid: tuple[Tensor, Tensor, Tensor], boxes: Tensor, valid_mask: Tensor) -> Tensor:
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("boxes must have shape [batch, proposals, 4]")
        if valid_mask.shape != boxes.shape[:2]:
            raise ValueError("valid_mask must match boxes")
        batch_size, proposal_count, _ = boxes.shape
        token_dim = self.fuse[-1].normalized_shape[0]
        result = boxes.new_zeros(batch_size * proposal_count, token_dim)
        valid_indices = valid_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return result.reshape(batch_size, proposal_count, token_dim)

        flat_boxes = boxes.reshape(-1, 4)[valid_indices]
        batch_indices = torch.div(valid_indices, proposal_count, rounding_mode="floor").to(dtype=boxes.dtype)
        rois = torch.cat((batch_indices.unsqueeze(1), flat_boxes), dim=1)
        level_tokens = []
        for features, projection, scale in zip(pyramid, self.per_level, (1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0), strict=True):
            pooled = roi_align(
                features,
                rois,
                output_size=self.output_size,
                spatial_scale=scale,
                sampling_ratio=2,
                aligned=True,
            )
            level_tokens.append(projection(pooled))
        result[valid_indices] = self.fuse(torch.cat(level_tokens, dim=-1))
        return result.reshape(batch_size, proposal_count, token_dim)


class RVEvidenceNetSTMamba(nn.Module):
    """Single-RGB spatial detector plus causal Mamba-style evidence recovery."""

    strides = (8, 16, 32)

    def __init__(
        self,
        width: float = 1.25,
        fpn_channels: int | None = None,
        token_dim: int = 128,
        temporal_model_dim: int = 128,
        temporal_state_dim: int = 16,
        temporal_layers: int = 2,
        temporal_backend: str = "selective_scan",
        top_k: int = 3,
        quality_head: bool = False,
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.top_k = top_k
        self.token_dim = token_dim
        self.quality_head = quality_head
        self.encoder = LightweightSpatialEncoder(width)
        self.fpn_channels = fpn_channels or _channels(192, width, minimum=48)
        self.neck = LiteFeaturePyramid(self.encoder.output_channels, self.fpn_channels)
        self.heads = nn.ModuleList(AnchorFreePersonHead(self.fpn_channels, quality_head=quality_head) for _ in self.strides)
        self.evidence_adapter = ProposalConditionedEvidenceAdapter(self.fpn_channels, token_dim)
        self.recovery = CausalMambaEvidenceRecovery(
            token_dim=token_dim,
            model_dim=temporal_model_dim,
            state_dim=temporal_state_dim,
            layers=temporal_layers,
            temporal_backend=temporal_backend,
        )

    @staticmethod
    def _geometry(boxes: Tensor, image_height: int, image_width: int) -> Tensor:
        anchor_x = (boxes[..., 0] + boxes[..., 2]) * 0.5 / image_width
        anchor_y = boxes[..., 1] / image_height
        width = (boxes[..., 2] - boxes[..., 0]).clamp_min(0.0) / image_width
        height = (boxes[..., 3] - boxes[..., 1]).clamp_min(0.0) / image_height
        base = torch.stack((anchor_x, anchor_y, width, height), dim=-1)
        displacement = torch.zeros_like(base[..., :2])
        displacement[:, 1:] = base[:, 1:, ..., :2] - base[:, :-1, ..., :2]
        return torch.cat((base, displacement), dim=-1)

    def _decode_level(self, maps: dict[str, Tensor], stride: int) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, _, height, width = maps["objectness"].shape
        objectness = maps["objectness"].sigmoid().reshape(batch_size, -1)
        quality = maps["quality"].sigmoid().reshape(batch_size, -1) if "quality" in maps else torch.ones_like(objectness)
        size = maps["size"].permute(0, 2, 3, 1).reshape(batch_size, -1, 2).clamp(-4.0, 4.0).exp() * stride
        offset = maps["offset"].sigmoid().permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=objectness.device, dtype=objectness.dtype),
            torch.arange(width, device=objectness.device, dtype=objectness.dtype),
            indexing="ij",
        )
        center_x = (grid_x.reshape(1, -1) + offset[..., 0]) * stride
        center_y = (grid_y.reshape(1, -1) + offset[..., 1]) * stride
        boxes = torch.stack(
            (
                center_x - size[..., 0] * 0.5,
                center_y - size[..., 1] * 0.5,
                center_x + size[..., 0] * 0.5,
                center_y + size[..., 1] * 0.5,
            ),
            dim=-1,
        )
        return boxes, objectness, quality

    def decode_topk(
        self,
        maps: tuple[dict[str, Tensor], ...],
        score_threshold: float = 0.05,
        score_mode: str = "objectness",
    ) -> tuple[Tensor, Tensor, Tensor]:
        boxes, scores, _, valid = self.decode_topk_with_quality(maps, score_threshold, score_mode)
        return boxes, scores, valid

    def decode_topk_with_quality(
        self,
        maps: tuple[dict[str, Tensor], ...],
        score_threshold: float = 0.05,
        score_mode: str = "objectness",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if score_mode not in {"objectness", "quality_gated"}:
            raise ValueError("score_mode must be 'objectness' or 'quality_gated'")
        decoded = [self._decode_level(level_maps, stride) for level_maps, stride in zip(maps, self.strides, strict=True)]
        boxes = torch.cat([item[0] for item in decoded], dim=1)
        objectness = torch.cat([item[1] for item in decoded], dim=1)
        quality = torch.cat([item[2] for item in decoded], dim=1)
        scores = objectness if score_mode == "objectness" else objectness * quality
        top_k = min(self.top_k, scores.shape[1])
        values, indices = torch.topk(scores, k=top_k, dim=1)
        chosen_boxes = boxes.gather(1, indices.unsqueeze(-1).expand(-1, -1, 4))
        chosen_quality = quality.gather(1, indices)
        valid = values >= score_threshold
        return chosen_boxes, values, chosen_quality, valid

    def forward_frame(
        self,
        images: Tensor,
        proposals: Tensor | None = None,
        proposal_valid_mask: Tensor | None = None,
        proposal_scores: Tensor | None = None,
        extract_evidence: bool = True,
        detection_score_mode: str = "objectness",
    ) -> SpatialFrameOutput:
        pyramid = self.neck(self.encoder(images))
        maps = tuple(head(features) for head, features in zip(self.heads, pyramid, strict=True))
        if proposals is None:
            proposals, scores, proposal_quality, valid = self.decode_topk_with_quality(maps, score_mode=detection_score_mode)
        else:
            if proposals.shape[0] != images.shape[0] or proposals.ndim != 3 or proposals.shape[-1] != 4:
                raise ValueError("proposals must have shape [batch, proposals, 4]")
            valid = proposal_valid_mask if proposal_valid_mask is not None else torch.ones(
                proposals.shape[:2], device=proposals.device, dtype=torch.bool
            )
            scores = proposal_scores if proposal_scores is not None else valid.to(dtype=images.dtype)
            proposal_quality = valid.to(dtype=images.dtype)
        raw_tokens = (
            self.evidence_adapter(pyramid, proposals, valid)
            if extract_evidence
            else images.new_zeros(proposals.shape[0], proposals.shape[1], self.token_dim)
        )
        return SpatialFrameOutput(pyramid, maps, proposals, scores, proposal_quality, raw_tokens, valid)

    def forward_clip(
        self,
        images: Tensor,
        proposals: Tensor | None = None,
        proposal_valid_mask: Tensor | None = None,
        proposal_scores: Tensor | None = None,
    ) -> SpatialTemporalOutput:
        if images.ndim != 5:
            raise ValueError("images must have shape [batch, time, channels, height, width]")
        batch_size, time_steps, channels, height, width = images.shape
        if channels != 3:
            raise ValueError("RV-EvidenceNet consumes RGB only")
        flattened_images = images.reshape(batch_size * time_steps, channels, height, width)
        flattened_proposals = None if proposals is None else proposals.reshape(batch_size * time_steps, proposals.shape[2], 4)
        flattened_valid = None if proposal_valid_mask is None else proposal_valid_mask.reshape(
            batch_size * time_steps, proposal_valid_mask.shape[2]
        )
        flattened_scores = None if proposal_scores is None else proposal_scores.reshape(
            batch_size * time_steps, proposal_scores.shape[2]
        )
        spatial = self.forward_frame(flattened_images, flattened_proposals, flattened_valid, flattened_scores)
        proposal_count = spatial.boxes.shape[1]
        boxes = spatial.boxes.reshape(batch_size, time_steps, proposal_count, 4)
        raw_scores = spatial.raw_scores.reshape(batch_size, time_steps, proposal_count)
        proposal_quality = spatial.proposal_quality.reshape(batch_size, time_steps, proposal_count)
        raw_tokens = spatial.raw_tokens.reshape(batch_size, time_steps, proposal_count, -1)
        valid = spatial.valid_mask.reshape(batch_size, time_steps, proposal_count)
        geometry = self._geometry(boxes, height, width)
        temporal = self.recovery(raw_tokens, geometry, raw_scores, valid)
        # Restore per-level dictionaries after temporal batch flattening.
        per_frame_maps = tuple(
            {
                "objectness": spatial.detection_maps[level]["objectness"].reshape(batch_size, time_steps, *spatial.detection_maps[level]["objectness"].shape[1:]),
                "size": spatial.detection_maps[level]["size"].reshape(batch_size, time_steps, *spatial.detection_maps[level]["size"].shape[1:]),
                "offset": spatial.detection_maps[level]["offset"].reshape(batch_size, time_steps, *spatial.detection_maps[level]["offset"].shape[1:]),
            }
            for level in range(len(self.strides))
        )
        return SpatialTemporalOutput(
            detection_maps=per_frame_maps,
            boxes=boxes,
            raw_scores=raw_scores,
            proposal_quality=proposal_quality,
            calibrated_scores=temporal.calibrated_scores,
            raw_tokens=raw_tokens,
            temporal_tokens=temporal.refined_tokens,
            temporal_support=temporal.temporal_support,
            valid_mask=temporal.valid_mask,
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def spatial_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.encoder.parameters()
        yield from self.neck.parameters()
        yield from self.heads.parameters()
        yield from self.evidence_adapter.parameters()



