from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalRecoveryOutput:
    """Proposal-level causal recovery outputs for a complete RGB clip."""

    refined_tokens: torch.Tensor
    temporal_support: torch.Tensor
    calibrated_scores: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class SpatialFrameOutput:
    """Spatial detections and proposal evidence from one RGB frame batch."""

    feature_pyramid: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    detection_maps: tuple[dict[str, torch.Tensor], ...]
    boxes: torch.Tensor
    raw_scores: torch.Tensor
    proposal_quality: torch.Tensor
    raw_tokens: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class SpatialTemporalOutput:
    """Public M2 output contract for a causal RGB clip."""

    detection_maps: tuple[dict[str, torch.Tensor], ...]
    boxes: torch.Tensor
    raw_scores: torch.Tensor
    proposal_quality: torch.Tensor
    calibrated_scores: torch.Tensor
    raw_tokens: torch.Tensor
    temporal_tokens: torch.Tensor
    temporal_support: torch.Tensor
    valid_mask: torch.Tensor



