from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .losses import anchor_free_detection_loss, corrupt_short_evidence, temporal_recovery_loss, temporal_token_contrastive_loss
from .model import RVEvidenceNetSTMamba


@dataclass
class EarlyStoppingTracker:
    """Track validation improvement without coupling policy to the training CLI."""

    patience: int
    min_delta: float
    best: float = float("inf")
    stale_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")

    def update(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.stale_epochs = 0
            return True
        self.stale_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.stale_epochs >= self.patience


@dataclass(frozen=True)
class BatchLoss:
    total: Tensor
    detection: Tensor
    objectness: Tensor
    box: Tensor
    quality: Tensor
    recovery: Tensor
    support: Tensor
    contrastive: Tensor


def flatten_detection_maps(output_maps: tuple[dict[str, Tensor], ...]) -> tuple[dict[str, Tensor], ...]:
    """Convert `[batch, time, ...]` maps to `[batch*time, ...]` maps."""

    flattened = []
    for level in output_maps:
        flattened.append({name: value.reshape(-1, *value.shape[2:]) for name, value in level.items()})
    return tuple(flattened)


def compute_batch_loss(model: RVEvidenceNetSTMamba, batch: dict[str, Tensor], stage: str) -> BatchLoss:
    """Compute a stage-specific loss using only the frozen M2 visual labels."""

    images = batch["images"]
    boxes = batch["boxes"]
    valid_mask = batch["valid_mask"].bool()
    if stage == "spatial":
        flattened_images = images.reshape(-1, *images.shape[2:])
        flattened_boxes = boxes.reshape(-1, boxes.shape[2], 4)
        flattened_valid = valid_mask.reshape(-1, valid_mask.shape[2])
        output = model.forward_frame(
            flattened_images,
            proposals=flattened_boxes,
            proposal_valid_mask=flattened_valid,
            extract_evidence=False,
        )
        detection = anchor_free_detection_loss(output.detection_maps, flattened_boxes, flattened_valid, model.strides)
        zero = detection.total.new_zeros(())
        return BatchLoss(detection.total, detection.total, detection.objectness, detection.box, detection.quality, zero, zero, zero)

    output = model.forward_clip(images, proposals=boxes, proposal_valid_mask=valid_mask)
    detection = anchor_free_detection_loss(
        flatten_detection_maps(output.detection_maps),
        boxes.reshape(-1, boxes.shape[2], 4),
        valid_mask.reshape(-1, valid_mask.shape[2]),
        model.strides,
    )
    zero = detection.total.new_zeros(())
    recovery_loss = zero
    support_loss = zero
    contrastive_loss = zero
    if stage in {"temporal", "joint"}:
        geometry = model._geometry(output.boxes, images.shape[-2], images.shape[-1])
        corrupted_tokens, corruption_mask = corrupt_short_evidence(output.raw_tokens, valid_mask)
        recovered = model.recovery(corrupted_tokens, geometry, output.raw_scores, valid_mask)
        recovery_loss, support_loss = temporal_recovery_loss(
            output.raw_tokens,
            recovered.refined_tokens,
            recovered.temporal_support,
            valid_mask,
            corruption_mask,
        )
        contrastive_loss = temporal_token_contrastive_loss(output.raw_tokens, valid_mask)
    if stage == "temporal":
        total = recovery_loss + 0.25 * support_loss + 0.1 * contrastive_loss
    elif stage == "joint":
        total = detection.total + 0.5 * recovery_loss + 0.125 * support_loss + 0.05 * contrastive_loss
    else:
        raise ValueError("stage must be spatial, temporal, or joint")
    return BatchLoss(total, detection.total, detection.objectness, detection.box, detection.quality, recovery_loss, support_loss, contrastive_loss)



