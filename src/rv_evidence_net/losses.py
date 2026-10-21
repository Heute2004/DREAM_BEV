from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional


@dataclass(frozen=True)
class DetectionLossOutput:
    total: Tensor
    objectness: Tensor
    box: Tensor
    quality: Tensor


def _aligned_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    intersection_top_left = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    intersection_bottom_right = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    intersection = (intersection_bottom_right - intersection_top_left).clamp_min(0.0).prod(dim=-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0.0).prod(dim=-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0.0).prod(dim=-1)
    return intersection / (area_a + area_b - intersection).clamp_min(1e-6)


def anchor_free_detection_loss(
    detection_maps: tuple[dict[str, Tensor], ...],
    boxes: Tensor,
    valid_mask: Tensor,
    strides: tuple[int, ...] = (8, 16, 32),
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> DetectionLossOutput:
    """Single-person anchor-free loss for manually reviewed YOLO boxes.

    One object center is assigned to a scale according to its square-root area.
    Empty visual frames contribute only objectness negatives.
    """

    if boxes.ndim != 3 or boxes.shape[-1] != 4 or valid_mask.shape != boxes.shape[:2]:
        raise ValueError("boxes must be [batch, proposals, 4] and valid_mask must match")
    if len(detection_maps) != len(strides):
        raise ValueError("each detection level must have a stride")
    batch_size = boxes.shape[0]
    objectness_loss = boxes.new_zeros(())
    box_loss = boxes.new_zeros(())
    quality_loss = boxes.new_zeros(())
    positive_count = boxes.new_zeros(())

    level_targets: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for level_maps in detection_maps:
        height, width = level_maps["objectness"].shape[-2:]
        level_targets.append(
            (
                boxes.new_zeros(batch_size, 1, height, width),
                boxes.new_zeros(batch_size, 2, height, width),
                boxes.new_zeros(batch_size, 2, height, width),
                torch.zeros(batch_size, 1, height, width, dtype=torch.bool, device=boxes.device),
                boxes.new_zeros(batch_size, 4, height, width),
            )
        )

    for batch_index in range(batch_size):
        for proposal_index in range(boxes.shape[1]):
            if not valid_mask[batch_index, proposal_index]:
                continue
            x1, y1, x2, y2 = boxes[batch_index, proposal_index]
            box_width = (x2 - x1).clamp_min(1.0)
            box_height = (y2 - y1).clamp_min(1.0)
            object_extent = torch.sqrt(box_width * box_height)
            level_index = 0 if object_extent < 96 else 1 if object_extent < 192 else 2
            stride = strides[level_index]
            heatmap, size_target, offset_target, positive_mask, box_target = level_targets[level_index]
            center_x = (x1 + x2) * 0.5 / stride
            center_y = (y1 + y2) * 0.5 / stride
            grid_x = int(center_x.floor().clamp(0, heatmap.shape[-1] - 1).item())
            grid_y = int(center_y.floor().clamp(0, heatmap.shape[-2] - 1).item())
            heatmap[batch_index, 0, grid_y, grid_x] = 1.0
            size_target[batch_index, :, grid_y, grid_x] = torch.log(torch.stack((box_width, box_height)) / stride)
            offset_target[batch_index, :, grid_y, grid_x] = torch.stack((center_x - grid_x, center_y - grid_y))
            positive_mask[batch_index, 0, grid_y, grid_x] = True
            box_target[batch_index, :, grid_y, grid_x] = torch.stack((x1, y1, x2, y2))

    for level_maps, (heatmap, size_target, offset_target, positive_mask, box_target) in zip(detection_maps, level_targets, strict=True):
        logits = level_maps["objectness"]
        probability = logits.sigmoid()
        binary_cross_entropy = functional.binary_cross_entropy_with_logits(logits, heatmap, reduction="none")
        probability_target = probability * heatmap + (1.0 - probability) * (1.0 - heatmap)
        alpha = focal_alpha * heatmap + (1.0 - focal_alpha) * (1.0 - heatmap)
        objectness_loss = objectness_loss + (alpha * (1.0 - probability_target).pow(focal_gamma) * binary_cross_entropy).sum()
        expanded_positive = positive_mask.expand_as(level_maps["size"])
        if expanded_positive.any():
            box_loss = box_loss + functional.smooth_l1_loss(
                level_maps["size"][expanded_positive], size_target[expanded_positive], reduction="sum"
            )
            box_loss = box_loss + functional.smooth_l1_loss(
                level_maps["offset"][expanded_positive], offset_target[expanded_positive], reduction="sum"
            )
        if "quality" in level_maps and positive_mask.any():
            batch_indices, grid_y, grid_x = positive_mask[:, 0].nonzero(as_tuple=True)
            predicted_size = level_maps["size"][batch_indices, :, grid_y, grid_x].clamp(-4.0, 4.0).exp() * stride
            predicted_offset = level_maps["offset"][batch_indices, :, grid_y, grid_x].sigmoid()
            center_x = (grid_x.to(dtype=boxes.dtype) + predicted_offset[:, 0]) * stride
            center_y = (grid_y.to(dtype=boxes.dtype) + predicted_offset[:, 1]) * stride
            predicted_boxes = torch.stack(
                (
                    center_x - predicted_size[:, 0] * 0.5,
                    center_y - predicted_size[:, 1] * 0.5,
                    center_x + predicted_size[:, 0] * 0.5,
                    center_y + predicted_size[:, 1] * 0.5,
                ),
                dim=-1,
            )
            quality_target = _aligned_iou(predicted_boxes.detach(), box_target[batch_indices, :, grid_y, grid_x])
            quality_loss = quality_loss + functional.binary_cross_entropy_with_logits(
                level_maps["quality"][batch_indices, 0, grid_y, grid_x], quality_target, reduction="sum"
            )
        positive_count = positive_count + positive_mask.sum()

    normalizer = positive_count.clamp_min(1.0)
    objectness_loss = objectness_loss / normalizer
    box_loss = box_loss / normalizer
    quality_loss = quality_loss / normalizer
    return DetectionLossOutput(
        total=objectness_loss + 2.0 * box_loss + 0.5 * quality_loss,
        objectness=objectness_loss,
        box=box_loss,
        quality=quality_loss,
    )


def temporal_recovery_loss(
    clean_tokens: Tensor,
    recovered_tokens: Tensor,
    temporal_support: Tensor,
    valid_mask: Tensor,
    corruption_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Self-supervise short visual evidence repair with token corruption only."""

    if corruption_mask.shape != valid_mask.shape:
        raise ValueError("corruption_mask must match valid_mask")
    active = valid_mask & corruption_mask
    if not active.any():
        zero = clean_tokens.new_zeros(())
        return zero, zero
    clean = functional.normalize(clean_tokens.detach()[active], dim=-1)
    recovered = functional.normalize(recovered_tokens[active], dim=-1)
    reconstruction = (1.0 - (clean * recovered).sum(dim=-1)).mean()
    support_target = (~corruption_mask & valid_mask).to(dtype=temporal_support.dtype)
    # Probability BCE is intentionally evaluated in FP32 because PyTorch marks
    # it unsafe under CUDA autocast. The recovery module exposes probabilities
    # as part of its public interface, so logits are not available here.
    with torch.autocast(device_type=temporal_support.device.type, enabled=False):
        support = functional.binary_cross_entropy(
            temporal_support[valid_mask].float().clamp(1e-5, 1.0 - 1e-5),
            support_target[valid_mask].float(),
        )
    return reconstruction, support


def temporal_token_contrastive_loss(tokens: Tensor, valid_mask: Tensor, temperature: float = 0.1) -> Tensor:
    """Learn discriminative proposal evidence from adjacent valid observations.

    Each clip's adjacent proposal is the positive. Other clip proposals in the
    mini-batch act as negatives, which prevents a trivial constant-token collapse.
    """

    if tokens.ndim != 4 or valid_mask.shape != tokens.shape[:3]:
        raise ValueError("token and valid-mask shapes are incompatible")
    losses: list[Tensor] = []
    for time_index in range(1, tokens.shape[1]):
        active = valid_mask[:, time_index] & valid_mask[:, time_index - 1]
        current = tokens[:, time_index][active]
        previous = tokens[:, time_index - 1][active]
        if current.shape[0] < 2:
            continue
        current = functional.normalize(current, dim=-1)
        previous = functional.normalize(previous.detach(), dim=-1)
        logits = current @ previous.transpose(0, 1) / temperature
        labels = torch.arange(current.shape[0], device=tokens.device)
        losses.append(functional.cross_entropy(logits, labels))
    if not losses:
        return tokens.new_zeros(())
    return torch.stack(losses).mean()


def corrupt_short_evidence(tokens: Tensor, valid_mask: Tensor, probability: float = 0.5) -> tuple[Tensor, Tensor]:
    """Corrupt at most one valid instant per proposal without changing detection validity."""

    if tokens.ndim != 4 or valid_mask.shape != tokens.shape[:3]:
        raise ValueError("token and valid-mask shapes are incompatible")
    corrupted = tokens.clone()
    corruption_mask = torch.zeros_like(valid_mask)
    batch_size, time_steps, proposal_count, _ = tokens.shape
    for batch_index in range(batch_size):
        for proposal_index in range(proposal_count):
            valid_times = valid_mask[batch_index, :, proposal_index].nonzero(as_tuple=False).flatten()
            if valid_times.numel() < 2 or torch.rand((), device=tokens.device) > probability:
                continue
            # Recovery is causal, so never corrupt the first valid observation:
            # it has no preceding visual evidence from which to recover.
            chosen = valid_times[1:][torch.randint(valid_times.numel() - 1, (), device=tokens.device)]
            standard_deviation = tokens[batch_index, valid_times, proposal_index].detach().std().clamp_min(0.1)
            corrupted[batch_index, chosen, proposal_index] = torch.randn_like(tokens[batch_index, chosen, proposal_index]) * standard_deviation
            corruption_mask[batch_index, chosen, proposal_index] = True
    return corrupted, corruption_mask



