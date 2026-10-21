from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional

from .quantized_lateral_residual import fixed_lateral_offsets
from .lattice_fusion import DenseLatticeFusionOutput


def _weighted_mean(value: Tensor, sample_weight: Tensor) -> Tensor:
    return (value * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


def quantized_lateral_residual_loss(
    output: DenseLatticeFusionOutput,
    target_xy_m: Tensor,
    sample_weight: Tensor,
    *,
    radius_m: float,
    step_m: float,
    margin_weight: float = 0.15,
) -> tuple[Tensor, dict[str, Tensor]]:
    output.validate()
    batch = output.radar_xy_m.shape[0]
    if tuple(target_xy_m.shape) != (batch, 2) or tuple(sample_weight.shape) != (batch,):
        raise ValueError("target_xy_m must be [batch,2] and sample_weight [batch]")
    offsets = fixed_lateral_offsets(
        radius_m=radius_m,
        step_m=step_m,
        device=target_xy_m.device,
        dtype=target_xy_m.dtype,
    )
    target_offset = (target_xy_m[:, 0:1] - output.radar_xy_m[:, 0:1]).clamp(
        -radius_m, radius_m
    )
    target_index = (target_offset - offsets[None]).abs().argmin(dim=1)
    target_index = torch.where(
        output.identity_mask, torch.zeros_like(target_index), target_index
    )
    class_raw = functional.cross_entropy(
        output.emission_logits, target_index, reduction="none"
    )
    class_loss = _weighted_mean(class_raw, sample_weight)
    row = torch.arange(batch, device=target_xy_m.device)
    selected_offset = offsets[target_index][:, None]
    residual_target = (target_offset - selected_offset).clamp(
        -0.5 * step_m, 0.5 * step_m
    )
    residual_target = torch.where(
        output.identity_mask[:, None], torch.zeros_like(residual_target), residual_target
    )
    residual_raw = functional.smooth_l1_loss(
        output.candidate_residual_m[row, target_index][:, None],
        residual_target,
        beta=0.005,
        reduction="none",
    )[:, 0]
    residual_loss = _weighted_mean(residual_raw, sample_weight)
    decoded_error = torch.linalg.vector_norm(output.decoded_xy_m - target_xy_m, dim=1)
    margin_raw = functional.softplus((decoded_error - 0.30) / 0.025) * 0.025
    margin_loss = _weighted_mean(
        margin_raw * (~output.identity_mask).to(margin_raw.dtype), sample_weight
    )
    total = class_loss + residual_loss + float(margin_weight) * margin_loss
    return total, {
        "quantized_classification": class_loss,
        "quantized_residual": residual_loss,
        "quantized_ap30_margin": margin_loss,
    }
