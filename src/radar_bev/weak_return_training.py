from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from .recovery_training import recovery_loss_terms
from .losses import QualityAwareBEVLoss


@dataclass(frozen=True)
class WeakReturnLossOutput:
    total: Tensor
    base: object
    coarse: Tensor
    fine: Tensor
    threshold_flip: Tensor


def weak_return_loss_terms(
    output: dict[str, Tensor],
    target_xy_m: Tensor,
    history_target_xy_m: Tensor,
    history_target_valid: Tensor,
    sample_weight: Tensor,
    identity_mask: Tensor,
    criterion: QualityAwareBEVLoss,
    acceptance_enabled: bool,
    coarse_weight: float = 0.10,
    fine_weight: float = 0.15,
    threshold_flip_weight: float = 0.30,
    **base_kwargs: float,
) -> WeakReturnLossOutput:
    base = recovery_loss_terms(
        output=output, target_xy_m=target_xy_m, history_target_xy_m=history_target_xy_m,
        history_target_valid=history_target_valid, sample_weight=sample_weight,
        identity_mask=identity_mask, criterion=criterion, acceptance_enabled=acceptance_enabled,
        gain_enabled=False, **base_kwargs,
    )
    mask = ~identity_mask.bool()
    zero = target_xy_m.new_zeros(())
    if "coarse_xy_m" not in output or not bool(mask.any()):
        return WeakReturnLossOutput(total=base.total, base=base, coarse=zero, fine=zero, threshold_flip=zero)
    coarse = functional.smooth_l1_loss(output["coarse_xy_m"][mask], target_xy_m[mask])
    fine = functional.smooth_l1_loss(output["proposal_xy_m"][mask], target_xy_m[mask])
    base_error = (output["base_peak_xy_m"][mask] - target_xy_m[mask]).norm(dim=-1).detach()
    proposal_error = (output["proposal_xy_m"][mask] - target_xy_m[mask]).norm(dim=-1)
    acceptance = output["acceptance"][mask, 0]
    # Protect already-correct 0.30 m detections from being pushed across the paper's strict threshold.
    threshold_flip = (acceptance * (base_error <= 0.30).to(dtype=proposal_error.dtype) * functional.relu(proposal_error - 0.30)).mean()
    return WeakReturnLossOutput(
        total=base.total + coarse_weight * coarse + fine_weight * fine + threshold_flip_weight * threshold_flip,
        base=base, coarse=coarse.detach(), fine=fine.detach(), threshold_flip=threshold_flip.detach(),
    )



