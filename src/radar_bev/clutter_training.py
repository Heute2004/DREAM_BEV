from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from .recovery_training import recovery_loss_terms
from .losses import QualityAwareBEVLoss


@dataclass(frozen=True)
class ClutterLossOutput:
    total: Tensor
    base: object
    selector: Tensor
    proposal: Tensor
    selected_improvement: Tensor


def clutter_loss_terms(
    output: dict[str, Tensor],
    target_xy_m: Tensor,
    history_target_xy_m: Tensor,
    history_target_valid: Tensor,
    sample_weight: Tensor,
    identity_mask: Tensor,
    criterion: QualityAwareBEVLoss,
    acceptance_enabled: bool,
    selector_weight: float = 1.0,
    proposal_weight: float = 0.45,
    **base_kwargs: float,
) -> ClutterLossOutput:
    """Teach a hard candidate choice, while the residual remains a safe fallback."""
    base = recovery_loss_terms(
        output=output,
        target_xy_m=target_xy_m,
        history_target_xy_m=history_target_xy_m,
        history_target_valid=history_target_valid,
        sample_weight=sample_weight,
        identity_mask=identity_mask,
        criterion=criterion,
        acceptance_enabled=acceptance_enabled,
        gain_enabled=False,
        **base_kwargs,
    )
    target_mask = ~identity_mask.bool()
    zero = target_xy_m.new_zeros(())
    if not bool(target_mask.any()):
        return ClutterLossOutput(total=base.total, base=base, selector=zero, proposal=zero, selected_improvement=zero)
    candidates = output["candidate_xy_m"][target_mask]
    candidate_error = (candidates - target_xy_m[target_mask, None]).norm(dim=-1)
    winner = candidate_error.argmin(dim=1)
    selector = functional.cross_entropy(output["candidate_selector_logits"][target_mask].float(), winner)
    proposal = functional.smooth_l1_loss(output["proposal_xy_m"][target_mask], target_xy_m[target_mask])
    selected_error = (output["proposal_xy_m"][target_mask] - target_xy_m[target_mask]).norm(dim=-1)
    base_error = (output["base_peak_xy_m"][target_mask] - target_xy_m[target_mask]).norm(dim=-1)
    selected_improvement = (selected_error + 0.02 < base_error).float().mean()
    return ClutterLossOutput(
        total=base.total + selector_weight * selector + proposal_weight * proposal,
        base=base,
        selector=selector.detach(),
        proposal=proposal.detach(),
        selected_improvement=selected_improvement.detach(),
    )



