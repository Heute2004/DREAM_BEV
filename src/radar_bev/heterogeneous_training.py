from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from .losses import QualityAwareBEVLoss


@dataclass(frozen=True)
class HeterogeneousLossOutput:
    total: Tensor
    localization: Tensor
    consistency: Tensor
    residual_energy: Tensor
    regret: Tensor
    covariance_nll: Tensor
    acceptance: Tensor
    identity: Tensor
    identity_residual: Tensor
    identity_acceptance: Tensor


def expert_improves_target(base_error_m: Tensor, proposal_error_m: Tensor, margin_m: float = 0.02) -> Tensor:
    """Label acceptance only when the proposal beats frozen DREAM-BEV by margin."""
    if base_error_m.shape != proposal_error_m.shape:
        raise ValueError("base_error_m and proposal_error_m must have the same shape")
    if margin_m < 0.0:
        raise ValueError("margin_m must be non-negative")
    return (proposal_error_m + margin_m < base_error_m).to(dtype=proposal_error_m.dtype)


def identity_negative_terms(delta_bev_logit: Tensor, acceptance: Tensor, identity_mask: Tensor) -> dict[str, Tensor]:
    """Force an out-of-route expert toward zero residual and zero acceptance."""
    if identity_mask.ndim != 1 or identity_mask.shape[0] != delta_bev_logit.shape[0]:
        raise ValueError("identity_mask must be [batch]")
    if acceptance.shape != (delta_bev_logit.shape[0], 1):
        raise ValueError("acceptance must be [batch,1]")
    zero = delta_bev_logit.new_zeros(())
    if not bool(identity_mask.any()):
        return {"identity": zero, "identity_residual": zero, "identity_acceptance": zero}
    residual = delta_bev_logit[identity_mask].square().mean()
    acceptance_penalty = acceptance[identity_mask].square().mean()
    return {
        "identity": residual + acceptance_penalty,
        "identity_residual": residual,
        "identity_acceptance": acceptance_penalty,
    }


def heterogeneous_loss_terms(
    output: dict[str, Tensor],
    target_xy_m: Tensor,
    sample_weight: Tensor,
    identity_mask: Tensor,
    criterion: QualityAwareBEVLoss,
    acceptance_enabled: bool,
    consistency_weight: float = 0.01,
    residual_weight: float = 1e-4,
    regret_weight: float = 0.05,
    covariance_weight: float = 0.02,
    acceptance_weight: float = 0.10,
    identity_weight: float = 0.25,
    improvement_margin_m: float = 0.02,
) -> HeterogeneousLossOutput:
    """Losses for a bounded residual expert; GT never enters its runtime inputs."""
    if identity_mask.dtype != torch.bool:
        identity_mask = identity_mask.bool()
    target_mask = ~identity_mask
    zero = output["logits"].new_zeros(())
    if bool(target_mask.any()):
        localization = criterion(
            output["logits"][target_mask], target_xy_m[target_mask], sample_weight[target_mask]
        ).total
        consistency = functional.mse_loss(
            output["logits"][target_mask].sigmoid(), output["base_logits"][target_mask].sigmoid()
        )
        proposal_error = (output["proposal_xy_m"][target_mask] - target_xy_m[target_mask]).norm(dim=-1)
        base_error = (output["base_peak_xy_m"][target_mask] - target_xy_m[target_mask]).norm(dim=-1).detach()
        regret = functional.relu(proposal_error - base_error + improvement_margin_m).mean()
        covariance = output["covariance"][target_mask]
        variance = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-5)
        covariance_nll = 0.5 * (
            (output["proposal_xy_m"][target_mask] - target_xy_m[target_mask]).square() / variance
            + variance.log()
        ).sum(dim=-1).mean()
        if acceptance_enabled:
            target = expert_improves_target(base_error, proposal_error.detach(), improvement_margin_m)
            # BCEWithLogits is autocast-safe; `acceptance` remains the runtime
            # probability used by the residual gate and identity-negative loss.
            acceptance = functional.binary_cross_entropy_with_logits(
                output["acceptance_logit"][target_mask, 0].float(), target.float()
            )
        else:
            acceptance = zero
    else:
        localization = consistency = regret = covariance_nll = acceptance = zero
    residual_energy = output["delta_bev_logit"].square().mean()
    identity = identity_negative_terms(output["delta_bev_logit"], output["acceptance"], identity_mask)
    total = (
        localization
        + consistency_weight * consistency
        + residual_weight * residual_energy
        + regret_weight * regret
        + covariance_weight * covariance_nll
        + acceptance_weight * acceptance
        + identity_weight * identity["identity"]
    )
    return HeterogeneousLossOutput(
        total=total,
        localization=localization.detach(),
        consistency=consistency.detach(),
        residual_energy=residual_energy.detach(),
        regret=regret.detach(),
        covariance_nll=covariance_nll.detach(),
        acceptance=acceptance.detach(),
        identity=identity["identity"].detach(),
        identity_residual=identity["identity_residual"].detach(),
        identity_acceptance=identity["identity_acceptance"].detach(),
    )



