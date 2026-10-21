from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional

from .heterogeneous_training import expert_improves_target, identity_negative_terms
from .losses import QualityAwareBEVLoss


@dataclass(frozen=True)
class RecoveryLossOutput:
    total: Tensor
    localization: Tensor
    consistency: Tensor
    residual_energy: Tensor
    polar_residual_energy: Tensor
    regret: Tensor
    covariance_nll: Tensor
    acceptance: Tensor
    gain: Tensor
    edge: Tensor
    identity: Tensor


def _graph_edge_loss(
    output: dict[str, Tensor],
    target_xy_m: Tensor,
    history_target_xy_m: Tensor,
    history_target_valid: Tensor,
    target_mask: Tensor,
    radius_m: float = 0.40,
) -> Tensor:
    if "graph_link_logits" not in output or not bool(target_mask.any()):
        return target_xy_m.new_zeros(())
    current = output["candidate_xy_m"][target_mask]
    history = output["graph_history_xy_m"][target_mask]
    logits = output["graph_link_logits"][target_mask]
    current_target = target_xy_m[target_mask]
    history_target = history_target_xy_m[target_mask]
    valid_history = history_target_valid[target_mask]
    candidate_count = history.shape[1] // history_target.shape[1]
    history_gt = history_target[:, :, None, :].expand(-1, -1, candidate_count, -1).reshape_as(history)
    history_valid = valid_history[:, :, None].expand(-1, -1, candidate_count).reshape(history.shape[:2])
    current_good = (current - current_target[:, None]).norm(dim=-1) <= radius_m
    history_good = ((history - history_gt).norm(dim=-1) <= radius_m) & history_valid
    target = current_good[:, :, None] & history_good[:, None, :]
    feasible = logits > -1e3
    if not bool(feasible.any()):
        return logits.new_zeros(())
    weights = torch.where(target, torch.full_like(logits, 3.0), torch.ones_like(logits))
    return functional.binary_cross_entropy_with_logits(logits[feasible], target[feasible].to(dtype=logits.dtype), weight=weights[feasible])


def recovery_loss_terms(
    output: dict[str, Tensor],
    target_xy_m: Tensor,
    history_target_xy_m: Tensor,
    history_target_valid: Tensor,
    sample_weight: Tensor,
    identity_mask: Tensor,
    criterion: QualityAwareBEVLoss,
    acceptance_enabled: bool,
    gain_enabled: bool,
    consistency_weight: float = 0.01,
    residual_weight: float = 1e-4,
    polar_residual_weight: float = 1e-4,
    regret_weight: float = 0.05,
    covariance_weight: float = 0.02,
    acceptance_weight: float = 0.10,
    gain_weight: float = 0.05,
    edge_weight: float = 0.05,
    identity_weight: float = 0.25,
    improvement_margin_m: float = 0.02,
) -> RecoveryLossOutput:
    """Recovery losses; offline route labels form training edges and never enter runtime inputs."""
    identity_mask = identity_mask.bool()
    target_mask = ~identity_mask
    zero = output["logits"].new_zeros(())
    if bool(target_mask.any()):
        localization = criterion(output["logits"][target_mask], target_xy_m[target_mask], sample_weight[target_mask]).total
        consistency = functional.mse_loss(output["logits"][target_mask].sigmoid(), output["base_logits"][target_mask].sigmoid())
        proposal_error = (output["proposal_xy_m"][target_mask] - target_xy_m[target_mask]).norm(dim=-1)
        base_error = (output["base_peak_xy_m"][target_mask] - target_xy_m[target_mask]).norm(dim=-1).detach()
        regret = functional.relu(proposal_error - base_error + improvement_margin_m).mean()
        covariance = output["covariance"][target_mask]
        variance = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-5)
        covariance_nll = 0.5 * (((output["proposal_xy_m"][target_mask] - target_xy_m[target_mask]).square() / variance) + variance.log()).sum(dim=-1).mean()
        if acceptance_enabled:
            target = expert_improves_target(base_error, proposal_error.detach(), improvement_margin_m)
            acceptance = functional.binary_cross_entropy_with_logits(output["acceptance_logit"][target_mask, 0].float(), target.float())
        else:
            acceptance = zero
        if gain_enabled and "gain_prediction" in output:
            gain_target = (base_error - proposal_error).detach().clamp(-1.0, 1.0)
            gain = functional.smooth_l1_loss(output["gain_prediction"][target_mask, 0], gain_target)
        else:
            gain = zero
        edge = _graph_edge_loss(output, target_xy_m, history_target_xy_m, history_target_valid, target_mask)
    else:
        localization = consistency = regret = covariance_nll = acceptance = gain = edge = zero
    residual_energy = output["delta_bev_logit"].square().mean()
    polar_residual_energy = output.get("polar_delta", output["delta_bev_logit"]).square().mean()
    identity = identity_negative_terms(output["delta_bev_logit"], output["acceptance"], identity_mask)["identity"]
    total = (
        localization
        + consistency_weight * consistency
        + residual_weight * residual_energy
        + polar_residual_weight * polar_residual_energy
        + regret_weight * regret
        + covariance_weight * covariance_nll
        + acceptance_weight * acceptance
        + gain_weight * gain
        + edge_weight * edge
        + identity_weight * identity
    )
    return RecoveryLossOutput(
        total=total,
        localization=localization.detach(),
        consistency=consistency.detach(),
        residual_energy=residual_energy.detach(),
        polar_residual_energy=polar_residual_energy.detach(),
        regret=regret.detach(),
        covariance_nll=covariance_nll.detach(),
        acceptance=acceptance.detach(),
        gain=gain.detach(),
        edge=edge.detach(),
        identity=identity.detach(),
    )



