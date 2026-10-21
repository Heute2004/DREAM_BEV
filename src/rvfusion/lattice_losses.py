from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from .lattice_fusion import DenseLatticeFusionOutput


@dataclass(frozen=True)
class LatticeLossWeights:
    candidate: float = 1.0
    residual: float = 0.25
    continuous_location: float = 0.0
    continuous_ap30_margin: float = 0.0
    mixture_nll: float = 0.40
    gain: float = 1.0
    harm: float = 2.0
    utility: float = 0.0
    candidate_utility: float = 0.0
    candidate_utility_rank: float = 0.0
    regret: float = 4.0
    identity: float = 10.0


def _weighted_mean(value: Tensor, sample_weight: Tensor) -> Tensor:
    while sample_weight.ndim < value.ndim:
        sample_weight = sample_weight.unsqueeze(-1)
    numerator = (value * sample_weight).sum()
    denominator = sample_weight.expand_as(value).sum().clamp_min(1e-6)
    return numerator / denominator


def _smooth_hit(error_m: Tensor, threshold_m: float, temperature_m: float) -> Tensor:
    return torch.sigmoid((float(threshold_m) - error_m) / float(temperature_m))


def dense_lattice_loss(
    output: DenseLatticeFusionOutput,
    target_xy_m: Tensor,
    sample_weight: Tensor,
    *,
    weights: LatticeLossWeights | None = None,
    threshold_temperature_m: float = 0.02,
    utility_temperature: float = 0.10,
    minimum_useful_gain_m: float = 0.02,
    candidate_positive_weight: float = 10.0,
    candidate_presence_weight: float = 1.0,
    nearest_x_background_weight: float = 0.1,
    gain_positive_weight: float = 6.0,
    harm_positive_weight: float = 10.0,
    candidate_target_mode: str = "utility",
    candidate_target_sigma_m: float = 0.075,
    utility_target_mode: str = "broad",
    continuous_target_mode: str = "candidate_span",
    continuous_radius_m: float = 0.5,
    continuous_positive_weight: float = 1.0,
    continuous_ap30_margin_temperature_m: float = 0.025,
) -> tuple[Tensor, dict[str, Tensor]]:
    output.validate()
    batch, candidate_count = output.candidate_x_m.shape
    if tuple(target_xy_m.shape) != (batch, 2):
        raise ValueError("target_xy_m must be [batch,2]")
    if tuple(sample_weight.shape) != (batch,):
        raise ValueError("sample_weight must be [batch]")
    if not torch.isfinite(target_xy_m).all() or not torch.isfinite(sample_weight).all():
        raise ValueError("targets and sample weights must be finite")
    if candidate_positive_weight <= 0.0:
        raise ValueError("candidate_positive_weight must be positive")
    if candidate_presence_weight < 0.0:
        raise ValueError("candidate_presence_weight must be non-negative")
    if not 0.0 <= nearest_x_background_weight <= 1.0:
        raise ValueError(
            "nearest_x_background_weight must be in [0,1]"
        )
    if gain_positive_weight <= 0.0 or harm_positive_weight <= 0.0:
        raise ValueError("gain/harm positive weights must be positive")
    if candidate_target_mode not in {"utility", "nearest_x"}:
        raise ValueError(
            "candidate_target_mode must be utility or nearest_x"
        )
    if candidate_target_sigma_m <= 0.0:
        raise ValueError("candidate_target_sigma_m must be positive")
    if continuous_target_mode not in {"candidate_span", "radar_radius"}:
        raise ValueError(
            "continuous_target_mode must be candidate_span or radar_radius"
        )
    if continuous_radius_m <= 0.0:
        raise ValueError("continuous_radius_m must be positive")
    if continuous_positive_weight <= 0.0:
        raise ValueError(
            "continuous_positive_weight must be positive"
        )
    if continuous_ap30_margin_temperature_m <= 0.0:
        raise ValueError(
            "continuous_ap30_margin_temperature_m must be positive"
        )
    if utility_target_mode not in {
        "broad",
        "ap30_crossing",
        "decomposed_correctness",
    }:
        raise ValueError(
            "utility_target_mode must be broad, ap30_crossing, "
            "or decomposed_correctness"
        )
    active_weights = weights or LatticeLossWeights()

    target_x = target_xy_m[:, 0:1]
    target_y = target_xy_m[:, 1:2]
    candidate_y = output.radar_xy_m[:, 1:2].expand(-1, candidate_count)
    base_candidate_x = output.candidate_x_m - output.candidate_residual_m
    base_candidate_x = base_candidate_x.clone()
    base_candidate_x[:, 0] = output.radar_xy_m[:, 0]
    center_error = torch.sqrt(
        (base_candidate_x - target_x).square()
        + (candidate_y - target_y).square()
    )
    refined_error = torch.sqrt(
        (output.candidate_x_m - target_x).square()
        + (candidate_y - target_y).square()
    )
    no_op_error = refined_error[:, 0:1]
    utility = (
        _smooth_hit(center_error, 0.30, threshold_temperature_m)
        - _smooth_hit(no_op_error, 0.30, threshold_temperature_m)
        + 0.25
        * (
            _smooth_hit(center_error, 0.25, threshold_temperature_m)
            - _smooth_hit(no_op_error, 0.25, threshold_temperature_m)
        )
        + 0.25
        * (
            _smooth_hit(center_error, 0.50, threshold_temperature_m)
            - _smooth_hit(no_op_error, 0.50, threshold_temperature_m)
        )
        - 2.0 * (center_error - no_op_error).clamp_min(0.0)
    )
    utility = utility.masked_fill(~output.candidate_valid_mask, -1e4)
    utility[:, 0] = 0.0
    best_utility, best_index = utility.max(dim=1)
    best_index = torch.where(
        (best_utility > 0.01) & ~output.identity_mask,
        best_index,
        torch.zeros_like(best_index),
    )
    oracle_positive = (best_index != 0) & ~output.identity_mask
    if candidate_target_mode == "nearest_x":
        target_logits = -0.5 * (
            (
                base_candidate_x[:, 1:] - target_x
            )
            / float(candidate_target_sigma_m)
        ).square()
        target_logits = target_logits.masked_fill(
            ~output.candidate_valid_mask[:, 1:],
            -1e4,
        )
        soft_target = target_logits.softmax(dim=1)
        location_target_index = target_logits.argmax(dim=1) + 1
        candidate_sample_weight = (
            sample_weight
            * (~output.identity_mask).to(sample_weight.dtype)
            * torch.where(
                oracle_positive,
                torch.ones_like(sample_weight),
                torch.full_like(
                    sample_weight,
                    float(nearest_x_background_weight),
                ),
            )
        )
    else:
        soft_target = (
            utility[:, 1:] / utility_temperature
        ).softmax(dim=1)
        location_target_index = best_index
        candidate_sample_weight = (
            sample_weight
            * oracle_positive.to(sample_weight.dtype)
        )
    candidate_cross_entropy = -(
        soft_target.detach()
        * output.emission_logits[:, 1:].log_softmax(dim=1)
    ).sum(dim=1)
    candidate_classification_loss = _weighted_mean(
        candidate_cross_entropy,
        candidate_sample_weight,
    )
    presence_logit = output.candidate_presence_logit[:, 0]
    presence_target = oracle_positive.to(presence_logit.dtype)
    presence_raw = functional.binary_cross_entropy_with_logits(
        presence_logit.float(),
        presence_target.float(),
        reduction="none",
    )
    presence_sample_weight = (
        sample_weight
        * (~output.identity_mask).to(sample_weight.dtype)
        * torch.where(
            presence_target.bool(),
            torch.full_like(sample_weight, float(candidate_positive_weight)),
            torch.ones_like(sample_weight),
        )
    )
    candidate_presence_loss = _weighted_mean(
        presence_raw,
        presence_sample_weight,
    )
    candidate_loss = (
        candidate_classification_loss
        + float(candidate_presence_weight) * candidate_presence_loss
    )

    row_index = torch.arange(batch, device=target_xy_m.device)
    selected_base_x = base_candidate_x[row_index, location_target_index]
    residual_target = (
        target_xy_m[:, 0] - selected_base_x
    ).clamp(
        -0.5 * 0.025,
        0.5 * 0.025,
    )
    predicted_residual = output.candidate_residual_m[
        row_index,
        location_target_index,
    ]
    residual_raw = functional.smooth_l1_loss(
        predicted_residual,
        residual_target,
        reduction="none",
        beta=0.005,
    )
    residual_mask = (
        ~output.identity_mask
        if candidate_target_mode == "nearest_x"
        else best_index != 0
    ).to(residual_raw.dtype)
    residual_loss = _weighted_mean(
        residual_raw * residual_mask,
        sample_weight,
    )

    if continuous_target_mode == "radar_radius":
        minimum_candidate_x = (
            output.radar_xy_m[:, 0:1] - continuous_radius_m
        )
        maximum_candidate_x = (
            output.radar_xy_m[:, 0:1] + continuous_radius_m
        )
    else:
        minimum_candidate_x = output.candidate_x_m.amin(
            dim=1,
            keepdim=True,
        )
        maximum_candidate_x = output.candidate_x_m.amax(
            dim=1,
            keepdim=True,
        )
    continuous_target_x = torch.maximum(
        torch.minimum(target_x, maximum_candidate_x),
        minimum_candidate_x,
    )
    continuous_location_raw = functional.smooth_l1_loss(
        output.continuous_x_m,
        continuous_target_x,
        reduction="none",
        beta=0.025,
    )[:, 0]
    continuous_corrected_error = torch.sqrt(
        (continuous_target_x - target_x).square()
        + (output.radar_xy_m[:, 1:2] - target_y).square()
    )[:, 0]
    continuous_positive = (
        (no_op_error[:, 0] > 0.30)
        & (continuous_corrected_error <= 0.30)
    )
    continuous_sample_weight = (
        sample_weight
        * (~output.identity_mask).to(sample_weight.dtype)
        * torch.where(
            continuous_positive,
            torch.full_like(
                sample_weight,
                float(continuous_positive_weight),
            ),
            torch.ones_like(sample_weight),
        )
    )
    continuous_location_loss = _weighted_mean(
        continuous_location_raw,
        continuous_sample_weight,
    )
    continuous_prediction_error = torch.sqrt(
        (output.continuous_x_m - target_x).square()
        + (output.radar_xy_m[:, 1:2] - target_y).square()
        + 1e-12
    )[:, 0]
    continuous_recoverable = (
        (no_op_error[:, 0] > 0.30)
        & (continuous_corrected_error <= 0.30)
    )
    continuous_protected = no_op_error[:, 0] <= 0.30
    continuous_margin_mask = (
        continuous_recoverable | continuous_protected
    )
    continuous_margin_raw = (
        functional.softplus(
            (
                continuous_prediction_error - 0.30
            )
            / float(continuous_ap30_margin_temperature_m)
        )
        * float(continuous_ap30_margin_temperature_m)
    )
    continuous_margin_sample_weight = (
        continuous_sample_weight
        * continuous_margin_mask.to(continuous_sample_weight.dtype)
    )
    continuous_ap30_margin_loss = _weighted_mean(
        continuous_margin_raw,
        continuous_margin_sample_weight,
    )

    sigma_m = output.candidate_log_sigma_m.exp().clamp(0.015, 0.15)
    component_log_probability = (
        -0.5 * ((target_x - output.candidate_x_m) / sigma_m).square()
        - sigma_m.log()
    )
    mixture_log_probability = torch.logsumexp(
        output.posterior.clamp_min(1e-12).log() + component_log_probability,
        dim=1,
    )
    mixture_nll = _weighted_mean(-mixture_log_probability, sample_weight)

    selected_index = output.selected_index.detach()
    lattice_selected_error = refined_error[row_index, selected_index]
    selected_error = (
        torch.sqrt(
            (
                output.correction_x_m[:, 0]
                - target_xy_m[:, 0]
            ).square()
            + (
                output.radar_xy_m[:, 1]
                - target_xy_m[:, 1]
            ).square()
        )
        if output.correction_source == "continuous"
        else lattice_selected_error
    )
    radar_error = no_op_error[:, 0]
    crossing_gain = (radar_error > 0.30) & (selected_error <= 0.30)
    meaningful_gain = selected_error + minimum_useful_gain_m < radar_error
    protected_harm = (
        ((radar_error <= 0.25) & (selected_error > 0.25))
        | ((radar_error <= 0.30) & (selected_error > 0.30))
        | ((radar_error <= 0.50) & (selected_error > 0.50))
        | (selected_error > radar_error + minimum_useful_gain_m)
    )
    if utility_target_mode == "decomposed_correctness":
        gain_target = (
            (radar_error > 0.30)
            & ~output.identity_mask
        ).to(output.predicted_gain.dtype)
        harm_target = (
            (selected_error > 0.30)
            | protected_harm
        ).to(output.predicted_harm.dtype)
    else:
        useful_gain = (
            crossing_gain
            if utility_target_mode == "ap30_crossing"
            else crossing_gain | meaningful_gain
        )
        gain_target = (
            useful_gain
            & ~protected_harm
            & (selected_index != 0)
            & ~output.identity_mask
        ).to(output.predicted_gain.dtype)
        harm_target = protected_harm.to(output.predicted_harm.dtype)
    gain_raw = functional.binary_cross_entropy_with_logits(
        output.predicted_gain_logit[:, 0].float(),
        gain_target,
        reduction="none",
    )
    gain_sample_weight = sample_weight * torch.where(
        gain_target.bool(),
        torch.full_like(sample_weight, float(gain_positive_weight)),
        torch.ones_like(sample_weight),
    )
    gain_loss = _weighted_mean(gain_raw, gain_sample_weight)
    harm_raw = functional.binary_cross_entropy_with_logits(
        output.predicted_harm_logit[:, 0].float(),
        harm_target,
        reduction="none",
    )
    harm_sample_weight = sample_weight * torch.where(
        harm_target.bool(),
        torch.full_like(sample_weight, float(harm_positive_weight)),
        torch.ones_like(sample_weight),
    )
    harm_loss = _weighted_mean(harm_raw, harm_sample_weight)
    utility_raw = functional.binary_cross_entropy_with_logits(
        output.predicted_utility_logit[:, 0].float(),
        gain_target,
        reduction="none",
    )
    utility_sample_weight = sample_weight * torch.where(
        gain_target.bool(),
        torch.full_like(sample_weight, float(gain_positive_weight)),
        torch.where(
            protected_harm,
            torch.full_like(
                sample_weight,
                float(harm_positive_weight),
            ),
            torch.ones_like(sample_weight),
        ),
    )
    utility_loss = _weighted_mean(utility_raw, utility_sample_weight)
    candidate_crossing_gain = (
        (radar_error[:, None] > 0.30)
        & (refined_error <= 0.30)
    )
    candidate_meaningful_gain = (
        refined_error + minimum_useful_gain_m < radar_error[:, None]
    )
    candidate_protected_harm = (
        ((radar_error[:, None] <= 0.25) & (refined_error > 0.25))
        | ((radar_error[:, None] <= 0.30) & (refined_error > 0.30))
        | ((radar_error[:, None] <= 0.50) & (refined_error > 0.50))
        | (
            refined_error
            > radar_error[:, None] + minimum_useful_gain_m
        )
    )
    candidate_useful_gain = (
        candidate_crossing_gain
        if utility_target_mode in {
            "ap30_crossing",
            "decomposed_correctness",
        }
        else candidate_crossing_gain | candidate_meaningful_gain
    )
    candidate_utility_target = (
        candidate_useful_gain
        & ~candidate_protected_harm
        & output.candidate_valid_mask
        & ~output.identity_mask[:, None]
    )
    candidate_utility_target[:, 0] = False
    candidate_utility_raw = functional.binary_cross_entropy_with_logits(
        output.candidate_utility_logit.float(),
        candidate_utility_target.to(output.candidate_utility_logit.dtype),
        reduction="none",
    )
    candidate_utility_sample_weight = (
        sample_weight[:, None]
        * output.candidate_valid_mask.to(sample_weight.dtype)
        * (~output.identity_mask[:, None]).to(sample_weight.dtype)
        * torch.where(
            candidate_utility_target,
            torch.full_like(
                candidate_utility_raw,
                float(gain_positive_weight),
            ),
            torch.where(
                candidate_protected_harm,
                torch.full_like(
                    candidate_utility_raw,
                    float(harm_positive_weight),
                ),
                torch.ones_like(candidate_utility_raw),
            ),
        )
    )
    candidate_utility_sample_weight[:, 0] = 0.0
    candidate_utility_loss = _weighted_mean(
        candidate_utility_raw,
        candidate_utility_sample_weight,
    )
    candidate_utility_rank_raw = -(
        soft_target.detach()
        * output.candidate_utility_logit[:, 1:].log_softmax(dim=1)
    ).sum(dim=1)
    candidate_utility_rank_loss = _weighted_mean(
        candidate_utility_rank_raw,
        candidate_sample_weight,
    )

    expected_error = (
        selected_error
        if output.correction_source == "continuous"
        else (output.posterior * refined_error).sum(dim=1)
    )
    regret_raw = (expected_error - radar_error).clamp_min(0.0)
    regret_loss = _weighted_mean(regret_raw, sample_weight)
    identity_raw = (
        -output.posterior[:, 0].clamp_min(1e-12).log()
        + output.effective_acceptance[:, 0]
    ) * output.identity_mask.to(output.posterior.dtype)
    identity_loss = _weighted_mean(identity_raw, sample_weight)
    selected_non_noop = selected_index != 0
    oracle_non_noop = best_index != 0
    selected_oracle_match = selected_index == best_index
    active_candidate = ~output.identity_mask
    oracle_positive = oracle_non_noop & active_candidate
    oracle_negative = ~oracle_non_noop & active_candidate
    selected_positive = selected_non_noop & active_candidate
    oracle_non_noop_recall = (
        (selected_oracle_match & oracle_positive).float().sum()
        / oracle_positive.float().sum().clamp_min(1.0)
    )
    candidate_presence_recall = (
        (selected_positive & oracle_positive).float().sum()
        / oracle_positive.float().sum().clamp_min(1.0)
    )
    candidate_false_positive_rate = (
        (selected_positive & oracle_negative).float().sum()
        / oracle_negative.float().sum().clamp_min(1.0)
    )
    candidate_presence_precision = (
        (selected_positive & oracle_positive).float().sum()
        / selected_positive.float().sum().clamp_min(1.0)
    )
    selected_useful = (
        selected_positive
        & ~protected_harm
        & (
            lattice_selected_error + minimum_useful_gain_m
            < radar_error
        )
    )
    candidate_useful_recall = (
        (selected_useful & oracle_positive).float().sum()
        / oracle_positive.float().sum().clamp_min(1.0)
    )
    candidate_presence_balanced_accuracy = 0.5 * (
        candidate_presence_recall + (1.0 - candidate_false_positive_rate)
    )
    predicted_gain_positive = output.predicted_gain[:, 0] >= 0.5
    predicted_harm_positive = output.predicted_harm[:, 0] >= 0.5
    predicted_utility_positive = (
        output.predicted_utility_probability[:, 0] >= 0.5
    )
    gain_positive = gain_target.bool()
    harm_positive = harm_target.bool()
    gain_recall = (
        (predicted_gain_positive & gain_positive).float().sum()
        / gain_positive.float().sum().clamp_min(1.0)
    )
    gain_false_positive_rate = (
        (predicted_gain_positive & ~gain_positive).float().sum()
        / (~gain_positive).float().sum().clamp_min(1.0)
    )
    harm_recall = (
        (predicted_harm_positive & harm_positive).float().sum()
        / harm_positive.float().sum().clamp_min(1.0)
    )
    harm_false_positive_rate = (
        (predicted_harm_positive & ~harm_positive).float().sum()
        / (~harm_positive).float().sum().clamp_min(1.0)
    )
    utility_recall = (
        (predicted_utility_positive & gain_positive).float().sum()
        / gain_positive.float().sum().clamp_min(1.0)
    )
    utility_false_positive_rate = (
        (predicted_utility_positive & ~gain_positive).float().sum()
        / (~gain_positive).float().sum().clamp_min(1.0)
    )
    selected_candidate_utility_positive = (
        output.candidate_utility_probability[
            row_index,
            selected_index,
        ]
        >= 0.5
    )
    selected_candidate_utility_target = candidate_utility_target[
        row_index,
        selected_index,
    ]
    candidate_utility_recall = (
        (
            selected_candidate_utility_positive
            & selected_candidate_utility_target
        ).float().sum()
        / selected_candidate_utility_target.float().sum().clamp_min(1.0)
    )
    candidate_utility_false_positive_rate = (
        (
            selected_candidate_utility_positive
            & ~selected_candidate_utility_target
        ).float().sum()
        / (~selected_candidate_utility_target).float().sum().clamp_min(1.0)
    )
    terms = {
        "candidate": candidate_loss,
        "candidate_classification": candidate_classification_loss,
        "candidate_presence": candidate_presence_loss,
        "residual": residual_loss,
        "continuous_location": continuous_location_loss,
        "continuous_ap30_margin": continuous_ap30_margin_loss,
        "mixture_nll": mixture_nll,
        "gain": gain_loss,
        "harm": harm_loss,
        "utility": utility_loss,
        "candidate_utility": candidate_utility_loss,
        "candidate_utility_rank": candidate_utility_rank_loss,
        "regret": regret_loss,
        "identity": identity_loss,
        "oracle_non_noop_rate": oracle_non_noop.float().mean(),
        "selected_non_noop_rate": selected_non_noop.float().mean(),
        "candidate_index_accuracy": selected_oracle_match.float().mean(),
        "oracle_non_noop_recall": oracle_non_noop_recall,
        "candidate_presence_recall": candidate_presence_recall,
        "candidate_false_positive_rate": candidate_false_positive_rate,
        "candidate_presence_precision": candidate_presence_precision,
        "candidate_useful_recall": candidate_useful_recall,
        "candidate_presence_balanced_accuracy": (
            candidate_presence_balanced_accuracy
        ),
        "gain_positive_rate": gain_target.mean(),
        "harm_positive_rate": harm_target.mean(),
        "gain_recall": gain_recall,
        "gain_false_positive_rate": gain_false_positive_rate,
        "harm_recall": harm_recall,
        "harm_false_positive_rate": harm_false_positive_rate,
        "utility_recall": utility_recall,
        "utility_false_positive_rate": utility_false_positive_rate,
        "candidate_utility_positive_rate": (
            candidate_utility_target.float().mean()
        ),
        "candidate_utility_recall": candidate_utility_recall,
        "candidate_utility_false_positive_rate": (
            candidate_utility_false_positive_rate
        ),
        "predicted_gain_positive_rate": predicted_gain_positive.float().mean(),
        "predicted_harm_positive_rate": predicted_harm_positive.float().mean(),
    }
    total = (
        active_weights.candidate * candidate_loss
        + active_weights.residual * residual_loss
        + active_weights.continuous_location * continuous_location_loss
        + active_weights.continuous_ap30_margin
        * continuous_ap30_margin_loss
        + active_weights.mixture_nll * mixture_nll
        + active_weights.gain * gain_loss
        + active_weights.harm * harm_loss
        + active_weights.utility * utility_loss
        + active_weights.candidate_utility * candidate_utility_loss
        + active_weights.candidate_utility_rank
        * candidate_utility_rank_loss
        + active_weights.regret * regret_loss
        + active_weights.identity * identity_loss
    )
    return total, terms



