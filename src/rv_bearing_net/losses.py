from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from .contracts import RVBearingOutput


@dataclass(frozen=True)
class RVBearingLoss:
    total: Tensor
    bearing_distribution: Tensor
    theta_huber: Tensor
    x_huber: Tensor
    uncertainty_nll: Tensor
    person_u: Tensor
    reliability: Tensor
    causal_velocity: Tensor
    geometry_teacher: Tensor
    candidate_ranking: Tensor


def wrap_angle(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def gaussian_bin_target(
    target: Tensor,
    centers: Tensor,
    *,
    sigma: float,
) -> Tensor:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    distance = target.unsqueeze(-1) - centers
    values = torch.exp(-0.5 * (distance / sigma).square())
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    if values.shape != weights.shape:
        raise ValueError("values and weights must have identical shapes")
    active = weights > 0.0
    safe_values = torch.where(active, values, torch.zeros_like(values))
    safe_weights = torch.where(active, weights, torch.zeros_like(weights))
    denominator = safe_weights.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        return safe_values.sum() * 0.0
    return (safe_values * safe_weights).sum() / denominator


def _zero_from_output(current: RVBearingOutput) -> Tensor:
    return current.bearing_logits.sum() * 0.0


def _geometry_teacher_loss(
    *,
    current: RVBearingOutput,
    batch: dict[str, Tensor],
    coordinate_mask: Tensor,
    base_weight: Tensor,
) -> Tensor:
    required = (
        "geometry_theta_rad",
        "geometry_sigma_rad",
        "geometry_valid",
    )
    if any(key not in batch for key in required):
        return _zero_from_output(current)
    theta = batch["geometry_theta_rad"].float()
    sigma = batch["geometry_sigma_rad"].float()
    valid = batch["geometry_valid"].bool()
    if (
        theta.ndim != 1
        or sigma.shape != theta.shape
        or valid.shape != theta.shape
    ):
        raise ValueError("geometry teacher fields must have shape [batch]")
    finite = torch.isfinite(theta) & torch.isfinite(sigma)
    mask = coordinate_mask & valid & finite
    theta = torch.where(finite, theta, torch.zeros_like(theta))
    sigma = torch.where(
        finite,
        sigma,
        torch.full_like(sigma, math.radians(1.0)),
    )
    sigma = sigma.clamp(math.radians(0.25), math.radians(5.0))
    confidence = (math.radians(1.0) / sigma).clamp(0.25, 2.0)
    error = wrap_angle(current.theta[:, 0] - theta)
    raw = functional.smooth_l1_loss(
        error,
        torch.zeros_like(error),
        beta=math.radians(0.5),
        reduction="none",
    ) * confidence
    return _weighted_mean(
        raw,
        base_weight * mask.to(base_weight.dtype),
    )


def _sample_logits_at_angles(
    logits: Tensor,
    candidate_theta: Tensor,
    angle_centers: Tensor,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2 or candidate_theta.ndim != 2:
        raise ValueError(
            "logits and candidate angles must be two-dimensional"
        )
    if logits.shape[0] != candidate_theta.shape[0]:
        raise ValueError("candidate batch does not match logits")
    if (
        angle_centers.ndim != 1
        or angle_centers.numel() != logits.shape[1]
    ):
        raise ValueError("angle centers do not match bearing logits")
    centers = angle_centers.to(
        device=logits.device,
        dtype=logits.dtype,
    )
    if not bool(torch.all(centers[1:] > centers[:-1])):
        raise ValueError("angle centers must be strictly increasing")
    minimum = centers[0]
    maximum = centers[-1]
    scale = (centers.numel() - 1) / (maximum - minimum)
    position = (
        (candidate_theta.to(logits.dtype) - minimum) * scale
    ).clamp(0.0, float(centers.numel() - 1))
    left = position.floor().long()
    right = (left + 1).clamp_max(centers.numel() - 1)
    alpha = position - left.to(position.dtype)
    left_value = logits.gather(1, left)
    right_value = logits.gather(1, right)
    sampled = left_value + alpha * (right_value - left_value)
    inside = (
        (candidate_theta >= minimum)
        & (candidate_theta <= maximum)
    )
    return sampled, inside


def _candidate_ranking_loss(
    *,
    current: RVBearingOutput,
    batch: dict[str, Tensor],
    angle_centers: Tensor,
    coordinate_mask: Tensor,
    base_weight: Tensor,
    margin: float,
) -> Tensor:
    required = (
        "radar_candidate_theta_rad",
        "radar_candidate_support",
        "radar_candidate_valid",
        "radar_candidate_positive_index",
    )
    if any(key not in batch for key in required):
        return _zero_from_output(current)
    theta = batch["radar_candidate_theta_rad"].float()
    support = batch["radar_candidate_support"].float().detach()
    valid = batch["radar_candidate_valid"].bool()
    positive_index = batch["radar_candidate_positive_index"].long()
    finite = torch.isfinite(theta) & torch.isfinite(support)
    valid = valid & finite
    theta = torch.where(valid, theta, torch.zeros_like(theta))
    support = torch.where(valid, support, torch.zeros_like(support))
    if (
        theta.ndim != 2
        or support.shape != theta.shape
        or valid.shape != theta.shape
    ):
        raise ValueError(
            "radar candidate fields must have shape [batch,K]"
        )
    if positive_index.shape != (theta.shape[0],):
        raise ValueError(
            "candidate positive index must have shape [batch]"
        )
    if bool(
        (
            (positive_index < 0)
            | (positive_index >= theta.shape[1])
        ).any()
    ):
        raise ValueError("candidate positive index is outside K")
    scores, inside = _sample_logits_at_angles(
        current.bearing_logits,
        theta,
        angle_centers,
    )
    valid = valid & inside
    positive_score = scores.gather(
        1,
        positive_index[:, None],
    ).squeeze(1)
    positive_valid = valid.gather(
        1,
        positive_index[:, None],
    ).squeeze(1)
    candidate_ids = torch.arange(
        theta.shape[1],
        device=theta.device,
    )[None]
    negative_mask = (
        valid & (candidate_ids != positive_index[:, None])
    )
    negative_weight = (
        support.clamp_min(0.0)
        * negative_mask.to(support.dtype)
    )
    fallback_weight = negative_mask.to(support.dtype)
    negative_weight = torch.where(
        negative_weight.sum(dim=1, keepdim=True) > 0.0,
        negative_weight,
        fallback_weight,
    )
    denominator = negative_weight.sum(dim=1)
    has_negative = denominator > 0.0
    pairwise = functional.softplus(
        scores - positive_score[:, None] + margin
    )
    sample_loss = (
        pairwise * negative_weight
    ).sum(dim=1) / denominator.clamp_min(1e-8)
    sample_mask = coordinate_mask & positive_valid & has_negative
    return _weighted_mean(
        sample_loss,
        base_weight * sample_mask.to(base_weight.dtype),
    )


def compute_rv_bearing_loss(
    *,
    current: RVBearingOutput,
    previous: RVBearingOutput,
    batch: dict[str, Tensor],
    angle_centers: Tensor,
    bearing_target_sigma_rad: float = math.radians(1.0),
    person_target_sigma: float = 0.025,
    reliability_tolerance_m: float = 0.30,
    temporal_weight: float = 0.10,
    x_huber_weight: float = 1.0,
    x_huber_beta_m: float = 0.10,
    geometry_teacher_weight: float = 0.0,
    candidate_ranking_weight: float = 0.0,
    candidate_ranking_margin: float = 0.25,
) -> RVBearingLoss:
    if temporal_weight < 0.0:
        raise ValueError("temporal_weight must be non-negative")
    if x_huber_weight < 0.0:
        raise ValueError("x_huber_weight must be non-negative")
    if x_huber_beta_m <= 0.0:
        raise ValueError("x_huber_beta_m must be positive")
    if (
        geometry_teacher_weight < 0.0
        or candidate_ranking_weight < 0.0
    ):
        raise ValueError("teacher loss weights must be non-negative")
    if candidate_ranking_margin < 0.0:
        raise ValueError("candidate ranking margin must be non-negative")
    theta_history = batch["target_theta_history"]
    xy_history = batch["target_xy_history_m"]
    target_valid = batch["target_valid_history"].bool()
    person_u_history = batch["person_u_history"]
    person_valid = batch["person_valid_history"].bool()
    coordinate_weight = batch["coordinate_weight"].float()
    sample_weight = batch["sample_weight"].float()
    frame_ids = batch["history_frame_ids"]
    if theta_history.ndim != 2 or theta_history.shape[1] < 2:
        raise ValueError("target_theta_history must be [batch,time>=2]")
    batch_size = theta_history.shape[0]
    if tuple(current.theta.shape) != (batch_size, 1):
        raise ValueError("current output batch does not match targets")
    if angle_centers.ndim != 1:
        raise ValueError("angle_centers must be one-dimensional")

    target_theta = theta_history[:, -1]
    base_weight = coordinate_weight * sample_weight
    coordinate_mask = (
        target_valid[:, -1]
        & current.valid
        & (coordinate_weight > 0.0)
    )
    coordinate_weights = (
        base_weight * coordinate_mask.to(base_weight.dtype)
    )

    bearing_target = gaussian_bin_target(
        target_theta,
        angle_centers.to(
            device=target_theta.device,
            dtype=target_theta.dtype,
        ),
        sigma=bearing_target_sigma_rad,
    )
    distribution_raw = -(
        bearing_target
        * functional.log_softmax(current.bearing_logits, dim=1)
    ).sum(dim=1)
    bearing_distribution = _weighted_mean(
        distribution_raw,
        coordinate_weights,
    )

    angular_error = wrap_angle(current.theta[:, 0] - target_theta)
    theta_huber = _weighted_mean(
        functional.smooth_l1_loss(
            angular_error,
            torch.zeros_like(angular_error),
            beta=math.radians(0.5),
            reduction="none",
        ),
        coordinate_weights,
    )
    sigma = current.sigma_theta[:, 0].clamp_min(1e-4)
    uncertainty_raw = (
        0.5 * (angular_error / sigma).square()
        + sigma.log()
    )
    uncertainty_nll = _weighted_mean(
        uncertainty_raw,
        coordinate_weights,
    )

    person_target = gaussian_bin_target(
        person_u_history[:, -1],
        torch.linspace(
            0.0,
            1.0,
            angle_centers.numel(),
            device=person_u_history.device,
            dtype=person_u_history.dtype,
        ),
        sigma=person_target_sigma,
    )
    person_raw = -(
        person_target
        * functional.log_softmax(current.person_u_logits, dim=1)
    ).sum(dim=1)
    person_weights = (
        sample_weight
        * person_valid[:, -1].to(sample_weight.dtype)
        * current.valid.to(sample_weight.dtype)
    )
    person_u = _weighted_mean(person_raw, person_weights)

    target_y = xy_history[:, -1, 1].clamp_min(1e-3)
    predicted_x = target_y * torch.tan(current.theta[:, 0])
    target_x = xy_history[:, -1, 0]
    x_residual = predicted_x - target_x
    x_huber = _weighted_mean(
        functional.smooth_l1_loss(
            x_residual,
            torch.zeros_like(x_residual),
            beta=x_huber_beta_m,
            reduction="none",
        ),
        coordinate_weights,
    )
    x_error = (predicted_x - target_x).abs()
    reliability_target = (
        coordinate_mask
        & person_valid[:, -1]
        & (x_error <= reliability_tolerance_m)
    ).to(current.reliability.dtype)
    reliability_probability = current.reliability[:, 0].float().clamp(
        1e-6,
        1.0 - 1e-6,
    )
    reliability_raw = functional.binary_cross_entropy_with_logits(
        torch.logit(reliability_probability),
        reliability_target.float(),
        reduction="none",
    )
    reliability = _weighted_mean(reliability_raw, sample_weight)

    adjacent = frame_ids[:, -1] - frame_ids[:, -2] == 1
    temporal_mask = (
        coordinate_mask
        & target_valid[:, -2]
        & previous.valid
        & adjacent
    )
    predicted_delta = wrap_angle(
        current.theta[:, 0] - previous.theta[:, 0]
    )
    target_delta = wrap_angle(
        theta_history[:, -1] - theta_history[:, -2]
    )
    velocity_raw = functional.smooth_l1_loss(
        predicted_delta,
        target_delta,
        beta=math.radians(0.5),
        reduction="none",
    )
    causal_velocity = _weighted_mean(
        velocity_raw,
        base_weight * temporal_mask.to(base_weight.dtype),
    )
    geometry_teacher = _geometry_teacher_loss(
        current=current,
        batch=batch,
        coordinate_mask=coordinate_mask,
        base_weight=base_weight,
    )
    candidate_ranking = _candidate_ranking_loss(
        current=current,
        batch=batch,
        angle_centers=angle_centers,
        coordinate_mask=coordinate_mask,
        base_weight=base_weight,
        margin=candidate_ranking_margin,
    )

    total = (
        bearing_distribution
        + 0.5 * theta_huber
        + x_huber_weight * x_huber
        + 0.25 * uncertainty_nll
        + 0.25 * person_u
        + 0.20 * reliability
        + temporal_weight * causal_velocity
        + geometry_teacher_weight * geometry_teacher
        + candidate_ranking_weight * candidate_ranking
    )
    return RVBearingLoss(
        total=total,
        bearing_distribution=bearing_distribution,
        theta_huber=theta_huber,
        x_huber=x_huber,
        uncertainty_nll=uncertainty_nll,
        person_u=person_u,
        reliability=reliability,
        causal_velocity=causal_velocity,
        geometry_teacher=geometry_teacher,
        candidate_ranking=candidate_ranking,
    )



