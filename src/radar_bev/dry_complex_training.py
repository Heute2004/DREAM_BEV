from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as functional


@dataclass(frozen=True)
class CandidateRankingLossOutput:
    loss: Tensor
    correction_count: Tensor
    identity_count: Tensor


@dataclass(frozen=True)
class PBMHBRParameters:
    candidate_limit: int = 6
    max_step_m: float = 0.40
    base_bonus_m: float = 0.20
    polar_threshold_intercept_m: float = 0.575
    polar_threshold_range_slope_m: float = 0.25
    polar_interpolation: float = 0.20

    def __post_init__(self) -> None:
        if self.candidate_limit < 6:
            raise ValueError("PB-MHBR requires at least six candidates")
        if self.max_step_m <= 0.0 or self.base_bonus_m < 0.0:
            raise ValueError("PB-MHBR step and bonus parameters are invalid")
        if self.polar_threshold_intercept_m < 0.0:
            raise ValueError("PB-MHBR polar threshold must be non-negative")
        if self.polar_threshold_range_slope_m < 0.0:
            raise ValueError("PB-MHBR range slope must be non-negative")
        if not 0.0 <= self.polar_interpolation <= 1.0:
            raise ValueError("PB-MHBR interpolation must lie in [0,1]")


@dataclass(frozen=True)
class PBMHBRFrameResult:
    refined_xy_m: Tensor
    tracked_xy_m: Tensor
    selected_index: Tensor
    refinement_applied: Tensor


class PBMHBRStreamingRefiner:
    """Stateful causal PB-MHBR inference for ordered sequence frames."""

    def __init__(self, parameters: PBMHBRParameters | None = None) -> None:
        self.parameters = parameters or PBMHBRParameters()
        self._state: dict[str, tuple[int, Tensor]] = {}

    def reset(self, sequence: str | None = None) -> None:
        if sequence is None:
            self._state.clear()
        else:
            self._state.pop(str(sequence), None)

    def step(
        self,
        sequence: str,
        frame_id: int,
        candidate_xy_m: Tensor,
    ) -> PBMHBRFrameResult:
        parameters = self.parameters
        if candidate_xy_m.ndim != 2 or candidate_xy_m.shape != (parameters.candidate_limit, 2):
            raise ValueError(
                "candidate_xy_m must have shape "
                f"[{parameters.candidate_limit},2]"
            )
        key = str(sequence)
        state = self._state.get(key)
        if state is None or int(frame_id) <= state[0]:
            tracked = candidate_xy_m[0]
            selected = torch.zeros((), dtype=torch.long, device=candidate_xy_m.device)
        else:
            previous = state[1].to(
                device=candidate_xy_m.device,
                dtype=candidate_xy_m.dtype,
            )
            candidates = candidate_xy_m[: parameters.candidate_limit]
            cost = (candidates - previous).norm(dim=1)
            cost[0] = cost[0] - parameters.base_bonus_m
            feasible = (candidates - previous).norm(dim=1) <= parameters.max_step_m
            if bool(feasible.any()):
                cost = cost.masked_fill(~feasible, torch.inf)
            selected = cost.argmin()
            tracked = candidates[selected]
        self._state[key] = (int(frame_id), tracked.detach().clone())

        polar_candidates = candidate_xy_m[3:6]
        polar_distance = (polar_candidates - tracked).norm(dim=1)
        nearest_relative = polar_distance.argmin()
        nearest_polar = polar_candidates[nearest_relative]
        threshold_m = (
            parameters.polar_threshold_intercept_m
            + parameters.polar_threshold_range_slope_m * (tracked.norm() / 20.0)
        )
        refinement_applied = polar_distance[nearest_relative] <= threshold_m
        refined = tracked.clone()
        if bool(refinement_applied):
            refined = tracked + parameters.polar_interpolation * (
                nearest_polar - tracked
            )
        return PBMHBRFrameResult(
            refined_xy_m=refined,
            tracked_xy_m=tracked,
            selected_index=selected,
            refinement_applied=refinement_applied,
        )


def causal_nearest_candidate_track(
    candidate_xy_m: Tensor,
    *,
    candidate_limit: int | None = None,
    velocity_weight: float = 0.5,
    velocity_ema: float = 0.5,
    max_step_m: float = 1.0,
    base_bonus_m: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Track one causal candidate branch using only current and past frames."""
    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[-1] != 2:
        raise ValueError("candidate_xy_m must have shape [time,candidates,2]")
    if not 0.0 <= velocity_weight <= 1.0 or not 0.0 <= velocity_ema <= 1.0:
        raise ValueError("velocity weights must lie in [0,1]")
    if max_step_m <= 0.0 or base_bonus_m < 0.0:
        raise ValueError("max_step_m must be positive and base_bonus_m non-negative")
    time, count, _ = candidate_xy_m.shape
    limit = count if candidate_limit is None else min(int(candidate_limit), count)
    if time == 0 or limit <= 0:
        raise ValueError("candidate sequence must be non-empty")
    selected_index = torch.zeros(time, dtype=torch.long, device=candidate_xy_m.device)
    track = torch.empty(time, 2, dtype=candidate_xy_m.dtype, device=candidate_xy_m.device)
    track[0] = candidate_xy_m[0, 0]
    velocity = torch.zeros(2, dtype=candidate_xy_m.dtype, device=candidate_xy_m.device)
    for frame in range(1, time):
        previous = track[frame - 1]
        prediction = previous + velocity_weight * velocity
        candidates = candidate_xy_m[frame, :limit]
        cost = (candidates - prediction).norm(dim=1)
        cost[0] = cost[0] - base_bonus_m
        step = (candidates - previous).norm(dim=1)
        feasible = step <= max_step_m
        if bool(feasible.any()):
            cost = cost.masked_fill(~feasible, torch.inf)
        index = cost.argmin()
        current = candidates[index]
        observed_velocity = current - previous
        velocity = velocity_ema * velocity + (1.0 - velocity_ema) * observed_velocity
        track[frame] = current
        selected_index[frame] = index
    return track, selected_index


def pbmhbr_locked_sequence_refinement(
    candidate_xy_m: Tensor,
    *,
    parameters: PBMHBRParameters | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply the locked dry-complex PB-MHBR causal inference contract."""
    parameters = parameters or PBMHBRParameters()
    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[1] < parameters.candidate_limit:
        raise ValueError("PB-MHBR requires [time,at-least-6,2] candidates")
    tracked, selected_index = causal_nearest_candidate_track(
        candidate_xy_m,
        candidate_limit=parameters.candidate_limit,
        velocity_weight=0.0,
        velocity_ema=0.0,
        max_step_m=parameters.max_step_m,
        base_bonus_m=parameters.base_bonus_m,
    )
    polar_candidates = candidate_xy_m[:, 3:6]
    polar_distance = (polar_candidates - tracked[:, None]).norm(dim=-1)
    nearest_relative = polar_distance.argmin(dim=1)
    row = torch.arange(candidate_xy_m.shape[0], device=candidate_xy_m.device)
    nearest_polar = polar_candidates[row, nearest_relative]
    nearest_distance = polar_distance[row, nearest_relative]
    threshold_m = (
        parameters.polar_threshold_intercept_m
        + parameters.polar_threshold_range_slope_m * (tracked.norm(dim=1) / 20.0)
    )
    refinement_mask = nearest_distance <= threshold_m
    refined = tracked.clone()
    refined[refinement_mask] = refined[refinement_mask] + parameters.polar_interpolation * (
        nearest_polar[refinement_mask] - refined[refinement_mask]
    )
    return refined, tracked, selected_index, refinement_mask


@dataclass(frozen=True)
class CorrectnessCalibrationLossOutput:
    total: Tensor
    calibration: Tensor
    ranking: Tensor
    correct_fraction: Tensor


@dataclass(frozen=True)
class SelectiveConfidenceLossOutput:
    total: Tensor
    preservation: Tensor
    suppression: Tensor
    ranking: Tensor
    correct_fraction: Tensor


def selective_confidence_preservation_loss(
    calibrated_logits: Tensor,
    base_logits: Tensor,
    correctness_target: Tensor,
    preservation_weight: float = 1.0,
    suppression_weight: float = 1.0,
    ranking_weight: float = 1.0,
) -> SelectiveConfidenceLossOutput:
    """Preserve reliable base scores and only suppress failed-frame confidence."""
    calibrated = calibrated_logits.reshape(-1).float()
    base = base_logits.reshape(-1).float()
    target = correctness_target.reshape(-1).bool()
    if calibrated.shape != base.shape or calibrated.shape != target.shape:
        raise ValueError("calibrated, base, and target tensors must align")
    if min(preservation_weight, suppression_weight, ranking_weight) < 0.0:
        raise ValueError("loss weights must be non-negative")
    positive = target
    negative = ~target
    zero = calibrated.sum() * 0.0
    preservation = (
        (calibrated[positive] - base[positive]).square().mean()
        if bool(positive.any())
        else zero
    )
    suppression = (
        functional.softplus(calibrated[negative]).mean()
        if bool(negative.any())
        else zero
    )
    ranking = (
        functional.softplus(
            calibrated[negative][:, None] - calibrated[positive][None, :]
        ).mean()
        if bool(positive.any()) and bool(negative.any())
        else zero
    )
    return SelectiveConfidenceLossOutput(
        total=(
            preservation_weight * preservation
            + suppression_weight * suppression
            + ranking_weight * ranking
        ),
        preservation=preservation.detach(),
        suppression=suppression.detach(),
        ranking=ranking.detach(),
        correct_fraction=target.float().mean().detach(),
    )


def correctness_calibration_loss(
    correctness_logits: Tensor,
    correctness_target: Tensor,
    ranking_weight: float = 1.0,
) -> CorrectnessCalibrationLossOutput:
    """Calibrate correctness and rank reliable frames above failed frames."""
    logits = correctness_logits.reshape(-1).float()
    target = correctness_target.reshape(-1).bool()
    if logits.shape != target.shape:
        raise ValueError("correctness logits and targets must contain the same items")
    if ranking_weight < 0.0:
        raise ValueError("ranking_weight must be non-negative")
    positive = target
    negative = ~target
    positive_count = positive.sum().clamp_min(1)
    negative_count = negative.sum().clamp_min(1)
    weights = torch.where(
        positive,
        0.5 / positive_count.to(logits.dtype),
        0.5 / negative_count.to(logits.dtype),
    )
    calibration = (
        functional.binary_cross_entropy_with_logits(
            logits,
            target.to(logits.dtype),
            reduction="none",
        )
        * weights
    ).sum()
    if bool(positive.any()) and bool(negative.any()):
        ranking = functional.softplus(
            logits[negative][:, None] - logits[positive][None, :]
        ).mean()
    else:
        ranking = logits.sum() * 0.0
    return CorrectnessCalibrationLossOutput(
        total=calibration + ranking_weight * ranking,
        calibration=calibration.detach(),
        ranking=ranking.detach(),
        correct_fraction=target.float().mean().detach(),
    )


def candidate_correction_mask(
    candidate_xy_m: Tensor,
    target_xy_m: Tensor,
    correction_radius_m: float = 0.30,
) -> Tensor:
    """Return frames where base fails and an observable alternative succeeds."""
    if candidate_xy_m.ndim != 3 or candidate_xy_m.shape[-1] != 2:
        raise ValueError("candidate_xy_m must have shape [batch,candidates,2]")
    if target_xy_m.shape != candidate_xy_m.shape[:1] + (2,):
        raise ValueError("target_xy_m must have shape [batch,2]")
    if candidate_xy_m.shape[1] < 2:
        return torch.zeros(candidate_xy_m.shape[0], dtype=torch.bool, device=candidate_xy_m.device)
    candidate_error = (candidate_xy_m - target_xy_m[:, None]).norm(dim=-1)
    return (candidate_error[:, 0] > correction_radius_m) & (
        candidate_error[:, 1:].amin(dim=1) <= correction_radius_m
    )


def balanced_candidate_sample_weights(
    correction_mask: Tensor,
    correction_fraction: float = 0.50,
) -> Tensor:
    """Build sampler weights with a requested expected correction fraction."""
    if correction_mask.ndim != 1 or correction_mask.dtype is not torch.bool:
        raise ValueError("correction_mask must be a one-dimensional bool tensor")
    if not 0.0 < correction_fraction < 1.0:
        raise ValueError("correction_fraction must be strictly between zero and one")
    positive_count = int(correction_mask.sum().item())
    negative_count = correction_mask.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        return torch.ones(correction_mask.numel(), dtype=torch.float32, device=correction_mask.device)
    positive_weight = correction_fraction * negative_count / ((1.0 - correction_fraction) * positive_count)
    return torch.where(
        correction_mask,
        torch.full_like(correction_mask, positive_weight, dtype=torch.float32),
        torch.ones_like(correction_mask, dtype=torch.float32),
    )


def candidate_ranking_margin_loss(
    selector_logits: Tensor,
    candidate_xy_m: Tensor,
    target_xy_m: Tensor,
    correction_radius_m: float = 0.30,
    margin: float = 0.50,
    correction_weight: float = 1.0,
) -> CandidateRankingLossOutput:
    """Rank an observable correction above base, otherwise preserve base."""
    candidate_error = (candidate_xy_m - target_xy_m[:, None]).norm(dim=-1)
    base_error = candidate_error[:, 0]
    nonbase_error, best_nonbase_relative = candidate_error[:, 1:].min(dim=1)
    correction_mask = candidate_correction_mask(candidate_xy_m, target_xy_m, correction_radius_m)
    identity_mask = ~correction_mask
    zero = selector_logits.sum() * 0.0

    if bool(correction_mask.any()):
        row = torch.arange(selector_logits.shape[0], device=selector_logits.device)[correction_mask]
        target_index = best_nonbase_relative[correction_mask] + 1
        correction_loss = functional.relu(
            margin + selector_logits[correction_mask, 0] - selector_logits[row, target_index]
        ).mean()
    else:
        correction_loss = zero

    if bool(identity_mask.any()):
        strongest_nonbase = selector_logits[identity_mask, 1:].max(dim=1).values
        identity_loss = functional.relu(
            margin + strongest_nonbase - selector_logits[identity_mask, 0]
        ).mean()
    else:
        identity_loss = zero

    return CandidateRankingLossOutput(
        loss=correction_weight * correction_loss + identity_loss,
        correction_count=correction_mask.sum(),
        identity_count=identity_mask.sum(),
    )



