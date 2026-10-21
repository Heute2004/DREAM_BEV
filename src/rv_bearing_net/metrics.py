from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from .losses import wrap_angle


def _masked_mean(values: Tensor, mask: Tensor) -> float:
    selected = values[mask]
    return float(selected.mean().item()) if selected.numel() else float("nan")


def _masked_quantile(values: Tensor, mask: Tensor, quantile: float) -> float:
    selected = values[mask]
    return (
        float(torch.quantile(selected.float(), quantile).item())
        if selected.numel()
        else float("nan")
    )


def bearing_metrics(
    *,
    predicted_theta: Tensor,
    target_theta: Tensor,
    radar_y_m: Tensor,
    target_x_m: Tensor,
    eligible: Tensor,
    reliability: Tensor,
    visual_condition_eligible: Tensor,
) -> dict[str, float]:
    shape = predicted_theta.shape
    for name, value in {
        "target_theta": target_theta,
        "radar_y_m": radar_y_m,
        "target_x_m": target_x_m,
        "eligible": eligible,
        "reliability": reliability,
        "visual_condition_eligible": visual_condition_eligible,
    }.items():
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {tuple(shape)}")
    finite = (
        torch.isfinite(predicted_theta)
        & torch.isfinite(target_theta)
        & torch.isfinite(radar_y_m)
        & torch.isfinite(target_x_m)
        & torch.isfinite(reliability)
    )
    evaluation_mask = eligible.bool() & finite
    angular_error_deg = (
        wrap_angle(predicted_theta - target_theta).abs()
        * (180.0 / math.pi)
    )
    predicted_x = radar_y_m * torch.tan(predicted_theta)
    x_error = (predicted_x - target_x_m).abs()
    eligible_reliability = visual_condition_eligible.bool() & finite
    ineligible_reliability = (~visual_condition_eligible.bool()) & finite
    return {
        "samples": float(predicted_theta.numel()),
        "eligible_samples": float(evaluation_mask.sum().item()),
        "bearing_mae_deg": _masked_mean(
            angular_error_deg,
            evaluation_mask,
        ),
        "bearing_p90_deg": _masked_quantile(
            angular_error_deg,
            evaluation_mask,
            0.90,
        ),
        "x_mae_m": _masked_mean(x_error, evaluation_mask),
        "x_p90_m": _masked_quantile(x_error, evaluation_mask, 0.90),
        "eligible_reliability_mean": _masked_mean(
            reliability,
            eligible_reliability,
        ),
        "ineligible_reliability_mean": _masked_mean(
            reliability,
            ineligible_reliability,
        ),
        "non_finite_outputs": float((~finite).sum().item()),
    }


def visual_checkpoint_decision(
    metrics: Mapping[str, float],
) -> dict[str, object]:
    failures: list[str] = []
    maximums = {
        "bearing_mae_deg": 1.0,
        "bearing_p90_deg": 2.0,
        "x_mae_m": 0.15,
        "x_p90_m": 0.30,
        "non_finite_outputs": 0.0,
    }
    for name, maximum in maximums.items():
        value = float(metrics.get(name, float("nan")))
        if not math.isfinite(value) or value > maximum:
            failures.append(name)
    eligible_mean = float(
        metrics.get("eligible_reliability_mean", float("nan"))
    )
    ineligible_mean = float(
        metrics.get("ineligible_reliability_mean", float("nan"))
    )
    if (
        not math.isfinite(eligible_mean)
        or not math.isfinite(ineligible_mean)
        or eligible_mean <= ineligible_mean
    ):
        failures.append("reliability_separation")
    return {
        "accepted": not failures,
        "failures": failures,
        "thresholds": {
            **maximums,
            "reliability_requirement": (
                "eligible_reliability_mean > "
                "ineligible_reliability_mean"
            ),
        },
    }




