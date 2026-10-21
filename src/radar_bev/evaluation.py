from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from .geometry import BEVGeometry


def summarize_forward_latency(durations_s: Iterable[float]) -> dict[str, float | int]:
    """Summarize batch-one model-forward timings after a fixed warm-up."""

    durations = [float(value) for value in durations_s]
    if not durations:
        return {"measured_batches": 0, "latency_ms_per_frame": 0.0, "fps": 0.0}
    if any(value <= 0.0 for value in durations):
        raise ValueError("forward durations must be positive")
    latency_ms = 1000.0 * sum(durations) / len(durations)
    return {
        "measured_batches": len(durations),
        "latency_ms_per_frame": latency_ms,
        "fps": 1000.0 / latency_ms,
    }


def extract_single_peak(logits: Tensor, geometry: BEVGeometry | None = None) -> tuple[Tensor, Tensor]:
    """Decode exactly one confidence peak per frame under the single-person contract."""

    geometry = geometry or BEVGeometry()
    if logits.ndim != 4 or logits.shape[1:] != (1, geometry.x_bins, geometry.y_bins):
        raise ValueError("logits must be [batch,1,200,234]")
    probabilities = logits[:, 0].sigmoid()
    scores, flattened = probabilities.flatten(start_dim=1).max(dim=1)
    x_index = torch.div(flattened, geometry.y_bins, rounding_mode="floor")
    y_index = flattened.remainder(geometry.y_bins)
    prediction_xy = geometry.index_to_metric(torch.stack((x_index, y_index), dim=-1)).to(device=logits.device)
    return prediction_xy, scores


def object_location_similarity(prediction_xy: Tensor, target_xy: Tensor, kappa: float = 0.15) -> Tensor:
    """RODNet/CRUW OLS: exp(-d^2 / (2 * (range * kappa)^2))."""

    error = torch.linalg.vector_norm(prediction_xy - target_xy, dim=-1)
    scale = torch.linalg.vector_norm(target_xy, dim=-1).clamp_min(0.5) * kappa
    return torch.exp(-error.square() / (2.0 * scale.square()))


def _average_precision(scores: Tensor, matched: Tensor) -> float:
    if scores.numel() == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    matched = matched[order].to(dtype=torch.float64)
    true_positive = torch.cumsum(matched, dim=0)
    false_positive = torch.cumsum(1.0 - matched, dim=0)
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / matched.numel()
    precision = torch.cat((torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)))
    recall = torch.cat((torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)))
    for index in range(precision.numel() - 1, 0, -1):
        precision[index - 1] = torch.maximum(precision[index - 1], precision[index])
    changed = recall[1:] != recall[:-1]
    return float(((recall[1:][changed] - recall[:-1][changed]) * precision[1:][changed]).sum().item())


def _normalize_distance_thresholds(distance_thresholds: Iterable[float]) -> tuple[float, ...]:
    """Validate distance thresholds and return a deterministic ascending tuple."""

    thresholds = tuple(sorted({float(value) for value in distance_thresholds}))
    if not thresholds:
        raise ValueError("at least one distance threshold is required")
    if any(not torch.isfinite(torch.tensor(value)) or value <= 0.0 for value in thresholds):
        raise ValueError("distance thresholds must be finite positive values")
    return thresholds


def _distance_metric_key(threshold_m: float) -> str:
    """Return stable machine-readable keys such as ap_0p5m for AP@0.5 m."""

    text = f"{threshold_m:.3f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return f"ap_{text.replace('.', 'p')}m"


def evaluate_single_person_predictions(
    scores: Tensor,
    prediction_xy: Tensor,
    target_xy: Tensor,
    ols_kappa: float = 0.15,
    distance_thresholds: Iterable[float] = (0.5, 1.0),
) -> dict[str, float]:
    """Return AP and localization summaries for one GT person per valid frame."""

    if scores.ndim != 1 or prediction_xy.shape != target_xy.shape or prediction_xy.shape[-1] != 2:
        raise ValueError("scores must be [N] and prediction/target must both be [N,2]")
    thresholds = _normalize_distance_thresholds(distance_thresholds)
    errors = torch.linalg.vector_norm(prediction_xy - target_xy, dim=-1)
    ols = object_location_similarity(prediction_xy, target_xy, ols_kappa)
    return {
        **{
            _distance_metric_key(threshold_m): _average_precision(scores, errors <= threshold_m)
            for threshold_m in thresholds
        },
        "ap_ols_0p5": _average_precision(scores, ols >= 0.5),
        "mean_ols": float(ols.mean().item()),
        "mean_location_error_m": float(errors.mean().item()),
        "median_location_error_m": float(errors.median().item()),
        "p90_location_error_m": float(torch.quantile(errors, 0.90).item()),
    }


def evaluate_grouped_single_person_predictions(
    scores: Tensor,
    prediction_xy: Tensor,
    target_xy: Tensor,
    groups: Iterable[str],
    ols_kappa: float = 0.15,
    distance_thresholds: Iterable[float] = (0.5, 1.0),
) -> dict[str, dict[str, float]]:
    """Evaluate the same valid frames globally and by a manifest-derived group."""

    group_values = list(groups)
    if len(group_values) != scores.numel():
        raise ValueError("groups must contain one value for every prediction")
    thresholds = _normalize_distance_thresholds(distance_thresholds)
    indices_by_group: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_values):
        indices_by_group[str(group)].append(index)
    result: dict[str, dict[str, float]] = {}
    for group, indices in sorted(indices_by_group.items()):
        index_tensor = torch.tensor(indices, device=scores.device, dtype=torch.long)
        metric = evaluate_single_person_predictions(
            scores[index_tensor],
            prediction_xy[index_tensor],
            target_xy[index_tensor],
            ols_kappa,
            thresholds,
        )
        result[group] = {"frames": float(len(indices)), **metric}
    return result



