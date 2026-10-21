"""Shared single-target localization decoding and evaluation."""

from __future__ import annotations

import numpy as np
import torch

from .allscene import CRUWBEVGeometry


def decode_ra_logits(
    logits: torch.Tensor,
    range_grid: np.ndarray,
    angle_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decode one confidence-map peak per frame."""

    confidence = torch.sigmoid(logits.detach()).float().cpu().numpy()
    if confidence.ndim != 4 or confidence.shape[1] != 1:
        raise ValueError(f"expected [B,1,H,W] logits, got {confidence.shape}")
    batch_size, _, height, width = confidence.shape
    flat = confidence.reshape(batch_size, height * width)
    peak = flat.argmax(axis=1)
    rows = peak // width
    columns = peak % width
    range_m = np.asarray(range_grid, dtype=np.float32)[rows]
    angle_rad = np.asarray(angle_grid, dtype=np.float32)[columns]
    return {
        "range_m": range_m,
        "angle_rad": angle_rad,
        "x_m": range_m * np.sin(angle_rad),
        "y_m": range_m * np.cos(angle_rad),
        "score": flat[np.arange(batch_size), peak],
    }


def decode_bev_logits(logits: torch.Tensor, geometry: CRUWBEVGeometry) -> dict[str, np.ndarray]:
    """Decode one metric BEV peak per frame under the canonical ``[x, y]`` layout."""

    confidence = torch.sigmoid(logits.detach()).float().cpu().numpy()
    if confidence.ndim != 4 or confidence.shape[1:] != (1, geometry.x_bins, geometry.y_bins):
        raise ValueError(f"unexpected canonical BEV logits shape: {confidence.shape}")
    batch_size = confidence.shape[0]
    flat = confidence.reshape(batch_size, -1)
    peak = flat.argmax(axis=1)
    x_index = peak // geometry.y_bins
    y_index = peak % geometry.y_bins
    return {
        "x_m": geometry.x_min_m + x_index.astype(np.float32) * geometry.resolution_m,
        "y_m": geometry.y_min_m + y_index.astype(np.float32) * geometry.resolution_m,
        "score": flat[np.arange(batch_size), peak],
    }


def rank_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary AUROC from average ranks, including tied scores."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    positive = targets == 1
    negative = targets == 0
    if values.ndim != 1 or targets.shape != values.shape:
        raise ValueError("scores and labels must be aligned one-dimensional arrays")
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = ranks[positive].sum()
    positives = int(positive.sum())
    negatives = int(negative.sum())
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _interpolated_ap(matches: np.ndarray, scores: np.ndarray) -> float:
    if matches.size == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    true_positive = matches[order].astype(np.float64)
    false_positive = 1.0 - true_positive
    recall = np.cumsum(true_positive) / float(matches.size)
    precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive + false_positive), 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def localization_metrics(
    prediction_x: np.ndarray,
    prediction_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Evaluate one prediction per valid unique-pedestrian frame."""

    errors = np.hypot(np.asarray(prediction_x) - np.asarray(target_x), np.asarray(prediction_y) - np.asarray(target_y))
    output = {
        "frame_count": float(errors.size),
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "P90_error_m": float(np.quantile(errors, 0.90)),
    }
    for threshold in (0.25, 0.30, 0.50, 0.75, 1.00):
        output[f"AP@{threshold:.2f}"] = _interpolated_ap(errors <= threshold, np.asarray(scores))
    return output
