from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


def condition_groups(frame_record: Mapping[str, str], ledger_record: Mapping[str, str]) -> set[str]:
    """Return auditable condition groups without conflating camera and radar failure."""

    groups = {"overall"}
    camera_environment = ledger_record.get("相机环境条件", "")
    weather = ledger_record.get("天气", "")
    radar_occlusion = ledger_record.get("雷达遮挡状态集合", "")
    visual_status = frame_record.get("visual_evidence_status", "")
    try:
        person_count = int(frame_record.get("person_count", "0"))
    except ValueError as error:
        raise ValueError("person_count must be an integer string") from error

    if "夜" in camera_environment and "雨" in weather:
        groups.add("night_rain")
    if any(keyword in camera_environment for keyword in ("低照度", "低光", "视觉退化", "黑夜")):
        groups.add("low_light")
    if person_count == 0 or visual_status != "available":
        groups.add("camera_target_unavailable")
    else:
        groups.add("camera_target_available")
    if "完全遮挡" in radar_occlusion:
        groups.add("radar_full_occlusion")
    return groups


def _binary_auroc(scores: Tensor, outcomes: Tensor) -> float | None:
    positives = int(outcomes.sum())
    negatives = int(outcomes.numel() - positives)
    if positives == 0 or negatives == 0:
        return None
    ordered_scores, _ = torch.sort(scores)
    unique_scores, counts = torch.unique_consecutive(ordered_scores, return_counts=True)
    ranks_by_score: dict[float, float] = {}
    first_rank = 1
    for score, count in zip(unique_scores.tolist(), counts.tolist(), strict=True):
        last_rank = first_rank + count - 1
        ranks_by_score[float(score)] = (first_rank + last_rank) * 0.5
        first_rank = last_rank + 1
    positive_rank_sum = sum(ranks_by_score[float(score)] for score in scores[outcomes].tolist())
    return float((positive_rank_sum - positives * (positives + 1) * 0.5) / (positives * negatives))


def _expected_calibration_error(scores: Tensor, outcomes: Tensor, bins: int) -> float:
    total = float(scores.numel())
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        in_bin = (scores >= lower) & (scores <= upper if index == bins - 1 else scores < upper)
        if not in_bin.any():
            continue
        confidence = float(scores[in_bin].mean())
        accuracy = float(outcomes[in_bin].float().mean())
        error += float(in_bin.sum()) / total * abs(confidence - accuracy)
    return error


def summarize_binary_quality(
    quality: Tensor,
    correct_detection: Tensor,
    *,
    calibration_bins: int = 10,
    coverage_points: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> dict[str, Any]:
    """Summarize an uncalibrated proposal-quality score against binary detection correctness."""

    if quality.shape != correct_detection.shape:
        raise ValueError("quality and correct_detection must have the same shape")
    if calibration_bins < 1:
        raise ValueError("calibration_bins must be positive")
    quality = quality.detach().flatten().to(dtype=torch.float64, device="cpu").clamp(0.0, 1.0)
    correct_detection = correct_detection.detach().flatten().to(dtype=torch.bool, device="cpu")
    if not quality.numel():
        return {
            "samples": 0,
            "positives": 0,
            "positive_rate": None,
            "quality_mean": None,
            "quality_p10": None,
            "quality_p50": None,
            "quality_p90": None,
            "auroc": None,
            "brier": None,
            "ece": None,
            "coverage_risk": [],
        }

    quality_order = torch.argsort(quality, descending=True)
    coverage_risk: list[dict[str, float]] = []
    for coverage in coverage_points:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage points must lie in (0, 1]")
        kept = max(1, math.ceil(float(coverage) * quality.numel()))
        risk = 1.0 - float(correct_detection[quality_order[:kept]].float().mean())
        coverage_risk.append({"coverage": float(coverage), "risk": risk})

    outcome_float = correct_detection.to(dtype=quality.dtype)
    return {
        "samples": int(quality.numel()),
        "positives": int(correct_detection.sum()),
        "positive_rate": float(outcome_float.mean()),
        "quality_mean": float(quality.mean()),
        "quality_p10": float(torch.quantile(quality, 0.10)),
        "quality_p50": float(torch.quantile(quality, 0.50)),
        "quality_p90": float(torch.quantile(quality, 0.90)),
        "auroc": _binary_auroc(quality, correct_detection),
        "brier": float(((quality - outcome_float).square()).mean()),
        "ece": _expected_calibration_error(quality, correct_detection, calibration_bins),
        "coverage_risk": coverage_risk,
    }



