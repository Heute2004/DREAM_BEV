from __future__ import annotations

import torch
from torch import Tensor

from .evaluation import _average_precision, evaluate_single_person_predictions, object_location_similarity


STRICT_DISTANCE_THRESHOLDS_M = (0.25, 0.30, 0.40, 0.50, 1.00)
STRICT_OLS_THRESHOLDS = (0.30, 0.50, 0.70, 0.75)


def _ols_metric_key(threshold: float) -> str:
    text = f"{threshold:.2f}".rstrip("0").rstrip(".")
    return f"ap_ols_{text.replace('.', 'p')}"


def evaluate_strict_single_person_predictions(
    scores: Tensor,
    prediction_xy: Tensor,
    target_xy: Tensor,
) -> dict[str, float]:
    """Evaluate strict distance and OLS operating points on one-person frames.

    Higher OLS thresholds are stricter; OLS@0.3 is retained for compatibility
    only and must not be described as a stricter operating point than OLS@0.5.
    """

    metrics = evaluate_single_person_predictions(
        scores,
        prediction_xy,
        target_xy,
        distance_thresholds=STRICT_DISTANCE_THRESHOLDS_M,
    )
    ols = object_location_similarity(prediction_xy, target_xy)
    metrics.update(
        {
            _ols_metric_key(threshold): _average_precision(scores, ols >= threshold)
            for threshold in STRICT_OLS_THRESHOLDS
        }
    )
    return metrics



