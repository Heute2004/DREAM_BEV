from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import torch
from torch import Tensor, nn

from .evaluation import extract_single_peak, summarize_forward_latency
from .geometry import BEVGeometry
from .strict_evaluation import evaluate_strict_single_person_predictions


@dataclass(frozen=True)
class FrozenBackboneEvaluation:
    metrics: dict[str, float]
    runtime: dict[str, float | int]
    rows: list[dict[str, float | int | str]]


@torch.inference_mode()
def evaluate_frozen_backbone(
    model: nn.Module,
    loader: Iterable[dict[str, Tensor | list[str]]],
    device: torch.device,
    *,
    latency_warmup_batches: int = 20,
) -> FrozenBackboneEvaluation:
    if latency_warmup_batches < 0:
        raise ValueError("latency warm-up batches must be non-negative")
    model.eval()
    scores: list[Tensor] = []
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    rows: list[dict[str, float | int | str]] = []
    durations_s: list[float] = []
    geometry = BEVGeometry()

    for batch_index, batch in enumerate(loader):
        radar = batch["radar"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        logits = model(radar)["logits"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if batch_index >= latency_warmup_batches:
            durations_s.append(perf_counter() - start)

        prediction, score = extract_single_peak(logits, geometry)
        target = batch["target_xy_m"].to(prediction.device)
        errors = torch.linalg.vector_norm(prediction - target, dim=-1)
        prediction_cpu = prediction.cpu()
        score_cpu = score.cpu()
        target_cpu = target.cpu()
        errors_cpu = errors.cpu()
        scores.append(score_cpu)
        predictions.append(prediction_cpu)
        targets.append(target_cpu)

        sequences = batch["sequence"]
        frame_ids = batch["frame_id"]
        for index in range(prediction_cpu.shape[0]):
            rows.append(
                {
                    "sequence": str(sequences[index]),
                    "frame_id": int(frame_ids[index].item()),
                    "score": float(score_cpu[index].item()),
                    "prediction_x_m": float(prediction_cpu[index, 0].item()),
                    "prediction_y_m": float(prediction_cpu[index, 1].item()),
                    "target_x_m": float(target_cpu[index, 0].item()),
                    "target_y_m": float(target_cpu[index, 1].item()),
                    "error_m": float(errors_cpu[index].item()),
                }
            )

    score_tensor = torch.cat(scores)
    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    return FrozenBackboneEvaluation(
        metrics=evaluate_strict_single_person_predictions(
            score_tensor, prediction_tensor, target_tensor
        ),
        runtime=summarize_forward_latency(durations_s),
        rows=rows,
    )
