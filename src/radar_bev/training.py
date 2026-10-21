from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .evaluation import evaluate_single_person_predictions, extract_single_peak
from .geometry import BEVGeometry
from .losses import QualityAwareBEVLoss
from .strict_evaluation import evaluate_strict_single_person_predictions


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    data_root: str
    manifest: str
    output_dir: str
    train_epochs: int = 20
    batch_size: int = 2
    clip_length: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 3407
    eligibility: str = "later"
    coordinate_weight: float = 0.05
    loss_variant: str = "quality_aware_focal_coordinate"
    gradient_accumulation: int = 1
    device: str = "cuda"
    amp: bool = True
    max_train_batches: int = 0
    max_eval_batches: int = 0
    ram_cache: bool = False
    progress_every_batches: int = 500
    early_stop_patience: int = 3
    teacher_cache: str | None = None
    selection_metric: str = "ap_0p5m"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def has_converged_early_stop(ap_history: list[float], patience: int) -> bool:
    """Return whether the best AP is followed by enough completed non-improving epochs."""

    if patience < 1 or len(ap_history) <= patience:
        return False
    best_epoch_index = max(range(len(ap_history)), key=ap_history.__getitem__)
    return len(ap_history) - best_epoch_index - 1 >= patience


def manifest_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_device(batch: dict[str, Tensor | str], device: torch.device) -> dict[str, Tensor | str]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: QualityAwareBEVLoss,
    device: torch.device,
    amp: bool,
    max_batches: int | None = None,
    gradient_accumulation: int = 1,
    progress_every_batches: int = 0,
) -> dict[str, float]:
    if gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    model.train()
    totals = {"loss": 0.0, "heatmap": 0.0, "coordinate": 0.0}
    samples = 0
    batches = 0
    optimizer_steps = 0
    pending_micro_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        # All three target GPUs support BF16. It avoids the first-step FP16
        # overflow seen with sparse heatmap focal gradients and needs no scaler.
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            output = model(batch["radar"])
            teacher_logits = batch.get("teacher_logits")
            teacher_peak_xy_m = batch.get("teacher_peak_xy_m")
            if teacher_logits is not None and not isinstance(teacher_logits, Tensor):
                raise TypeError("teacher_logits must be a Tensor when present")
            if teacher_peak_xy_m is not None and not isinstance(teacher_peak_xy_m, Tensor):
                raise TypeError("teacher_peak_xy_m must be a Tensor when present")
            loss = criterion(
                output["logits"],
                batch["target_xy_m"],
                batch["sample_weight"],
                teacher_logits=teacher_logits,
                teacher_peak_xy_m=teacher_peak_xy_m,
            )
        (loss.total / gradient_accumulation).backward()
        pending_micro_batches += 1
        if pending_micro_batches == gradient_accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            pending_micro_batches = 0
        count = int(batch["radar"].shape[0])
        batches += 1
        samples += count
        totals["loss"] += float(loss.total.detach().item()) * count
        totals["heatmap"] += float(loss.heatmap.detach().item()) * count
        totals["coordinate"] += float(loss.coordinate.detach().item()) * count
        if progress_every_batches and batches % progress_every_batches == 0:
            print(
                json.dumps(
                    {
                        "phase": "train",
                        "micro_batch": batches,
                        "optimizer_steps": optimizer_steps,
                        "mean_loss_so_far": totals["loss"] / max(samples, 1),
                    }
                ),
                flush=True,
            )
    if pending_micro_batches:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
    return {
        **{key: value / max(samples, 1) for key, value in totals.items()},
        "batches": float(batches),
        "optimizer_steps": float(optimizer_steps),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: QualityAwareBEVLoss,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    scores: list[Tensor] = []
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    losses: list[float] = []
    durations: list[float] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        output = model(batch["radar"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(perf_counter() - start)
        loss = criterion(output["logits"], batch["target_xy_m"], batch["sample_weight"])
        prediction_xy, confidence = extract_single_peak(output["logits"], BEVGeometry())
        scores.append(confidence.cpu())
        predictions.append(prediction_xy.cpu())
        targets.append(batch["target_xy_m"].cpu())
        losses.append(float(loss.total.item()) * int(confidence.shape[0]))
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    metric = evaluate_strict_single_person_predictions(torch.cat(scores), prediction, target)
    metric["loss"] = sum(losses) / max(prediction.shape[0], 1)
    metric["batches"] = float(len(durations))
    metric["latency_ms_per_frame"] = 1000.0 * sum(durations) / max(prediction.shape[0], 1)
    metric["fps"] = 1000.0 / max(metric["latency_ms_per_frame"], 1e-6)
    return metric


def write_run_metadata(config: TrainingConfig, parameter_count: int) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": asdict(config),
        "checkpoint_selection_metric": config.selection_metric,
        "manifest_sha256": manifest_sha256(config.manifest),
        "parameter_count": parameter_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")



