from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.data import RadarWindowDataset
from radar_bev.protocol import DEV_SPLIT, TRAIN_SPLIT, validate_training_manifest
from radar_bev.losses import DistilledQualityAwareBEVLoss, QualityAwareBEVLoss
from radar_bev.models import build_model
from radar_bev.screening_status import write_converged_early_stop_record
from radar_bev.teacher_cache import TeacherCacheDataset
from radar_bev.training import (
    TrainingConfig,
    evaluate,
    has_converged_early_stop,
    seed_everything,
    train_one_epoch,
    write_run_metadata,
)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train the DREAM-BEV radar anchor")
    parser.add_argument(
        "--model",
        required=True,
        choices=("dream_bev", "erodnet_core", "trodnet_core"),
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eligibility", choices=("first", "later"), default="later")
    parser.add_argument("--coordinate-weight", type=float, default=0.05)
    parser.add_argument("--loss-variant", choices=QualityAwareBEVLoss.VARIANTS, default="quality_aware_focal_coordinate")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--progress-every-batches", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--ram-cache", action="store_true")
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--selection-metric", choices=("ap_0p3m", "ap_0p5m"), default="ap_0p5m")
    parser.add_argument("--no-amp", action="store_true")
    values = vars(parser.parse_args())
    values["amp"] = not values.pop("no_amp")
    if values["early_stop_patience"] < 0:
        parser.error("--early-stop-patience must be non-negative")
    return TrainingConfig(**values)


def main() -> None:
    config = parse_args()
    validate_training_manifest(Path(config.manifest), scope="base")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)
    train_set = RadarWindowDataset(
        Path(config.data_root), Path(config.manifest), TRAIN_SPLIT, config.clip_length, config.eligibility,
        cache_in_memory=config.ram_cache,
    )
    if config.teacher_cache is not None:
        train_set = TeacherCacheDataset(train_set, Path(config.teacher_cache), TRAIN_SPLIT)
    eval_set = RadarWindowDataset(
        Path(config.data_root), Path(config.manifest), DEV_SPLIT, config.clip_length, config.eligibility,
        cache_in_memory=config.ram_cache,
    )
    loader_arguments = {
        "batch_size": config.batch_size,
        # Windows worker spawn duplicates a RAM cache. Cache users run in-process.
        "num_workers": 0 if config.ram_cache else config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": (not config.ram_cache) and config.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_arguments)
    eval_loader = DataLoader(eval_set, shuffle=False, **loader_arguments)
    model = build_model(config.model).to(device)
    criterion_type = DistilledQualityAwareBEVLoss if config.teacher_cache is not None else QualityAwareBEVLoss
    criterion = criterion_type(
        coordinate_weight=config.coordinate_weight,
        variant=config.loss_variant,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epochs)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(config, sum(parameter.numel() for parameter in model.parameters()))

    history: list[dict[str, float | int]] = []
    best = {config.selection_metric: -1.0}
    for epoch in range(1, config.train_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            config.amp,
            config.max_train_batches or None,
            config.gradient_accumulation,
            config.progress_every_batches,
        )
        eval_metrics = evaluate(model, eval_loader, criterion, device, config.max_eval_batches or None)
        scheduler.step()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"eval_{k}": v for k, v in eval_metrics.items()}}
        history.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
        if eval_metrics[config.selection_metric] > best[config.selection_metric]:
            best = {"epoch": epoch, **eval_metrics}
            torch.save({"model": config.model, "epoch": epoch, "model_state": model.state_dict(), "metrics": best}, output_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_dir / "best_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        ap_history = [float(item[f"eval_{config.selection_metric}"]) for item in history]
        if epoch < config.train_epochs and has_converged_early_stop(ap_history, config.early_stop_patience):
            if config.selection_metric == "ap_0p5m":
                marker = write_converged_early_stop_record(
                    output_dir,
                    vars(config),
                    "validated_three_completed_epochs_without_ap_0p5m_improvement",
                )
                print(json.dumps({"early_stop": str(marker), "best": best}, ensure_ascii=True), flush=True)
            else:
                completion = {
                    "completed": True,
                    "early_stopped": True,
                    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "requested_epochs": config.train_epochs,
                    "completed_epochs": len(history),
                    "best_epoch": best["epoch"],
                    "checkpoint_selection_metric": config.selection_metric,
                }
                (output_dir / "training_complete.json").write_text(
                    json.dumps(completion, indent=2), encoding="utf-8"
                )
                print(json.dumps({"early_stop": completion, "best": best}, ensure_ascii=True), flush=True)
            return
    completion = {
        "completed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_epochs": config.train_epochs,
        "completed_epochs": len(history),
        "best_epoch": best["epoch"],
        "checkpoint_selection_metric": config.selection_metric,
    }
    (output_dir / "training_complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps({"best": best}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()



