"""Train one source-pinned public radar backbone under the common DREAM-BEV contract.

The command intentionally constructs only the canonical Train and Development
datasets.  A locked-test split is rejected at the protocol layer and is never
selected by this entry point.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from radar_bev.data import RadarWindowDataset
from radar_bev.losses import QualityAwareBEVLoss
from radar_bev.protocol import DEV_SPLIT, TRAIN_SPLIT, validate_training_manifest
from radar_bev.public_full_adapted import (
    PUBLIC_FULL_SPECS,
    PublicFullPolarBEVAdapter,
    public_source_metadata,
)
from radar_bev.training import (
    TrainingConfig,
    evaluate,
    has_converged_early_stop,
    manifest_sha256,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an unmodified public radar body under the shared causal BEV contract."
    )
    parser.add_argument("--model", required=True, choices=tuple(PUBLIC_FULL_SPECS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-root", default=str(REPOSITORY_ROOT / "third_party"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--coordinate-weight", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--progress-every-batches", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--ram-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    if args.gradient_accumulation < 1:
        parser.error("--gradient-accumulation must be positive")
    return args


def _training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        model=args.model,
        data_root=args.data_root,
        manifest=args.manifest,
        output_dir=args.output_dir,
        train_epochs=args.train_epochs,
        batch_size=args.batch_size,
        clip_length=16,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        eligibility="later",
        coordinate_weight=args.coordinate_weight,
        loss_variant="quality_aware_focal_coordinate",
        gradient_accumulation=args.gradient_accumulation,
        device=args.device,
        amp=not args.no_amp,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        ram_cache=args.ram_cache,
        progress_every_batches=args.progress_every_batches,
        early_stop_patience=args.early_stop_patience,
    )


def _write_metadata(
    output_dir: Path,
    config: TrainingConfig,
    public_source: dict[str, object],
    parameter_count: int,
    split_counts: dict[str, int],
) -> None:
    payload = {
        "training_config": asdict(config),
        "public_source": public_source,
        "manifest_sha256": manifest_sha256(config.manifest),
        "validated_sequence_splits": split_counts,
        "datasets_constructed": [TRAIN_SPLIT, DEV_SPLIT],
        "locked_test_read": False,
        "checkpoint_selection_metric": "Development Val AP@0.5 m",
        "parameter_count": parameter_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = _training_config(args)
    manifest = Path(config.manifest)
    split_counts = validate_training_manifest(manifest, scope="base")
    public_source = public_source_metadata(args.model, args.source_root)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    seed_everything(config.seed)

    train_set = RadarWindowDataset(
        Path(config.data_root), manifest, TRAIN_SPLIT, clip_length=16, eligibility=config.eligibility,
        cache_in_memory=config.ram_cache,
    )
    development_set = RadarWindowDataset(
        Path(config.data_root), manifest, DEV_SPLIT, clip_length=16, eligibility=config.eligibility,
        cache_in_memory=config.ram_cache,
    )
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": 0 if config.ram_cache else config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": (not config.ram_cache) and config.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    development_loader = DataLoader(development_set, shuffle=False, **loader_options)
    model = PublicFullPolarBEVAdapter(args.model, args.source_root).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    _write_metadata(output_dir, config, public_source, parameter_count, split_counts)

    criterion = QualityAwareBEVLoss(
        coordinate_weight=config.coordinate_weight,
        variant=config.loss_variant,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epochs)

    history: list[dict[str, float | int]] = []
    best: dict[str, float | int] = {"ap_0p5m": -1.0}
    for epoch in range(1, config.train_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config.amp,
            config.max_train_batches or None, config.gradient_accumulation, config.progress_every_batches,
        )
        development_metrics = evaluate(
            model, development_loader, criterion, device, config.max_eval_batches or None
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"development_{key}": value for key, value in development_metrics.items()},
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
        if float(development_metrics["ap_0p5m"]) > float(best["ap_0p5m"]):
            best = {"epoch": epoch, **development_metrics}
            torch.save(
                {
                    "model": args.model,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "metrics": best,
                    "public_source": public_source,
                    "manifest_sha256": manifest_sha256(manifest),
                    "locked_test_read": False,
                },
                output_dir / "best.pt",
            )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_dir / "best_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        ap_history = [float(item["development_ap_0p5m"]) for item in history]
        if epoch < config.train_epochs and has_converged_early_stop(ap_history, config.early_stop_patience):
            break

    completion = {
        "completed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_epochs": config.train_epochs,
        "completed_epochs": len(history),
        "best_epoch": best.get("epoch"),
        "best_development_metrics": best,
        "locked_test_read": False,
    }
    (output_dir / "training_complete.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps({"best_development": best, "locked_test_read": False}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
