from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from rv_evidence_net.data import RGBClipDataset, build_sequence_balanced_sampler, load_student_sequences
from rv_evidence_net.model import RVEvidenceNetSTMamba
from rv_evidence_net.protocol import sha256
from rv_evidence_net.training import EarlyStoppingTracker, compute_batch_loss




@dataclass(frozen=True)
class TrainingContract:
    stage: str
    dataset_root: str
    output_root: str
    image_size: int
    clip_length: int
    frame_stride: int
    width: float
    token_dim: int
    temporal_model_dim: int
    temporal_state_dim: int
    temporal_layers: int
    temporal_backend: str
    top_k: int
    quality_head: bool
    train_augmentation: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    validation_modulo: int
    validation_bucket: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    train_sequences: int
    validation_sequences: int
    train_clips: int
    validation_clips: int
    teacher_distillation: str
    lockbox_protocol: str | None
    lockbox_protocol_sha256: str | None
    protocol_manifest: str | None
    protocol_manifest_sha256: str | None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RV-EvidenceNet-ST-Mamba from frozen M2 visual labels")
    parser.add_argument("--stage", choices=("spatial", "temporal", "joint"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-profile", type=Path, required=True)
    parser.add_argument("--lockbox-protocol", type=Path, default=PROJECT_ROOT / "contracts/protocol.json")
    parser.add_argument("--protocol-manifest", type=Path, default=PROJECT_ROOT / "contracts/frame_manifest.csv")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--width", type=float, default=1.25)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--temporal-model-dim", type=int, default=128)
    parser.add_argument("--temporal-state-dim", type=int, default=16)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-backend", choices=("selective_scan", "mamba_ssm"), default="selective_scan")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--quality-head", action="store_true", help="Enable IoU-aware visual reliability scoring for spatial proposals.")
    parser.add_argument("--train-augmentation", choices=("none", "mild_photometric"), default="none")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, help="Optional smoke-test cap; never use for a reported run.")
    parser.add_argument("--max-validation-batches", type=int, help="Optional smoke-test cap; never use for checkpoint selection.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--validation-modulo", type=int, default=10)
    parser.add_argument("--validation-bucket", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Zero disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def move_batch(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "images": batch["images"].to(device, non_blocking=True),  # type: ignore[union-attr]
        "boxes": batch["boxes"].to(device, non_blocking=True),  # type: ignore[union-attr]
        "valid_mask": batch["valid_mask"].to(device, non_blocking=True),  # type: ignore[union-attr]
    }


def set_stage_trainability(model: RVEvidenceNetSTMamba, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if stage != "temporal":
        return
    for module in (model.encoder, model.neck, model.heads):
        for parameter in module.parameters():
            parameter.requires_grad_(False)


def run_epoch(
    model: RVEvidenceNetSTMamba,
    loader: DataLoader[dict[str, object]],
    device: torch.device,
    stage: str,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if stage == "temporal":
        model.encoder.eval()
        model.neck.eval()
        model.heads.eval()
        model.evidence_adapter.eval()
    totals = {"total": 0.0, "detection": 0.0, "objectness": 0.0, "box": 0.0, "quality": 0.0, "recovery": 0.0, "support": 0.0, "contrastive": 0.0}
    batch_count = 0
    for raw_batch in loader:
        if max_batches is not None and batch_count >= max_batches:
            break
        batch = move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            losses = compute_batch_loss(model, batch, stage)
        if training:
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        values = {
            "total": losses.total,
            "detection": losses.detection,
            "objectness": losses.objectness,
            "box": losses.box,
            "quality": losses.quality,
            "recovery": losses.recovery,
            "support": losses.support,
            "contrastive": losses.contrastive,
        }
        for name, value in values.items():
            totals[name] += float(value.detach())
        batch_count += 1
    return {name: value / max(batch_count, 1) for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.stage == "spatial":
        args.clip_length = 1
    seed_everything(args.seed)
    device = torch.device(args.device)
    run_root = args.output_root / args.run_name
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_root}")
    run_root.mkdir(parents=True)

    sequences = load_student_sequences(
        args.dataset_root,
        manifest_path=args.teacher_manifest,
        profile_path=args.teacher_profile,
        validation_modulo=args.validation_modulo,
        validation_bucket=args.validation_bucket,
        protocol_manifest_path=args.protocol_manifest,
        protocol_path=args.lockbox_protocol,
    )
    train_dataset = RGBClipDataset(
        sequences,
        "Train",
        args.clip_length,
        args.frame_stride,
        args.image_size,
        args.validation_modulo,
        args.validation_bucket,
        train_augmentation=args.train_augmentation,
    )
    validation_dataset = RGBClipDataset(
        sequences, "InternalVal", args.clip_length, args.frame_stride, args.image_size, args.validation_modulo, args.validation_bucket
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=build_sequence_balanced_sampler(train_dataset),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = RVEvidenceNetSTMamba(
        width=args.width,
        token_dim=args.token_dim,
        temporal_model_dim=args.temporal_model_dim,
        temporal_state_dim=args.temporal_state_dim,
        temporal_layers=args.temporal_layers,
        temporal_backend=args.temporal_backend,
        top_k=args.top_k,
        quality_head=args.quality_head,
    ).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=True)
    set_stage_trainability(model, args.stage)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    contract = TrainingContract(
        stage=args.stage,
        dataset_root=str(args.dataset_root),
        output_root=str(run_root),
        image_size=args.image_size,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        width=args.width,
        token_dim=args.token_dim,
        temporal_model_dim=args.temporal_model_dim,
        temporal_state_dim=args.temporal_state_dim,
        temporal_layers=args.temporal_layers,
        temporal_backend=args.temporal_backend,
        top_k=args.top_k,
        quality_head=args.quality_head,
        train_augmentation=args.train_augmentation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        validation_modulo=args.validation_modulo,
        validation_bucket=args.validation_bucket,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        train_sequences=len({sequence.sequence_id for sequence, _ in train_dataset.samples}),
        validation_sequences=len({sequence.sequence_id for sequence, _ in validation_dataset.samples}),
        train_clips=len(train_dataset),
        validation_clips=len(validation_dataset),
        teacher_distillation="not_enabled_in_initial_training_entry",
        lockbox_protocol=(str(args.lockbox_protocol) if args.lockbox_protocol else None),
        lockbox_protocol_sha256=(sha256(args.lockbox_protocol) if args.lockbox_protocol else None),
        protocol_manifest=(str(args.protocol_manifest) if args.protocol_manifest else None),
        protocol_manifest_sha256=(sha256(args.protocol_manifest) if args.protocol_manifest else None),
    )
    (run_root / "training_contract.json").write_text(json.dumps(asdict(contract), indent=2) + "\n", encoding="utf-8")
    early_stopping = EarlyStoppingTracker(args.early_stopping_patience, args.early_stopping_min_delta)
    with (run_root / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "partition", "total", "detection", "objectness", "box", "quality", "recovery", "support", "contrastive"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, device, args.stage, optimizer, scaler, args.max_train_batches)
            with torch.no_grad():
                validation_metrics = run_epoch(
                    model, validation_loader, device, "spatial", None, scaler, args.max_validation_batches
                )
            for partition, metrics in (("train", train_metrics), ("internal_val_detection", validation_metrics)):
                record = {"epoch": epoch, "partition": partition, **metrics}
                writer.writerow(record)
                handle.flush()
                print(json.dumps(record))
            torch.save({"model": model.state_dict(), "contract": asdict(contract), "epoch": epoch}, run_root / "last.pt")
            if early_stopping.update(validation_metrics["total"]):
                torch.save({"model": model.state_dict(), "contract": asdict(contract), "epoch": epoch}, run_root / "best_internal_val.pt")
            if early_stopping.should_stop:
                print(json.dumps({"event": "early_stop", "epoch": epoch, "best_validation": early_stopping.best, "stale_epochs": early_stopping.stale_epochs}))
                break


if __name__ == "__main__":
    main()
