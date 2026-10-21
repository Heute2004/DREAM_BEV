#!/usr/bin/env python3
"""Pretrain and adapt the paper RV-BearingNet under the frozen CRUW protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from cruw_pedestrian_fusion.allscene import build_all_scene_index
from cruw_pedestrian_fusion.allscene_visual import AllSceneRGBDataset, visual_checkpoint_score
from cruw_pedestrian_fusion.protocol import load_protocol
from cruw_pedestrian_fusion.rv_bearing_net.model import RVBearingNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pretrain", "adapt"), required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument(
        "--image-height",
        type=int,
        help="Optional input height; image-size remains the input width.",
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=("independent", "clip_consistent", "none"),
        default="independent",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("bearing_mae", "lateral_mae", "lateral_p90"),
        default="bearing_mae",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lateral-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-sigma-deg", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def multilabel_focal(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    positive = target.eq(1.0)
    negative = target.lt(1.0)
    negative_weight = (1.0 - target).pow(4)
    positive_loss = -(probability.clamp_min(1e-6).log()) * (1.0 - probability).pow(2) * positive
    negative_loss = -((1.0 - probability).clamp_min(1e-6).log()) * probability.pow(2) * negative_weight * negative
    positive_count = positive.sum().clamp_min(1)
    return (positive_loss.sum() + negative_loss.sum()) / positive_count


def model_loss(
    output,
    batch: dict[str, torch.Tensor],
    stage: str,
    *,
    lateral_loss_weight: float = 1.0,
) -> torch.Tensor:
    dense = batch["person_u_target"].to(output.person_u_logits.device, non_blocking=True)
    loss = multilabel_focal(output.person_u_logits, dense)
    if stage == "pretrain":
        return loss
    single = batch["valid_bearing_count"].to(output.theta.device) == 1
    if not single.any():
        return loss
    bearing_target = batch["bearing_target"].to(output.bearing_logits.device, non_blocking=True)
    distribution = -(bearing_target[single] * F.log_softmax(output.bearing_logits[single], dim=1)).sum(dim=1).mean()
    theta_target = batch["theta_rad"].to(output.theta.device, non_blocking=True)[single]
    predicted_theta = output.theta[:, 0][single]
    theta_error = torch.atan2(
        torch.sin(predicted_theta - theta_target),
        torch.cos(predicted_theta - theta_target),
    )
    theta_huber = F.smooth_l1_loss(
        theta_error, torch.zeros_like(theta_error), beta=math.radians(0.5)
    )
    y_m = batch["y_m"].to(output.theta.device, non_blocking=True)[single]
    x_m = batch["x_m"].to(output.theta.device, non_blocking=True)[single]
    predicted_x = y_m * torch.tan(predicted_theta)
    x_huber = F.smooth_l1_loss(predicted_x, x_m, beta=0.10)
    sigma = output.sigma_theta[:, 0][single].clamp_min(1e-4)
    nll = (0.5 * (theta_error / sigma).square() + sigma.log()).mean()
    reliability_target = (predicted_x.detach() - x_m).abs().le(0.30).float()
    reliability = F.binary_cross_entropy(
        output.reliability[:, 0][single].clamp(1e-5, 1 - 1e-5), reliability_target
    )
    return (
        loss
        + distribution
        + theta_huber
        + lateral_loss_weight * x_huber
        + 0.10 * nll
        + 0.25 * reliability
    )


@torch.no_grad()
def evaluate(
    model: RVBearingNet,
    loader: DataLoader,
    stage: str,
    device: torch.device,
    lateral_loss_weight: float = 1.0,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    angular_errors: list[np.ndarray] = []
    x_errors: list[np.ndarray] = []
    presence_scores: list[np.ndarray] = []
    presence_labels: list[np.ndarray] = []
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        output = model(images, frame_valid=batch["frame_valid"].to(device, non_blocking=True))
        losses.append(
            float(
                model_loss(
                    output,
                    batch,
                    stage,
                    lateral_loss_weight=lateral_loss_weight,
                ).item()
            )
        )
        counts = batch["valid_bearing_count"].numpy()
        presence_scores.append(torch.sigmoid(output.person_u_logits).amax(dim=1).cpu().numpy())
        presence_labels.append((counts > 0).astype(np.int64))
        single = counts == 1
        if single.any():
            theta = output.theta[:, 0].cpu().numpy()[single]
            target_theta = batch["theta_rad"].numpy()[single]
            error = np.abs(np.arctan2(np.sin(theta - target_theta), np.cos(theta - target_theta)))
            angular_errors.append(np.degrees(error))
            predicted_x = batch["y_m"].numpy()[single] * np.tan(theta)
            x_errors.append(np.abs(predicted_x - batch["x_m"].numpy()[single]))
    scores = np.concatenate(presence_scores)
    labels = np.concatenate(presence_labels)
    output_metrics: dict[str, float] = {
        "loss": float(np.mean(losses)),
        "frames": float(labels.size),
        "presence_accuracy_at_0.5": float(np.mean((scores >= 0.5) == labels)),
    }
    if angular_errors:
        angle = np.concatenate(angular_errors)
        lateral = np.concatenate(x_errors)
        output_metrics.update(
            {
                "single_frames": float(angle.size),
                "bearing_mae_deg": float(angle.mean()),
                "bearing_p90_deg": float(np.quantile(angle, 0.90)),
                "lateral_mae_m": float(lateral.mean()),
                "lateral_p90_m": float(np.quantile(lateral, 0.90)),
            }
        )
    return output_metrics


def select_records(records, allowed_sequences: set[int], stage: str):
    selected = [record for record in records if record.sequence_id in allowed_sequences]
    if stage == "adapt":
        selected = [record for record in selected if record.pedestrian_count == 1]
    return selected


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda")
    protocol = load_protocol(args.protocol)
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    fit_all = build_all_scene_index(args.data_root, protocol, "fit", args.sample_stride)
    development_all = build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride)
    if args.stage == "pretrain":
        fit_allowed = set(protocol.fit_sequences)
        development_allowed = set(protocol.validation_sequences)
    else:
        fit_allowed = {int(value) for value in payload["single_person_fit_sequences"]}
        development_allowed = {int(value) for value in payload["single_person_validation_sequences"]}
    fit_records = select_records(fit_all, fit_allowed, args.stage)
    development_records = select_records(development_all, development_allowed, args.stage)
    fit_ids = {record.sequence_id for record in fit_records}
    development_ids = {record.sequence_id for record in development_records}
    if fit_ids & development_ids:
        raise RuntimeError(f"fit/development leakage: {sorted(fit_ids & development_ids)}")
    fit_dataset = AllSceneRGBDataset(
        fit_records,
        args.clip_length,
        args.image_size,
        image_height=args.image_height,
        training=True,
        augmentation_mode=args.augmentation_mode,
        target_sigma_rad=math.radians(args.target_sigma_deg),
    )
    development_dataset = AllSceneRGBDataset(
        development_records,
        args.clip_length,
        args.image_size,
        image_height=args.image_height,
        training=False,
        augmentation_mode="none",
        target_sigma_rad=math.radians(args.target_sigma_deg),
    )
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = RVBearingNet().to(device)
    if args.initial_checkpoint is not None:
        checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {
        "model": "RV-BearingNet",
        "stage": args.stage,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fit_sequences": sorted(fit_ids),
        "development_sequences": sorted(development_ids),
        "fit_frames": len(fit_records),
        "development_frames": len(development_records),
        "official_test_read": False,
        "protocol_sha256": sha256(args.protocol),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best_score = float("inf")
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in fit_loader:
            images = batch["images"].to(device, non_blocking=True)
            output = model(images, frame_valid=batch["frame_valid"].to(device, non_blocking=True))
            loss = model_loss(
                output,
                batch,
                args.stage,
                lateral_loss_weight=args.lateral_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
        scheduler.step()
        development = evaluate(
            model,
            development_loader,
            args.stage,
            device,
            args.lateral_loss_weight,
        )
        score = visual_checkpoint_score(args.stage, development, args.selection_metric)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "selection_score": score,
            "elapsed_seconds": time.time() - started,
            **development,
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if score < best_score - 1e-6:
            best_score = score
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": development, "metadata": metadata}, args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    reloaded = RVBearingNet().to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    report = {
        "best_epoch": checkpoint["epoch"],
        "independent_reload": evaluate(
            reloaded,
            development_loader,
            args.stage,
            device,
            args.lateral_loss_weight,
        ),
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
