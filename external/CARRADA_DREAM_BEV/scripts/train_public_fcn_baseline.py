#!/usr/bin/env python3
"""Train the public CARRADA FCN-8s architecture under the shared BEV contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carrada_dream_bev.data import (
    CARRADABEVGeometry,
    RadarBEVWindowDataset,
    apply_single_target_episode_manifest,
    build_all_scene_index,
    carrada_ra_grids,
    load_single_target_episode_manifest,
)
from carrada_dream_bev.metrics import decode_bev_logits, localization_metrics, rank_auroc
from carrada_dream_bev.models import categorized_frame_weights, normalized_gaussian_focal_loss, sample_weighted_gaussian_focal_loss
from carrada_dream_bev.protocol import load_protocol
from carrada_dream_bev.public_fcn_baseline import CARRADAFCN8sBEV


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--development-episode-manifest",
        type=Path,
        help="Fixed-identity episodes used for Development checkpoint selection.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--reference-effective-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--background-frame-loss-weight", type=float, default=0.02)
    parser.add_argument("--hard-negative-frame-loss-weight", type=float, default=0.75)
    parser.add_argument("--positive-frame-loss-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args(argv)


def episode_development_records(records, manifest_path: Path):
    episodes = load_single_target_episode_manifest(manifest_path, "development")
    selected = apply_single_target_episode_manifest(records, episodes)
    if not selected:
        raise RuntimeError("development episode manifest selected no synchronized records")
    return selected


def validate_frozen_contract(args: argparse.Namespace) -> None:
    if args.clip_length != 8:
        raise ValueError("the frozen comparison requires clip_length=8 and causal [t-7,...,t]")
    if args.batch_size * args.gradient_accumulation_steps != args.reference_effective_batch_size:
        raise ValueError("batch_size * gradient_accumulation_steps must equal reference_effective_batch_size")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")


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


def make_loader(dataset: RadarBEVWindowDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def build_model(public_root: Path, geometry: CARRADABEVGeometry) -> CARRADAFCN8sBEV:
    range_grid, angle_grid = carrada_ra_grids()
    return CARRADAFCN8sBEV(
        public_root=public_root,
        range_grid=range_grid,
        angle_grid=angle_grid,
        geometry=geometry,
    )


@torch.no_grad()
def evaluate(model: CARRADAFCN8sBEV, loader: DataLoader, geometry: CARRADABEVGeometry, device: torch.device) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    presence_scores: list[np.ndarray] = []
    presence_labels: list[np.ndarray] = []
    single_parts: dict[str, list[np.ndarray]] = {key: [] for key in ("x", "y", "target_x", "target_y", "score")}
    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        logits = model(radar).logits
        losses.append(float(normalized_gaussian_focal_loss(logits, target).item()))
        decoded = decode_bev_logits(logits, geometry)
        labels = batch["presence"].numpy().astype(np.int64)
        presence_scores.append(decoded["score"])
        presence_labels.append(labels)
        single = batch["target_count"].numpy() == 1
        if single.any():
            single_parts["x"].append(decoded["x_m"][single])
            single_parts["y"].append(decoded["y_m"][single])
            single_parts["score"].append(decoded["score"][single])
            single_parts["target_x"].append(batch["single_x_m"].numpy()[single])
            single_parts["target_y"].append(batch["single_y_m"].numpy()[single])
    scores = np.concatenate(presence_scores)
    labels = np.concatenate(presence_labels)
    metrics: dict[str, float] = {
        "loss": float(np.mean(losses)),
        "frame_count": float(labels.size),
        "positive_frame_count": float(labels.sum()),
        "empty_frame_count": float((labels == 0).sum()),
        "presence_auroc": rank_auroc(scores, labels),
        "empty_false_positive_rate_at_0.5": float(np.mean(scores[labels == 0] >= 0.5)) if (labels == 0).any() else float("nan"),
    }
    if single_parts["x"]:
        metrics.update({
            f"single_{key}": value
            for key, value in localization_metrics(
                np.concatenate(single_parts["x"]),
                np.concatenate(single_parts["y"]),
                np.concatenate(single_parts["target_x"]),
                np.concatenate(single_parts["target_y"]),
                np.concatenate(single_parts["score"]),
            ).items()
        })
    return metrics


def selection_score(metrics: dict[str, float]) -> float:
    ap30 = metrics.get("single_AP@0.30", 0.0)
    auroc = metrics.get("presence_auroc", 0.5)
    empty_fpr = metrics.get("empty_false_positive_rate_at_0.5", 0.0)
    return float(ap30 + 1e-3 * (0.5 if not np.isfinite(auroc) else auroc) - 1e-3 * (0.0 if not np.isfinite(empty_fpr) else empty_fpr))


def step_gradients(model: CARRADAFCN8sBEV, optimizer: torch.optim.Optimizer, accumulation_count: int, target: int) -> None:
    if accumulation_count < target:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(target / accumulation_count)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def main() -> None:
    args = parse_args()
    validate_frozen_contract(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    geometry = CARRADABEVGeometry()
    protocol = load_protocol(args.protocol)
    fit_records = build_all_scene_index(args.data_root, protocol, "fit")
    development_records = build_all_scene_index(args.data_root, protocol, "development")
    if args.development_episode_manifest is not None:
        development_records = episode_development_records(
            development_records,
            args.development_episode_manifest,
        )
    fit_ids = {record.sequence_id for record in fit_records}
    development_ids = {record.sequence_id for record in development_records}
    if fit_ids & development_ids:
        raise RuntimeError(f"fit/development leakage: {sorted(fit_ids & development_ids)}")
    fit_loader = make_loader(RadarBEVWindowDataset(fit_records, args.cache_root, args.clip_length, geometry), args.batch_size, args.workers, shuffle=True)
    development_loader = make_loader(RadarBEVWindowDataset(development_records, args.cache_root, args.clip_length, geometry), args.eval_batch_size, args.workers, shuffle=False)
    model = build_model(args.public_root, geometry).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {
        "model": "carrada_fcn8s",
        "public_source": model.public_source,
        "public_backbone_modified": False,
        "adapter": "causal evidence channel stacking + fixed parameter-free polar-to-BEV sampler and sensor-support mask",
        "input_contract": "RA+RA-MTI, causal 8-frame [t-7,...,t], stacked into 16 channels",
        "output_contract": "one-person confidence logits on fixed 200x234 metric BEV",
        "supervision_loss": "category-weighted CenterNet-style Gaussian focal loss",
        "frame_loss_weights": {"clean_background": args.background_frame_loss_weight, "car_or_cyclist_hard_negative": args.hard_negative_frame_loss_weight, "pedestrian_positive": args.positive_frame_loss_weight},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "optimizer": "AdamW",
        "effective_batch_size": args.reference_effective_batch_size,
        "source_partition_read": "fit/development only",
        "locked_test_read": False,
        "fit_sequences": sorted(fit_ids),
        "development_sequences": sorted(development_ids),
        "fit_frames": len(fit_records),
        "development_frames": len(development_records),
        "development_evaluation_contract": (
            "fixed_identity_episode_manifest"
            if args.development_episode_manifest is not None
            else "all_synchronized_development_frames"
        ),
        "development_episode_manifest_sha256": (
            sha256(args.development_episode_manifest)
            if args.development_episode_manifest is not None
            else None
        ),
        "protocol_sha256": sha256(args.protocol),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"event": "initialized", **metadata}), flush=True)
    history = args.output_dir / "history.jsonl"
    best_score, best_epoch, stale = -float("inf"), 0, 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        accumulated = optimizer_steps = 0
        for index, batch in enumerate(fit_loader):
            radar = batch["radar"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            frame_weights = categorized_frame_weights(
                batch["presence"].to(device, non_blocking=True),
                batch["distractor_count"].to(device, non_blocking=True),
                background_weight=args.background_frame_loss_weight,
                hard_negative_weight=args.hard_negative_frame_loss_weight,
                positive_weight=args.positive_frame_loss_weight,
            )
            loss = sample_weighted_gaussian_focal_loss(model(radar).logits, target, frame_weights=frame_weights)
            (loss / args.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().item()))
            accumulated += 1
            if accumulated == args.gradient_accumulation_steps or index + 1 == len(fit_loader):
                step_gradients(model, optimizer, accumulated, args.gradient_accumulation_steps)
                optimizer_steps += 1
                accumulated = 0
        scheduler.step()
        development = evaluate(model, development_loader, geometry, device)
        score = selection_score(development)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "selection_score": score, "learning_rate": float(optimizer.param_groups[0]["lr"]), "optimizer_steps": optimizer_steps, "elapsed_seconds": time.time() - started, **development}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": development, "metadata": metadata}, args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    reloaded = build_model(args.public_root, geometry).to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    result = {
        "model": "carrada_fcn8s",
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "independent_reload": evaluate(reloaded, development_loader, geometry, device),
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "locked_test_read": False,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
