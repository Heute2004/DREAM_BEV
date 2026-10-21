#!/usr/bin/env python3
"""Train the Empty-Scene Sentinel from a frozen canonical DREAM-BEV checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from cruw_pedestrian_fusion.allscene import CRUWBEVGeometry, RadarBEVWindowDataset, build_all_scene_index
from cruw_pedestrian_fusion.canonical_dream_bev import DREAMBEV
from cruw_pedestrian_fusion.geometry import load_cruw_grids
from cruw_pedestrian_fusion.metrics import rank_auroc
from cruw_pedestrian_fusion.protocol import load_protocol
from cruw_pedestrian_fusion.sentinel import EmptySceneSentinel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cruw-devkit-root", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_features(anchor, dataset, batch_size, workers, device):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    sequences: list[torch.Tensor] = []
    frames: list[torch.Tensor] = []
    anchor.eval()
    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        output = anchor(radar)
        features.append(EmptySceneSentinel.features(output.feature, output.logits, radar).cpu())
        labels.append(batch["presence"].float())
        sequences.append(batch["sequence_id"])
        frames.append(batch["frame_id"])
    return torch.cat(features), torch.cat(labels), torch.cat(sequences), torch.cat(frames)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    target = labels[order].astype(np.float64)
    if target.sum() == 0:
        return float("nan")
    precision = np.cumsum(target) / np.arange(1, target.size + 1)
    return float((precision * target).sum() / target.sum())


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.unique(np.concatenate(([0.0], scores, [1.0]))):
        predicted = scores >= threshold
        wrong_suppression = np.mean(~predicted[labels == 1]) if np.any(labels == 1) else 0.0
        empty_pass = np.mean(predicted[labels == 0]) if np.any(labels == 0) else 0.0
        candidates.append((wrong_suppression, empty_pass, float(threshold)))
    safe = [candidate for candidate in candidates if candidate[0] <= 0.01]
    if safe:
        return min(safe, key=lambda candidate: (candidate[1], candidate[0], -candidate[2]))[2]
    return min(candidates, key=lambda candidate: (candidate[0] + candidate[1], candidate[0]))[2]


@torch.no_grad()
def predict_presence_scores(model, features, device):
    model.eval()
    return torch.sigmoid(model.classifier(features.to(device)))[:, 0].cpu().numpy()


@torch.no_grad()
def evaluate(model, features, labels, threshold, device):
    scores = predict_presence_scores(model, features, device)
    target = labels.numpy().astype(np.int64)
    predicted = scores >= threshold
    return {
        "frame_count": float(target.size),
        "presence_auroc": rank_auroc(scores, target),
        "presence_auprc": average_precision(scores, target),
        "threshold": float(threshold),
        "wrong_suppression_rate": float(np.mean(~predicted[target == 1])) if np.any(target == 1) else float("nan"),
        "empty_suppression_rate": float(np.mean(~predicted[target == 0])) if np.any(target == 0) else float("nan"),
        "accuracy": float(np.mean(predicted == target)),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda")
    protocol = load_protocol(args.protocol)
    range_grid, angle_grid = load_cruw_grids(args.data_root, args.cruw_devkit_root)
    geometry = CRUWBEVGeometry()
    fit_records = build_all_scene_index(args.data_root, protocol, "fit", args.sample_stride)
    development_records = build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride)
    if {record.sequence_id for record in fit_records} & {record.sequence_id for record in development_records}:
        raise RuntimeError("fit/development sequence leakage")
    fit_dataset = RadarBEVWindowDataset(fit_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry)
    development_dataset = RadarBEVWindowDataset(
        development_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry
    )
    anchor = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    anchor_checkpoint = torch.load(args.anchor_checkpoint, map_location=device, weights_only=False)
    anchor.load_state_dict(anchor_checkpoint["model"], strict=True)
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    started = time.time()
    fit_features, fit_labels, fit_sequences, fit_frames = extract_features(
        anchor, fit_dataset, args.feature_batch_size, args.workers, device
    )
    development_features, development_labels, development_sequences, development_frames = extract_features(
        anchor, development_dataset, args.feature_batch_size, args.workers, device
    )
    torch.save(
        {
            "features": fit_features,
            "labels": fit_labels,
            "sequence_id": fit_sequences,
            "frame_id": fit_frames,
        },
        args.output_dir / "fit_feature_cache.pt",
    )
    torch.save(
        {
            "features": development_features,
            "labels": development_labels,
            "sequence_id": development_sequences,
            "frame_id": development_frames,
        },
        args.output_dir / "development_feature_cache.pt",
    )
    loader = DataLoader(
        TensorDataset(fit_features, fit_labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = EmptySceneSentinel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_auroc = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for features, labels in loader:
            logits = model.classifier(features.to(device))[:, 0]
            loss = F.binary_cross_entropy_with_logits(logits, labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            development_scores = torch.sigmoid(model.classifier(development_features.to(device)))[:, 0].cpu().numpy()
        development_target = development_labels.numpy().astype(np.int64)
        auroc = rank_auroc(development_scores, development_target)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "development_presence_auroc": auroc}
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if auroc > best_auroc + 1e-6:
            best_auroc = auroc
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    reloaded = EmptySceneSentinel().to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    dev_scores = predict_presence_scores(reloaded, development_features, device)
    threshold = choose_threshold(dev_scores, development_labels.numpy().astype(np.int64))
    report = {
        "model": "Empty-Scene Sentinel",
        "parameters": sum(parameter.numel() for parameter in reloaded.parameters()),
        "best_epoch": checkpoint["epoch"],
        "fit_sequences": sorted({record.sequence_id for record in fit_records}),
        "development_sequences": sorted({record.sequence_id for record in development_records}),
        "anchor_checkpoint_sha256": sha256(args.anchor_checkpoint),
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "independent_reload": evaluate(reloaded, development_features, development_labels, threshold, device),
        "elapsed_seconds": time.time() - started,
        "official_test_read": False,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
