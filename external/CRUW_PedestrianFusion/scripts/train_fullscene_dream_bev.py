#!/usr/bin/env python3
"""Train the canonical 1.203M DREAM-BEV on the frozen CRUW full-scene protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cruw_pedestrian_fusion.allscene import CRUWBEVGeometry, RadarBEVWindowDataset, build_all_scene_index
from cruw_pedestrian_fusion.canonical_dream_bev import DREAMBEV
from cruw_pedestrian_fusion.geometry import load_cruw_grids
from cruw_pedestrian_fusion.metrics import decode_bev_logits, localization_metrics, rank_auroc
from cruw_pedestrian_fusion.models import gaussian_focal_bce
from cruw_pedestrian_fusion.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cruw-devkit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


@torch.no_grad()
def evaluate(
    model: DREAMBEV,
    loader: DataLoader,
    geometry: CRUWBEVGeometry,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    presence_scores: list[np.ndarray] = []
    presence_labels: list[np.ndarray] = []
    single_parts: dict[str, list[np.ndarray]] = {key: [] for key in ("x", "y", "target_x", "target_y", "score")}
    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        logits = model(radar).logits
        losses.append(float(gaussian_focal_bce(logits, target).item()))
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
        metrics.update(
            {
                f"single_{key}": value
                for key, value in localization_metrics(
                    np.concatenate(single_parts["x"]),
                    np.concatenate(single_parts["y"]),
                    np.concatenate(single_parts["target_x"]),
                    np.concatenate(single_parts["target_y"]),
                    np.concatenate(single_parts["score"]),
                ).items()
            }
        )
    return metrics


def selection_score(metrics: dict[str, float]) -> float:
    ap30 = metrics.get("single_AP@0.30", 0.0)
    auroc = metrics.get("presence_auroc", 0.5)
    if not np.isfinite(auroc):
        auroc = 0.5
    empty_fpr = metrics.get("empty_false_positive_rate_at_0.5", 0.0)
    if not np.isfinite(empty_fpr):
        empty_fpr = 0.0
    return float(ap30 + 0.10 * auroc - 0.10 * empty_fpr)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    geometry = CRUWBEVGeometry()
    protocol = load_protocol(args.protocol)
    range_grid, angle_grid = load_cruw_grids(args.data_root, args.cruw_devkit_root)
    fit_records = build_all_scene_index(args.data_root, protocol, "fit", args.sample_stride)
    development_records = build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride)
    fit_ids = {record.sequence_id for record in fit_records}
    development_ids = {record.sequence_id for record in development_records}
    if fit_ids & development_ids:
        raise RuntimeError(f"fit/development leakage: {sorted(fit_ids & development_ids)}")
    fit_set = RadarBEVWindowDataset(fit_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry)
    development_set = RadarBEVWindowDataset(
        development_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry
    )
    fit_loader = make_loader(fit_set, args.batch_size, args.workers, shuffle=True)
    development_loader = make_loader(development_set, args.batch_size, args.workers, shuffle=False)
    model = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 1_202_785:
        raise RuntimeError(f"canonical DREAM-BEV parameter mismatch: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {
        "model": "DREAM-BEV",
        "parameters": parameter_count,
        "source_partition_read": "train",
        "official_test_read": False,
        "fit_sequences": sorted(fit_ids),
        "development_sequences": sorted(development_ids),
        "fit_frames": len(fit_records),
        "development_frames": len(development_records),
        "protocol_sha256": sha256(args.protocol),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    history_path = args.output_dir / "history.jsonl"
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in fit_loader:
            radar = batch["radar"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = gaussian_focal_bce(model(radar).logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
        scheduler.step()
        development = evaluate(model, development_loader, geometry, device)
        score = selection_score(development)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "selection_score": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - start,
            **development,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": development, "metadata": metadata},
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    reloaded = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    independent = evaluate(reloaded, development_loader, geometry, device)
    result = {
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "independent_reload": independent,
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
