#!/usr/bin/env python3
"""Train the canonical 1.203M DREAM-BEV on the frozen CARRADA full-scene protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from carrada_dream_bev.data import (
    CARRADABEVGeometry,
    RadarBEVWindowDataset,
    apply_single_target_episode_manifest,
    build_all_scene_index,
    carrada_ra_grids,
    load_single_target_episode_manifest,
)
from carrada_dream_bev.canonical_dream_bev import DREAMBEV
from carrada_dream_bev.metrics import decode_bev_logits, localization_metrics, rank_auroc
from carrada_dream_bev.models import (
    categorized_frame_weights,
    normalized_gaussian_focal_loss,
    sample_weighted_gaussian_focal_loss,
)
from carrada_dream_bev.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument(
        "--development-episode-manifest",
        type=Path,
        help="Optional fixed-identity manifest used for Development checkpoint selection.",
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--positive-sampling-fraction", type=float, default=0.0)
    parser.add_argument("--background-frame-loss-weight", type=float, default=0.02)
    parser.add_argument("--hard-negative-frame-loss-weight", type=float, default=0.75)
    parser.add_argument("--positive-frame-loss-weight", type=float, default=2.0)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def episode_development_records(records, manifest_path: Path):
    """Restrict Development selection to the declared causal single-target episodes."""

    episodes = load_single_target_episode_manifest(manifest_path, "development")
    selected = apply_single_target_episode_manifest(records, episodes)
    if not selected:
        raise RuntimeError("development episode manifest selected no synchronized records")
    return selected


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


def balanced_sampling_weights(
    records,
    geometry: CARRADABEVGeometry,
    positive_fraction: float,
) -> torch.Tensor:
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("positive_sampling_fraction must lie strictly between zero and one")
    positive = np.asarray(
        [any(geometry.contains(target.x_m, target.y_m) for target in record.targets) for record in records],
        dtype=bool,
    )
    positive_count = int(positive.sum())
    negative_count = int((~positive).sum())
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError("balanced sampling requires both positive and empty Fit frames")
    values = np.where(
        positive,
        positive_fraction / positive_count,
        (1.0 - positive_fraction) / negative_count,
    )
    return torch.as_tensor(values, dtype=torch.double)


def make_loader(
    dataset: RadarBEVWindowDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


@torch.no_grad()
def evaluate(
    model: DREAMBEV,
    loader: DataLoader,
    geometry: CARRADABEVGeometry,
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
    return float(ap30 + 1e-3 * auroc - 1e-3 * empty_fpr)


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
    geometry = CARRADABEVGeometry()
    protocol = load_protocol(args.protocol)
    range_grid, angle_grid = carrada_ra_grids()
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
    fit_set = RadarBEVWindowDataset(fit_records, args.cache_root, args.clip_length, geometry)
    development_set = RadarBEVWindowDataset(
        development_records, args.cache_root, args.clip_length, geometry
    )
    sampler = None
    if args.positive_sampling_fraction > 0.0:
        weights = balanced_sampling_weights(
            fit_records,
            geometry,
            args.positive_sampling_fraction,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(fit_records),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    fit_loader = make_loader(fit_set, args.batch_size, args.workers, shuffle=True, sampler=sampler)
    development_loader = make_loader(development_set, args.batch_size, args.workers, shuffle=False)
    model = DREAMBEV(
        range_grid,
        angle_grid,
        geometry,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    canonical_mamba = (args.mamba_d_state, args.mamba_d_conv, args.mamba_expand) == (16, 4, 2)
    if canonical_mamba and parameter_count != 1_202_785:
        raise RuntimeError(f"canonical DREAM-BEV parameter mismatch: {parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {
        "model": "DREAM-BEV",
        "supervision_loss": "category-weighted CenterNet-style Gaussian focal loss",
        "positive_sampling_fraction": args.positive_sampling_fraction,
        "frame_loss_weights": {
            "clean_background": args.background_frame_loss_weight,
            "car_or_cyclist_hard_negative": args.hard_negative_frame_loss_weight,
            "pedestrian_positive": args.positive_frame_loss_weight,
        },
        "parameters": parameter_count,
        "mamba_configuration": {
            "d_state": args.mamba_d_state,
            "d_conv": args.mamba_d_conv,
            "expand": args.mamba_expand,
            "canonical": canonical_mamba,
        },
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
            frame_weights = categorized_frame_weights(
                batch["presence"].to(device, non_blocking=True),
                batch["distractor_count"].to(device, non_blocking=True),
                background_weight=args.background_frame_loss_weight,
                hard_negative_weight=args.hard_negative_frame_loss_weight,
                positive_weight=args.positive_frame_loss_weight,
            )
            loss = sample_weighted_gaussian_focal_loss(
                model(radar).logits,
                target,
                frame_weights=frame_weights,
            )
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
    reloaded = DREAMBEV(
        range_grid,
        angle_grid,
        geometry,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
    ).to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    independent = evaluate(reloaded, development_loader, geometry, device)
    result = {
        "best_epoch": best_epoch,
        "selection_score": best_score,
        "independent_reload": independent,
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "locked_test_read": False,
        "elapsed_seconds": time.time() - start,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

