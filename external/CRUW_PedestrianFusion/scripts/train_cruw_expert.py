#!/usr/bin/env python3
"""Train the CRUW PB-MHBR-style single-person expert on frozen DREAM-BEV."""

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
from torch.utils.data import DataLoader

from cruw_pedestrian_fusion.allscene import CRUWBEVGeometry, RadarBEVWindowDataset, build_all_scene_index
from cruw_pedestrian_fusion.canonical_dream_bev import DREAMBEV
from cruw_pedestrian_fusion.cruw_expert import CRUWPBMHBRExpert, expert_target_indices
from cruw_pedestrian_fusion.geometry import load_cruw_grids
from cruw_pedestrian_fusion.metrics import localization_metrics
from cruw_pedestrian_fusion.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cruw-devkit-root", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--identity-negative-ratio", type=float, default=0.20)
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


def shifted_previous(radar: torch.Tensor) -> torch.Tensor:
    return torch.cat((radar[:, :1], radar[:, :-1]), dim=1)


def select_records(records, allowed: set[int], geometry: CRUWBEVGeometry, negative_ratio: float):
    positive = [
        record
        for record in records
        if record.sequence_id in allowed
        and sum(geometry.contains(pedestrian.x_m, pedestrian.y_m) for pedestrian in record.pedestrians) == 1
    ]
    empty = [
        record
        for record in records
        if record.sequence_id in allowed
        and sum(geometry.contains(pedestrian.x_m, pedestrian.y_m) for pedestrian in record.pedestrians) == 0
    ]
    negative_count = min(len(empty), int(round(len(positive) * negative_ratio)))
    return positive + empty[:negative_count], len(positive), negative_count


@torch.no_grad()
def evaluate(anchor, expert, loader, device):
    anchor.eval()
    expert.eval()
    parts = {key: [] for key in ("base_x", "base_y", "expert_x", "expert_y", "target_x", "target_y", "score")}
    accepted: list[np.ndarray] = []
    oracle_errors: list[np.ndarray] = []
    for batch in loader:
        valid = batch["target_count"].numpy() == 1
        if not valid.any():
            continue
        radar = batch["radar"].to(device, non_blocking=True)
        current = anchor(radar)
        previous = anchor(shifted_previous(radar))
        output = expert(
            polar_feature=current.polar_feature,
            shared_feature=current.feature,
            radar=radar,
            base_logits=current.logits,
            previous_logits=previous.logits,
        )
        score = torch.sigmoid(current.logits).flatten(1).amax(dim=1).cpu().numpy()
        target = torch.stack((batch["single_x_m"], batch["single_y_m"]), dim=1).to(device)
        candidates = output["candidate_xy_m"]
        oracle_errors.append(torch.linalg.vector_norm(candidates - target[:, None], dim=-1).amin(dim=1).cpu().numpy()[valid])
        parts["base_x"].append(output["base_xy_m"][:, 0].cpu().numpy()[valid])
        parts["base_y"].append(output["base_xy_m"][:, 1].cpu().numpy()[valid])
        parts["expert_x"].append(output["proposal_xy_m"][:, 0].cpu().numpy()[valid])
        parts["expert_y"].append(output["proposal_xy_m"][:, 1].cpu().numpy()[valid])
        parts["target_x"].append(target[:, 0].cpu().numpy()[valid])
        parts["target_y"].append(target[:, 1].cpu().numpy()[valid])
        parts["score"].append(score[valid])
        accepted.append((output["selected_index"].cpu().numpy()[valid] != 0).astype(np.float32))
    values = {key: np.concatenate(value) for key, value in parts.items()}
    base = localization_metrics(values["base_x"], values["base_y"], values["target_x"], values["target_y"], values["score"])
    refined = localization_metrics(
        values["expert_x"], values["expert_y"], values["target_x"], values["target_y"], values["score"]
    )
    oracle = np.concatenate(oracle_errors)
    return {
        "base": base,
        "expert": refined,
        "delta_AP@0.30_pp": 100.0 * (refined["AP@0.30"] - base["AP@0.30"]),
        "acceptance_rate": float(np.concatenate(accepted).mean()),
        "oracle_AP@0.30": float(np.mean(oracle <= 0.30)),
    }


@torch.no_grad()
def calibrate_nonbase_bias(anchor, expert, loader, device) -> dict[str, object]:
    """Select one global expert-admission bias on Development only."""
    anchor.eval()
    expert.eval()
    expert.nonbase_bias.zero_()
    candidate_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        current = anchor(radar)
        previous = anchor(shifted_previous(radar))
        output = expert(
            polar_feature=current.polar_feature,
            shared_feature=current.feature,
            radar=radar,
            base_logits=current.logits,
            previous_logits=previous.logits,
        )
        candidate_parts.append(output["candidate_xy_m"].cpu().numpy())
        logit_parts.append(output["selector_logits"].cpu().numpy())
        target_parts.append(torch.stack((batch["single_x_m"], batch["single_y_m"]), dim=1).numpy())
        score_parts.append(torch.sigmoid(current.logits).flatten(1).amax(dim=1).cpu().numpy())
    candidates = np.concatenate(candidate_parts)
    logits = np.concatenate(logit_parts)
    targets = np.concatenate(target_parts)
    scores = np.concatenate(score_parts)
    best: tuple[float, float, float] | None = None
    best_payload: dict[str, object] | None = None
    for bias in np.linspace(-2.0, 2.0, 161):
        adjusted = logits.copy()
        adjusted[:, 1:] += float(bias)
        selected = adjusted.argmax(axis=1)
        prediction = candidates[np.arange(candidates.shape[0]), selected]
        metrics = localization_metrics(prediction[:, 0], prediction[:, 1], targets[:, 0], targets[:, 1], scores)
        rank = (metrics["AP@0.30"], -metrics["P90_error_m"], -abs(float(bias)))
        if best is None or rank > best:
            best = rank
            best_payload = {
                "nonbase_bias": float(bias),
                "metrics": metrics,
                "acceptance_rate": float(np.mean(selected != 0)),
            }
    if best_payload is None:
        raise RuntimeError("expert calibration produced no candidate")
    return best_payload


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda")
    geometry = CRUWBEVGeometry()
    protocol = load_protocol(args.protocol)
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    fit_allowed = {int(value) for value in payload["single_person_fit_sequences"]}
    development_allowed = {int(value) for value in payload["single_person_validation_sequences"]}
    if fit_allowed & development_allowed:
        raise RuntimeError("single-person fit/development leakage")
    range_grid, angle_grid = load_cruw_grids(args.data_root, args.cruw_devkit_root)
    fit_all = build_all_scene_index(args.data_root, protocol, "fit", args.sample_stride)
    development_all = build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride)
    fit_records, fit_positive, fit_negative = select_records(
        fit_all, fit_allowed, geometry, args.identity_negative_ratio
    )
    development_records, development_positive, _ = select_records(
        development_all, development_allowed, geometry, 0.0
    )
    fit_dataset = RadarBEVWindowDataset(fit_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry)
    development_dataset = RadarBEVWindowDataset(
        development_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry
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
    anchor = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    checkpoint = torch.load(args.anchor_checkpoint, map_location=device, weights_only=False)
    anchor.load_state_dict(checkpoint["model"], strict=True)
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    anchor.eval()
    expert = CRUWPBMHBRExpert(range_grid, angle_grid, geometry).to(device)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_ap30 = -float("inf")
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        expert.train()
        losses: list[float] = []
        nonbase_targets: list[float] = []
        for batch in fit_loader:
            radar = batch["radar"].to(device, non_blocking=True)
            with torch.no_grad():
                current = anchor(radar)
                previous = anchor(shifted_previous(radar))
            output = expert(
                polar_feature=current.polar_feature,
                shared_feature=current.feature,
                radar=radar,
                base_logits=current.logits,
                previous_logits=previous.logits,
            )
            target_xy = torch.stack((batch["single_x_m"], batch["single_y_m"]), dim=1).to(device)
            positive = batch["target_count"].to(device) == 1
            safe_target = torch.where(positive[:, None], target_xy, output["base_xy_m"].detach())
            labels = expert_target_indices(output["candidate_xy_m"].detach(), safe_target)
            labels = torch.where(positive, labels, torch.zeros_like(labels))
            per_sample = F.cross_entropy(output["selector_logits"], labels, reduction="none")
            weights = torch.where(positive & labels.eq(0), per_sample.new_tensor(0.35), per_sample.new_tensor(1.0))
            loss = (per_sample * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            nonbase_targets.append(float(labels.ne(0).float().mean()))
        development = evaluate(anchor, expert, development_loader, device)
        ap30 = development["expert"]["AP@0.30"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "nonbase_target_rate": float(np.mean(nonbase_targets)),
            "elapsed_seconds": time.time() - started,
            **development,
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if ap30 > best_ap30 + 1e-6:
            best_ap30 = ap30
            stale = 0
            torch.save({"model": expert.state_dict(), "epoch": epoch, "metrics": development}, args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    reloaded = CRUWPBMHBRExpert(range_grid, angle_grid, geometry).to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    calibration = calibrate_nonbase_bias(anchor, reloaded, development_loader, device)
    reloaded.nonbase_bias.fill_(float(calibration["nonbase_bias"]))
    torch.save(
        {
            "model": reloaded.state_dict(),
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "development_calibration": calibration,
        },
        args.output_dir / "best.pt",
    )
    verified = CRUWPBMHBRExpert(range_grid, angle_grid, geometry).to(device)
    verified.load_state_dict(torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)["model"], strict=True)
    report = {
        "model": "CRUW PB-MHBR single-person expert",
        "parameters": sum(parameter.numel() for parameter in verified.parameters()),
        "best_epoch": checkpoint["epoch"],
        "fit_sequences": sorted(fit_allowed),
        "development_sequences": sorted(development_allowed),
        "fit_positive_frames": fit_positive,
        "fit_identity_negative_frames": fit_negative,
        "development_positive_frames": development_positive,
        "anchor_checkpoint_sha256": sha256(args.anchor_checkpoint),
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "development_calibration": calibration,
        "independent_reload": evaluate(anchor, verified, development_loader, device),
        "elapsed_seconds": time.time() - started,
        "official_test_read": False,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
