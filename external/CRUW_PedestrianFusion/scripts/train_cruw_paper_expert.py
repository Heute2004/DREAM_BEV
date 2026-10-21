#!/usr/bin/env python3
"""Train one paper recovery specialist on a frozen CRUW DREAM-BEV anchor."""

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

from cruw_pedestrian_fusion.allscene import (
    CRUWBEVGeometry,
    RadarBEVWindowDataset,
    build_all_scene_index,
)
from cruw_pedestrian_fusion.canonical_dream_bev import DREAMBEV
from cruw_pedestrian_fusion.cruw_expert import _decode_peak_tensor
from cruw_pedestrian_fusion.geometry import load_cruw_grids
from cruw_pedestrian_fusion.metrics import localization_metrics
from cruw_pedestrian_fusion.models import gaussian_focal_bce
from cruw_pedestrian_fusion.paper_experts import (
    PAPER_EXPERT_KINDS,
    CRUWPaperExpert,
    paper_expert_target_indices,
)
from cruw_pedestrian_fusion.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cruw-devkit-root", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-kind", choices=PAPER_EXPERT_KINDS, required=True)
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
    parser.add_argument("--minimum-improvement-m", type=float, default=0.05)
    parser.add_argument("--candidate-loss-weight", type=float, default=0.20)
    parser.add_argument("--coordinate-loss-weight", type=float, default=0.10)
    parser.add_argument("--acceptance-loss-weight", type=float, default=0.05)
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
def frozen_evidence(anchor: DREAMBEV, radar: torch.Tensor):
    current = anchor(radar)
    previous = anchor(shifted_previous(radar))
    return current, previous


@torch.no_grad()
def evaluate(anchor, expert, loader, device) -> dict[str, object]:
    anchor.eval()
    expert.eval()
    parts = {key: [] for key in ("base_x", "base_y", "expert_x", "expert_y", "target_x", "target_y", "score")}
    accepted: list[np.ndarray] = []
    for batch in loader:
        valid = batch["target_count"].numpy() == 1
        if not valid.any():
            continue
        radar = batch["radar"].to(device, non_blocking=True)
        current, previous = frozen_evidence(anchor, radar)
        output = expert(
            polar_feature=current.polar_feature,
            shared_feature=current.feature,
            radar=radar,
            base_logits=current.logits,
            previous_logits=previous.logits,
        )
        expert_xy = _decode_peak_tensor(output["logits"], expert.geometry)
        base_xy = output["base_xy_m"]
        target = torch.stack((batch["single_x_m"], batch["single_y_m"]), dim=1).to(device)
        score = torch.sigmoid(current.logits).flatten(1).amax(dim=1).cpu().numpy()
        parts["base_x"].append(base_xy[:, 0].cpu().numpy()[valid])
        parts["base_y"].append(base_xy[:, 1].cpu().numpy()[valid])
        parts["expert_x"].append(expert_xy[:, 0].cpu().numpy()[valid])
        parts["expert_y"].append(expert_xy[:, 1].cpu().numpy()[valid])
        parts["target_x"].append(target[:, 0].cpu().numpy()[valid])
        parts["target_y"].append(target[:, 1].cpu().numpy()[valid])
        parts["score"].append(score[valid])
        accepted.append((expert_xy.ne(base_xy).any(dim=1).cpu().numpy()[valid]).astype(np.float32))
    values = {key: np.concatenate(value) for key, value in parts.items()}
    base = localization_metrics(values["base_x"], values["base_y"], values["target_x"], values["target_y"], values["score"])
    refined = localization_metrics(
        values["expert_x"], values["expert_y"], values["target_x"], values["target_y"], values["score"]
    )
    return {
        "base": base,
        "expert": refined,
        "delta_AP@0.30_pp": 100.0 * (refined["AP@0.30"] - base["AP@0.30"]),
        "changed_prediction_rate": float(np.concatenate(accepted).mean()),
    }


def _training_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target"]
    positive = batch["target_count"] == 1
    target_xy = torch.stack((batch["single_x_m"], batch["single_y_m"]), dim=1)
    safe_target = torch.where(positive[:, None], target_xy, output["base_xy_m"].detach())
    labels = paper_expert_target_indices(
        output["candidate_xy_m"].detach(), safe_target, minimum_improvement_m=args.minimum_improvement_m
    )
    labels = torch.where(positive, labels, torch.zeros_like(labels))
    focal = gaussian_focal_bce(output["logits"], target)
    selector = F.cross_entropy(output["candidate_selector_logits"], labels)
    if positive.any():
        coordinate = F.smooth_l1_loss(output["proposal_xy_m"][positive], target_xy[positive], beta=0.10)
        base_error = torch.linalg.vector_norm(output["base_xy_m"][positive] - target_xy[positive], dim=1)
        proposal_error = torch.linalg.vector_norm(output["proposal_xy_m"][positive] - target_xy[positive], dim=1)
        improved = (proposal_error + args.minimum_improvement_m <= base_error).to(output["acceptance"].dtype)
        acceptance = F.binary_cross_entropy(output["acceptance"][positive, 0], improved)
    else:
        coordinate = focal.new_zeros(())
        acceptance = focal.new_zeros(())
    loss = (
        focal
        + args.candidate_loss_weight * selector
        + args.coordinate_loss_weight * coordinate
        + args.acceptance_loss_weight * acceptance
    )
    return loss, {
        "focal_loss": float(focal.detach()),
        "candidate_loss": float(selector.detach()),
        "coordinate_loss": float(coordinate.detach()),
        "acceptance_loss": float(acceptance.detach()),
        "nonbase_target_rate": float(labels.ne(0).float().mean()),
    }


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
    fit_records, fit_positive, fit_negative = select_records(
        build_all_scene_index(args.data_root, protocol, "fit", args.sample_stride),
        fit_allowed,
        geometry,
        args.identity_negative_ratio,
    )
    development_records, development_positive, _ = select_records(
        build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride),
        development_allowed,
        geometry,
        0.0,
    )
    fit_loader = DataLoader(
        RadarBEVWindowDataset(fit_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    development_loader = DataLoader(
        RadarBEVWindowDataset(development_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    anchor = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    anchor.load_state_dict(torch.load(args.anchor_checkpoint, map_location=device, weights_only=False)["model"], strict=True)
    anchor.requires_grad_(False)
    anchor.eval()
    expert = CRUWPaperExpert(args.expert_kind, geometry).to(device)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {
        "model": f"CRUW transfer of paper {args.expert_kind} specialist",
        "contract": "frozen DREAM-BEV; causal RA/RA-MTI only; single-person Fit/Development; no RGB, depth, VLM, future frame, or CRUW official test",
        "fit_sequences": sorted(fit_allowed),
        "development_sequences": sorted(development_allowed),
        "fit_positive_frames": fit_positive,
        "fit_identity_negative_frames": fit_negative,
        "development_positive_frames": development_positive,
        "trainable_parameters": sum(parameter.numel() for parameter in expert.parameters() if parameter.requires_grad),
        "anchor_sha256": sha256(args.anchor_checkpoint),
        "protocol_sha256": sha256(args.protocol),
        "official_test_read": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best_ap30 = -float("inf")
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        expert.train()
        rows: list[dict[str, float]] = []
        for source_batch in fit_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in source_batch.items()}
            with torch.no_grad():
                current, previous = frozen_evidence(anchor, batch["radar"])
            output = expert(
                polar_feature=current.polar_feature,
                shared_feature=current.feature,
                radar=batch["radar"],
                base_logits=current.logits,
                previous_logits=previous.logits,
            )
            loss, terms = _training_loss(output, batch, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 5.0)
            optimizer.step()
            rows.append({"loss": float(loss.detach()), **terms})
        scheduler.step()
        development = evaluate(anchor, expert, development_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
            **{key: float(np.mean([item[key] for item in rows])) for key in rows[0]},
            **development,
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        ap30 = float(development["expert"]["AP@0.30"])
        if ap30 > best_ap30 + 1e-6:
            best_ap30 = ap30
            stale = 0
            torch.save({"model": expert.state_dict(), "epoch": epoch, "metrics": development}, args.output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    verified = CRUWPaperExpert(args.expert_kind, geometry).to(device)
    verified.load_state_dict(checkpoint["model"], strict=True)
    report = {
        "model": f"CRUW transfer of paper {args.expert_kind} specialist",
        "parameters": sum(parameter.numel() for parameter in verified.parameters()),
        "best_epoch": checkpoint["epoch"],
        "fit_sequences": sorted(fit_allowed),
        "development_sequences": sorted(development_allowed),
        "fit_positive_frames": fit_positive,
        "fit_identity_negative_frames": fit_negative,
        "development_positive_frames": development_positive,
        "anchor_checkpoint_sha256": sha256(args.anchor_checkpoint),
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "independent_reload": evaluate(anchor, verified, development_loader, device),
        "elapsed_seconds": time.time() - started,
        "official_test_read": False,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
