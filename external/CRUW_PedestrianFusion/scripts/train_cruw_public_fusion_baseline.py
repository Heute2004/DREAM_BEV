#!/usr/bin/env python3
"""Train one source-backed public fusion core under the frozen CRUW protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cruw_pedestrian_fusion.allscene import (
    CRUWBEVGeometry,
    RadarBEVWindowDataset,
    build_all_scene_index,
)
from cruw_pedestrian_fusion.allscene_visual import AllSceneRGBDataset
from cruw_pedestrian_fusion.canonical_dream_bev import DREAMBEV
from cruw_pedestrian_fusion.cruw_expert import CRUWPBMHBRExpert
from cruw_pedestrian_fusion.global_association_expert import CRUWGlobalAssociationExpert
from cruw_pedestrian_fusion.geometry import load_cruw_grids
from cruw_pedestrian_fusion.metrics import decode_bev_logits, localization_metrics
from cruw_pedestrian_fusion.models import gaussian_focal_bce
from cruw_pedestrian_fusion.paper_experts import PAPER_EXPERT_KINDS, CRUWPaperExpert
from cruw_pedestrian_fusion.protocol import load_protocol
from cruw_pedestrian_fusion.public_fusion_baselines import (
    build_public_official_baseline,
    recenter_logits_on_radar_candidate,
)
from cruw_pedestrian_fusion.public_fusion_evidence import (
    RadarEvidence,
    VisualBearingEvidence,
)
from cruw_pedestrian_fusion.rv_bearing_net.model import RVBearingNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("rcrosnet", "crn", "rcbevdet"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cruw-devkit-root", type=Path, required=True)
    parser.add_argument("--public-source-root", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expert-kind",
        choices=("pbmhbr", "global-association", *PAPER_EXPERT_KINDS),
        default="pbmhbr",
    )
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
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


def select_single_person_records(records, allowed: set[int], geometry: CRUWBEVGeometry):
    return [
        record
        for record in records
        if record.sequence_id in allowed
        and sum(geometry.contains(person.x_m, person.y_m) for person in record.pedestrians) == 1
    ]


class AlignedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records,
        cache_root: Path,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        clip_length: int,
        image_size: int,
        geometry: CRUWBEVGeometry,
        training: bool,
    ) -> None:
        self.radar = RadarBEVWindowDataset(
            records, cache_root, range_grid, angle_grid, clip_length, geometry
        )
        self.visual = AllSceneRGBDataset(
            records, clip_length, image_size, training=training, geometry=geometry
        )

    def __len__(self) -> int:
        return len(self.radar)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        radar = self.radar[index]
        visual = self.visual[index]
        if not torch.equal(radar["sequence_id"], visual["sequence_id"]) or not torch.equal(
            radar["frame_id"], visual["frame_id"]
        ):
            raise RuntimeError("radar/RGB record alignment failure")
        radar["images"] = visual["images"]
        radar["frame_valid"] = visual["frame_valid"]
        return radar


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _build_expert(kind, range_grid, angle_grid, geometry):
    if kind == "pbmhbr":
        return CRUWPBMHBRExpert(range_grid, angle_grid, geometry)
    if kind == "global-association":
        return CRUWGlobalAssociationExpert(range_grid, angle_grid, geometry)
    if kind in PAPER_EXPERT_KINDS:
        return CRUWPaperExpert(kind, geometry)
    raise ValueError(f"unknown expert kind: {kind}")


def _run_expert(kind, expert, radar, current, previous, visual_output):
    arguments = {
        "polar_feature": current.polar_feature,
        "shared_feature": current.feature,
        "radar": radar,
        "base_logits": current.logits,
        "previous_logits": previous.logits,
    }
    if kind == "global-association":
        arguments.update(
            {
                "visual_theta": visual_output.theta[:, 0],
                "visual_sigma": visual_output.sigma_theta[:, 0],
                "visual_reliability": visual_output.reliability[:, 0],
                "visual_feature": visual_output.temporal_feature,
            }
        )
    return expert(**arguments)


def _expert_logits_for_public_fusion(expert_kind, current_logits, expert_output, geometry):
    """Select the deployed expert BEV map without substituting an internal proposal."""

    if expert_kind in PAPER_EXPERT_KINDS:
        return expert_output["logits"]
    return recenter_logits_on_radar_candidate(
        current_logits,
        expert_output["base_xy_m"],
        expert_output["proposal_xy_m"],
        geometry,
    )


@torch.no_grad()
def build_evidence(anchor, expert_kind, expert, visual, batch, device, geometry):
    radar_window = batch["radar"].to(device, non_blocking=True)
    current = anchor(radar_window)
    previous = anchor(shifted_previous(radar_window))
    visual_output = visual(
        batch["images"].to(device, non_blocking=True),
        frame_valid=batch["frame_valid"].to(device, non_blocking=True),
    )
    expert_output = _run_expert(
        expert_kind,
        expert,
        radar_window,
        current,
        previous,
        visual_output,
    )
    expert_logits = _expert_logits_for_public_fusion(
        expert_kind, current.logits, expert_output, geometry
    )
    radar_evidence = RadarEvidence(
        polar_feature=current.polar_feature,
        bev_feature=current.feature,
        logits=expert_logits,
    )
    visual_evidence = VisualBearingEvidence(
        theta=visual_output.theta[:, 0],
        sigma_theta=visual_output.sigma_theta[:, 0],
        reliability=visual_output.reliability[:, 0],
        temporal_feature=visual_output.temporal_feature,
    )
    anchor_score = torch.sigmoid(current.logits).flatten(1).amax(dim=1)
    return radar_evidence, visual_evidence, anchor_score


def bundle_metrics(parts: dict[str, list[np.ndarray]]) -> dict[str, object]:
    values = {key: np.concatenate(value) for key, value in parts.items()}
    radar = localization_metrics(
        values["radar_x"],
        values["radar_y"],
        values["target_x"],
        values["target_y"],
        values["radar_score"],
    )
    fusion = localization_metrics(
        values["fused_x"],
        values["fused_y"],
        values["target_x"],
        values["target_y"],
        values["fused_score"],
    )
    per_sequence: dict[str, object] = {}
    for sequence_id in sorted(np.unique(values["sequence_id"])):
        mask = values["sequence_id"] == sequence_id
        before = localization_metrics(
            values["radar_x"][mask],
            values["radar_y"][mask],
            values["target_x"][mask],
            values["target_y"][mask],
            values["radar_score"][mask],
        )
        after = localization_metrics(
            values["fused_x"][mask],
            values["fused_y"][mask],
            values["target_x"][mask],
            values["target_y"][mask],
            values["fused_score"][mask],
        )
        per_sequence[str(int(sequence_id))] = {
            "expert_radar": before,
            "fusion": after,
            "delta_AP@0.30_pp": 100.0 * (after["AP@0.30"] - before["AP@0.30"]),
        }
    return {
        "expert_radar": radar,
        "fusion": fusion,
        "delta_AP@0.30_pp": 100.0 * (fusion["AP@0.30"] - radar["AP@0.30"]),
        "per_sequence": per_sequence,
    }


@torch.no_grad()
def evaluate(model, anchor, expert_kind, expert, visual, loader, device, geometry, max_batches=0):
    model.eval()
    parts = {
        key: []
        for key in (
            "radar_x",
            "radar_y",
            "fused_x",
            "fused_y",
            "target_x",
            "target_y",
            "radar_score",
            "fused_score",
            "sequence_id",
        )
    }
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        radar_evidence, visual_evidence, anchor_score = build_evidence(
            anchor, expert_kind, expert, visual, batch, device, geometry
        )
        output = model(radar_evidence, visual_evidence)
        radar_prediction = decode_bev_logits(output.radar_logits, geometry)
        fused_prediction = decode_bev_logits(output.fused_logits, geometry)
        target_x = batch["single_x_m"].numpy()
        target_y = batch["single_y_m"].numpy()
        parts["radar_x"].append(radar_prediction["x_m"])
        parts["radar_y"].append(radar_prediction["y_m"])
        parts["fused_x"].append(fused_prediction["x_m"])
        parts["fused_y"].append(fused_prediction["y_m"])
        parts["target_x"].append(target_x)
        parts["target_y"].append(target_y)
        # Preserve the exact frozen-anchor ranking used by the expert report.
        parts["radar_score"].append(anchor_score.cpu().numpy())
        parts["fused_score"].append(anchor_score.cpu().numpy())
        parts["sequence_id"].append(batch["sequence_id"].numpy())
    return bundle_metrics(parts)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda")
    geometry = CRUWBEVGeometry()
    protocol = load_protocol(args.protocol)
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    fit_allowed = {int(value) for value in payload["single_person_fit_sequences"]}
    development_allowed = {
        int(value) for value in payload["single_person_validation_sequences"]
    }
    if fit_allowed & development_allowed:
        raise RuntimeError("single-person fit/development leakage")
    range_grid, angle_grid = load_cruw_grids(args.data_root, args.cruw_devkit_root)
    fit_records = select_single_person_records(
        build_all_scene_index(args.data_root, protocol, "fit", args.sample_stride),
        fit_allowed,
        geometry,
    )
    development_records = select_single_person_records(
        build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride),
        development_allowed,
        geometry,
    )
    fit_loader = make_loader(
        AlignedDataset(
            fit_records,
            args.cache_root,
            range_grid,
            angle_grid,
            args.clip_length,
            args.image_size,
            geometry,
            True,
        ),
        args.batch_size,
        args.workers,
        True,
    )
    development_loader = make_loader(
        AlignedDataset(
            development_records,
            args.cache_root,
            range_grid,
            angle_grid,
            args.clip_length,
            args.image_size,
            geometry,
            False,
        ),
        args.batch_size,
        args.workers,
        False,
    )

    anchor = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    anchor.load_state_dict(
        torch.load(args.anchor_checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    expert = _build_expert(args.expert_kind, range_grid, angle_grid, geometry).to(device)
    expert.load_state_dict(
        torch.load(args.expert_checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    visual = RVBearingNet().to(device)
    visual.load_state_dict(
        torch.load(args.visual_checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    for module in (anchor, expert, visual):
        freeze(module)
    model = build_public_official_baseline(args.baseline, args.public_source_root).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    metadata = {
        "stage": "source_backed_public_fusion_baseline_cruw_development",
        "baseline": args.baseline,
        "architecture_source": getattr(model, "source_metadata", None),
        "task_adaptation": "frozen CRUW expert-radar BEV + frozen causal left-RGB bearing evidence -> single-person metric BEV",
        "official_3d_reproduction": False,
        "official_fusion_core_modified": False,
        "radar_anchor": f"frozen CRUW DREAM-BEV plus frozen {args.expert_kind} expert",
        "expert_kind": args.expert_kind,
        "fit_sequences": sorted(fit_allowed),
        "development_sequences": sorted(development_allowed),
        "fit_frames": len(fit_records),
        "development_frames": len(development_records),
        "expert_anchor_representation": (
            "final_bounded_paper_expert_bev_logit_map"
            if args.expert_kind in PAPER_EXPERT_KINDS
            else "recentered_frozen_anchor_at_expert_proposal"
        ),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "anchor_sha256": sha256(args.anchor_checkpoint),
        "expert_sha256": sha256(args.expert_checkpoint),
        "visual_sha256": sha256(args.visual_checkpoint),
        "protocol_sha256": sha256(args.protocol),
        "seed": args.seed,
        "official_test_read": False,
        "depth_input": False,
        "future_frame_input": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    best_ap30 = -float("inf")
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(fit_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            radar_evidence, visual_evidence, _ = build_evidence(
                anchor, args.expert_kind, expert, visual, batch, device, geometry
            )
            output = model(radar_evidence, visual_evidence)
            target = batch["target"].to(device, non_blocking=True)
            loss = gaussian_focal_bce(output.fused_logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if epoch == 1 and batch_index == 0:
                print(
                    json.dumps(
                        {
                            "event": "first_optimizer_step",
                            "baseline": args.baseline,
                            "loss": float(loss.detach()),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "batch_size": int(target.shape[0]),
                        }
                    ),
                    flush=True,
                )
        scheduler.step()
        development = evaluate(
            model,
            anchor,
            args.expert_kind,
            expert,
            visual,
            development_loader,
            device,
            geometry,
            args.max_eval_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
            **development,
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        fusion_metrics = development["fusion"]
        assert isinstance(fusion_metrics, dict)
        ap30 = float(fusion_metrics["AP@0.30"])
        if ap30 > best_ap30 + 1e-6:
            best_ap30 = ap30
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": development,
                    "metadata": metadata,
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    verified = build_public_official_baseline(args.baseline, args.public_source_root).to(device)
    verified.load_state_dict(checkpoint["model"], strict=True)
    report = {
        "baseline": args.baseline,
        "best_epoch": checkpoint["epoch"],
        "independent_reload": evaluate(
            verified,
            anchor,
            args.expert_kind,
            expert,
            visual,
            development_loader,
            device,
            geometry,
            args.max_eval_batches,
        ),
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "elapsed_seconds": time.time() - started,
        "official_test_read": False,
    }
    (args.output_dir / "final_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
