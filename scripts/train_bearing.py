from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rv_bearing_net.cached_proposals import (
    CachedProposalBearingDataset,
    validate_cached_proposal_source,
)
from rv_bearing_net.data import RGBBearingClipDataset, SequenceGroupedSampler
from rv_bearing_net.evidence_initialized import (
    EvidenceInitializedRVBearingNet,
)
from rv_bearing_net.losses import RVBearingLoss, compute_rv_bearing_loss
from rv_bearing_net.metrics import bearing_metrics, visual_checkpoint_decision
from rv_bearing_net.model import RVBearingNet
from rv_bearing_net.monotonic_residual import (
    MonotonicResidualRVBearingNet,
)
from rv_bearing_net.proposal_geometry import (
    ProposalGeometryRVBearingNet,
)
from rvfusion.data import ConditionTimeline


def resolve_evidence_backbone_frozen(
    backbone: str,
    *,
    requested: bool,
) -> bool:
    if backbone in ("proposal_geometry_s1", "monotonic_residual_s1"):
        return True
    return bool(requested)


def forward_pair_from_batch(
    model: Any,
    batch: dict[str, object],
) -> tuple[object, object, int]:
    images = batch.get("images")
    if isinstance(images, Tensor):
        previous, current = model.forward_pair(images)
        return previous, current, int(images.shape[0])

    required = {
        "boxes": batch.get("proposal_boxes"),
        "scores": batch.get("proposal_raw_scores"),
        "tokens": batch.get("proposal_raw_tokens"),
        "valid_mask": batch.get("proposal_valid_mask"),
    }
    if not all(isinstance(value, Tensor) for value in required.values()):
        raise TypeError(
            "batch must contain images or aligned proposal cache tensors"
        )
    image_height = batch.get("proposal_image_height")
    image_width = batch.get("proposal_image_width")
    if not isinstance(image_height, Tensor) or not isinstance(
        image_width,
        Tensor,
    ):
        raise TypeError("proposal image dimensions must collate to tensors")
    if not torch.equal(
        image_height,
        torch.full_like(image_height, image_height.flatten()[0]),
    ) or not torch.equal(
        image_width,
        torch.full_like(image_width, image_width.flatten()[0]),
    ):
        raise ValueError("proposal image dimensions must match in a batch")
    forward_cached_pair = getattr(model, "forward_cached_pair", None)
    if forward_cached_pair is None:
        raise TypeError("selected backbone does not accept proposal caches")
    previous, current = forward_cached_pair(
        **required,
        image_height=int(image_height.flatten()[0]),
        image_width=int(image_width.flatten()[0]),
    )
    boxes = required["boxes"]
    assert isinstance(boxes, Tensor)
    return previous, current, int(boxes.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train causal RGB bearing evidence for RVFusion"
    )
    parser.add_argument("--stage", choices=("spatial", "joint"), required=True)
    parser.add_argument(
        "--backbone",
        choices=(
            "scratch",
            "evidence_s1",
            "proposal_geometry_s1",
            "monotonic_residual_s1",
        ),
        default="scratch",
    )
    parser.add_argument("--evidence-checkpoint", type=Path)
    parser.add_argument("--freeze-evidence-backbone", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-teacher-root", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--visual-evidence-cache-root", type=Path)
    parser.add_argument("--training-teacher-cache-root", type=Path)
    parser.add_argument("--condition-zip", type=Path, required=True)
    parser.add_argument(
        "--train-condition-members",
        nargs="+",
        default=["annotations/train_relative.jsonl"],
    )
    parser.add_argument("--eval-condition-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--x-huber-weight", type=float, default=1.0)
    parser.add_argument("--x-huber-beta-m", type=float, default=0.10)
    parser.add_argument("--geometry-teacher-weight", type=float, default=0.0)
    parser.add_argument("--candidate-ranking-weight", type=float, default=0.0)
    parser.add_argument("--candidate-ranking-margin", type=float, default=0.25)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=448)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--preload-rgb", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-validation-batches", type=int, default=0)
    parser.add_argument(
        "--excluded-sequences",
        nargs="*",
        default=["seq_113", "seq_148"],
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training_teacher_cache(
    cache_root: Path, manifest: Path
) -> dict[str, object]:
    cache_manifest = cache_root / "cache_manifest.json"
    if not cache_manifest.is_file():
        raise FileNotFoundError(f"teacher cache manifest not found: {cache_manifest}")
    payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("training_only_teacher") is not True:
        raise ValueError("teacher cache must declare training_only_teacher=true")
    if payload.get("runtime_inputs") != ["causal_left_rgb"]:
        raise ValueError("teacher cache changes the runtime input contract")
    expected_sha = sha256(manifest)
    if payload.get("manifest_sha256") != expected_sha:
        raise ValueError("teacher cache source manifest does not match training manifest")
    return {
        "cache_version": payload.get("cache_version"),
        "manifest_sha256": expected_sha,
        "cache_manifest": str(cache_manifest),
        "cache_manifest_sha256": sha256(cache_manifest),
        "sequence_count": payload.get("sequence_count"),
        "frame_count": payload.get("frame_count"),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _move_batch(
    batch: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _loss_scalars(loss: RVBearingLoss) -> dict[str, float]:
    return {
        "loss": float(loss.total.detach().cpu()),
        "bearing_distribution": float(
            loss.bearing_distribution.detach().cpu()
        ),
        "theta_huber": float(loss.theta_huber.detach().cpu()),
        "x_huber": float(loss.x_huber.detach().cpu()),
        "uncertainty_nll": float(loss.uncertainty_nll.detach().cpu()),
        "person_u": float(loss.person_u.detach().cpu()),
        "reliability": float(loss.reliability.detach().cpu()),
        "causal_velocity": float(loss.causal_velocity.detach().cpu()),
        "geometry_teacher": float(loss.geometry_teacher.detach().cpu()),
        "candidate_ranking": float(loss.candidate_ranking.detach().cpu()),
    }


def run_epoch(
    *,
    model: RVBearingNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    amp: bool,
    temporal_weight: float,
    x_huber_weight: float,
    x_huber_beta_m: float,
    geometry_teacher_weight: float,
    candidate_ranking_weight: float,
    candidate_ranking_margin: float,
    max_batches: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    sample_count = 0
    predicted_theta: list[Tensor] = []
    target_theta: list[Tensor] = []
    target_x: list[Tensor] = []
    radar_y_proxy: list[Tensor] = []
    eligible: list[Tensor] = []
    reliability: list[Tensor] = []
    visual_condition_eligible: list[Tensor] = []

    for batch_index, raw_batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            previous, current, batch_size = forward_pair_from_batch(
                model,
                batch,
            )
            loss = compute_rv_bearing_loss(
                current=current,
                previous=previous,
                batch=batch,  # type: ignore[arg-type]
                angle_centers=model.angle_centers,
                temporal_weight=temporal_weight,
                x_huber_weight=x_huber_weight,
                x_huber_beta_m=x_huber_beta_m,
                geometry_teacher_weight=geometry_teacher_weight,
                candidate_ranking_weight=candidate_ranking_weight,
                candidate_ranking_margin=candidate_ranking_margin,
            )
        if training:
            loss.total.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        sample_count += batch_size
        for name, value in _loss_scalars(loss).items():
            totals[name] = totals.get(name, 0.0) + value * batch_size

        xy_history = batch["target_xy_history_m"]
        theta_history = batch["target_theta_history"]
        target_valid = batch["target_valid_history"]
        coordinate_weight = batch["coordinate_weight"]
        if not all(
            isinstance(value, Tensor)
            for value in (
                xy_history,
                theta_history,
                target_valid,
                coordinate_weight,
            )
        ):
            raise TypeError("target fields must collate to tensors")
        current_eligible = (
            target_valid[:, -1].bool()
            & current.valid
            & (coordinate_weight > 0.0)
        )
        predicted_theta.append(current.theta[:, 0].detach().float().cpu())
        target_theta.append(theta_history[:, -1].detach().float().cpu())
        target_x.append(xy_history[:, -1, 0].detach().float().cpu())
        radar_y_proxy.append(xy_history[:, -1, 1].detach().float().cpu())
        eligible.append(current_eligible.detach().cpu())
        reliability.append(
            current.reliability[:, 0].detach().float().cpu()
        )
        visual_condition_eligible.append(
            (coordinate_weight > 0.0).detach().cpu()
        )

    if sample_count == 0:
        raise RuntimeError("loader produced no batches")
    averaged = {
        name: value / sample_count for name, value in totals.items()
    }
    averaged.update(
        bearing_metrics(
            predicted_theta=torch.cat(predicted_theta),
            target_theta=torch.cat(target_theta),
            radar_y_m=torch.cat(radar_y_proxy),
            target_x_m=torch.cat(target_x),
            eligible=torch.cat(eligible),
            reliability=torch.cat(reliability),
            visual_condition_eligible=torch.cat(
                visual_condition_eligible
            ),
        )
    )
    return averaged


def _checkpoint_payload(
    *,
    model: RVBearingNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> dict[str, object]:
    return {
        "model": "rv_bearing_net",
        "model_state": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "metadata": metadata,
    }


def _selection_score(metrics: dict[str, float]) -> float:
    x_mae = metrics.get("x_mae_m", float("inf"))
    bearing_p90 = metrics.get("bearing_p90_deg", float("inf"))
    if not math.isfinite(x_mae) or not math.isfinite(bearing_p90):
        return float("inf")
    return float(x_mae + 0.01 * bearing_p90)


def train_model(
    *,
    model: RVBearingNet,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temporal_weight: float,
    x_huber_weight: float,
    x_huber_beta_m: float,
    amp: bool,
    patience: int,
    max_train_batches: int,
    max_validation_batches: int,
    metadata: dict[str, Any],
    geometry_teacher_weight: float = 0.0,
    candidate_ranking_weight: float = 0.0,
    candidate_ranking_margin: float = 0.25,
) -> list[dict[str, object]]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    _atomic_json(output_dir / "run_metadata.json", metadata)
    history: list[dict[str, object]] = []
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            amp=amp,
            temporal_weight=temporal_weight,
            x_huber_weight=x_huber_weight,
            x_huber_beta_m=x_huber_beta_m,
            geometry_teacher_weight=geometry_teacher_weight,
            candidate_ranking_weight=candidate_ranking_weight,
            candidate_ranking_margin=candidate_ranking_margin,
            max_batches=max_train_batches,
        )
        with torch.inference_mode():
            validation_metrics = run_epoch(
                model=model,
                loader=validation_loader,
                device=device,
                optimizer=None,
                amp=amp,
                temporal_weight=temporal_weight,
                x_huber_weight=x_huber_weight,
                x_huber_beta_m=x_huber_beta_m,
                geometry_teacher_weight=geometry_teacher_weight,
                candidate_ranking_weight=candidate_ranking_weight,
                candidate_ranking_margin=candidate_ranking_margin,
                max_batches=max_validation_batches,
            )
        score = _selection_score(validation_metrics)
        row: dict[str, object] = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            "selection_score": score,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=validation_metrics,
            metadata=metadata,
        )
        _atomic_torch_save(output_dir / "last.pt", checkpoint)
        if best_epoch == 0 or score < best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            _atomic_torch_save(output_dir / "best.pt", checkpoint)
        else:
            stale_epochs += 1
        _atomic_json(output_dir / "history.json", history)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "val_loss": validation_metrics["loss"],
                    "val_bearing_mae_deg": validation_metrics[
                        "bearing_mae_deg"
                    ],
                    "val_x_mae_m_target_y_proxy": validation_metrics[
                        "x_mae_m"
                    ],
                    "best_epoch": best_epoch,
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stale_epochs >= patience:
            break

    best_payload = torch.load(
        output_dir / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    best_metrics = best_payload["metrics"]
    completion = {
        "status": "complete",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_metrics": best_metrics,
        "acceptance": visual_checkpoint_decision(best_metrics),
        "metric_note": (
            "x metrics use frozen Final-BEV target y as a training-time "
            "proxy; formal fusion acceptance recomputes x with frozen radar y"
        ),
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(output_dir / "training_complete.json", completion)
    return history


def _load_initial_checkpoint(
    model: RVBearingNet,
    path: Path | None,
) -> None:
    if path is None:
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("model_state", payload), strict=True)


def _build_loader(
    dataset: RGBBearingClipDataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    training: bool,
) -> DataLoader:
    sampler = SequenceGroupedSampler(
        dataset,
        seed=seed,
        shuffle_sequences=training,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=training,
    )


def main() -> None:
    args = parse_args()
    if (
        args.geometry_teacher_weight > 0.0
        or args.candidate_ranking_weight > 0.0
    ) and args.training_teacher_cache_root is None:
        raise ValueError(
            "teacher loss weights require --training-teacher-cache-root"
        )
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    train_timeline = ConditionTimeline.from_zip(
        args.condition_zip,
        args.train_condition_members,
    )
    common = {
        "data_root": args.data_root,
        "manifest_path": args.manifest,
        "visual_teacher_root": args.visual_teacher_root,
        "visual_manifest_path": args.visual_manifest,
        "condition_timeline": train_timeline,
        "clip_length": args.clip_length,
        "frame_stride": args.frame_stride,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "excluded_sequences": tuple(args.excluded_sequences),
        "preload_rgb": args.preload_rgb,
        "training_teacher_cache_root": args.training_teacher_cache_root,
    }
    proposal_cache_contract: dict[str, object] | None = None
    teacher_cache_contract = (
        validate_training_teacher_cache(
            args.training_teacher_cache_root, args.manifest
        )
        if args.training_teacher_cache_root is not None
        else None
    )
    dataset_class = RGBBearingClipDataset
    dataset_extra: dict[str, object] = {}
    if args.visual_evidence_cache_root is not None:
        if args.backbone not in (
            "proposal_geometry_s1",
            "monotonic_residual_s1",
        ):
            raise ValueError(
                "frozen proposal caches require a proposal backbone"
            )
        if args.evidence_checkpoint is None:
            raise ValueError(
                "frozen proposal caches require --evidence-checkpoint"
            )
        proposal_cache_contract = validate_cached_proposal_source(
            cache_root=args.visual_evidence_cache_root,
            checkpoint=args.evidence_checkpoint,
            image_height=args.image_height,
            image_width=args.image_width,
        )
        dataset_class = CachedProposalBearingDataset
        dataset_extra["visual_evidence_cache_root"] = (
            args.visual_evidence_cache_root
        )
    train_dataset = dataset_class(
        partition="Train",
        train_augmentation="mild_photometric",
        **dataset_extra,
        **common,
    )
    validation_dataset = dataset_class(
        partition="InternalVal",
        train_augmentation="none",
        **dataset_extra,
        **common,
    )
    train_loader = _build_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        training=True,
    )
    validation_loader = _build_loader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        training=False,
    )
    if args.backbone == "evidence_s1":
        model = EvidenceInitializedRVBearingNet()
        if args.evidence_checkpoint is None:
            raise ValueError(
                "evidence_s1 backbone requires --evidence-checkpoint"
            )
        evidence_payload = torch.load(
            args.evidence_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_evidence_state(evidence_payload)
        model.freeze_evidence_backbone(args.freeze_evidence_backbone)
    elif args.backbone in (
        "proposal_geometry_s1",
        "monotonic_residual_s1",
    ):
        if args.evidence_checkpoint is None:
            raise ValueError(
                f"{args.backbone} backbone requires --evidence-checkpoint"
            )
        evidence_payload = torch.load(
            args.evidence_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        evidence_contract = evidence_payload.get("contract", {})
        if not isinstance(evidence_contract, dict):
            raise TypeError("evidence checkpoint contract must be a mapping")
        model_class = (
            MonotonicResidualRVBearingNet
            if args.backbone == "monotonic_residual_s1"
            else ProposalGeometryRVBearingNet
        )
        model = model_class(
            evidence_width=float(evidence_contract.get("width", 1.25)),
            evidence_token_dim=int(
                evidence_contract.get("token_dim", 128)
            ),
            evidence_top_k=int(evidence_contract.get("top_k", 3)),
            evidence_quality_head=False,
        )
        model.load_evidence_state(evidence_payload)
    else:
        model = RVBearingNet()
    _load_initial_checkpoint(model, args.initial_checkpoint)
    metadata: dict[str, Any] = {
        "model": "RV-BearingNet",
        "backbone": args.backbone,
        "evidence_checkpoint": (
            str(args.evidence_checkpoint)
            if args.evidence_checkpoint is not None
            else None
        ),
        "evidence_checkpoint_sha256": (
            sha256(args.evidence_checkpoint)
            if args.evidence_checkpoint is not None
            else None
        ),
        "evidence_backbone_frozen": resolve_evidence_backbone_frozen(
            args.backbone,
            requested=args.freeze_evidence_backbone,
        ),
        "proposal_contract": (
            model.proposal_contract
            if isinstance(model, ProposalGeometryRVBearingNet)
            else None
        ),
        "stage": args.stage,
        "runtime_inputs": ["causal_left_rgb"],
        "training_input_source": (
            "checkpoint_bound_frozen_proposal_cache"
            if args.visual_evidence_cache_root is not None
            else "causal_left_rgb"
        ),
        "proposal_cache_contract": proposal_cache_contract,
        "training_teacher_cache_contract": teacher_cache_contract,
        "prohibited_runtime_inputs": [
            "depth",
            "right_rgb",
            "radar",
            "ground_truth",
            "future_frames",
        ],
        "target": (
            "bearing_with_range_aware_x_geometry_teacher_and_radar_candidate_ranking"
            if args.training_teacher_cache_root is not None
            else "bearing_with_training_only_range_aware_x_loss"
        ),
        "runtime_range_prediction": False,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_sequences": sorted(train_dataset.sequences),
        "validation_sequences": sorted(validation_dataset.sequences),
        "excluded_sequences": list(args.excluded_sequences),
        "clip_length": args.clip_length,
        "frame_stride": args.frame_stride,
        "image_size": [args.image_height, args.image_width],
        "parameter_count": model.trainable_parameter_count(),
        "seed": args.seed,
        "device": str(device),
        "amp": not args.no_amp,
        "preload_rgb": args.preload_rgb,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "visual_manifest": str(args.visual_manifest),
        "visual_manifest_sha256": sha256(args.visual_manifest),
        "condition_zip": str(args.condition_zip),
        "condition_zip_sha256": sha256(args.condition_zip),
        "condition_members": args.train_condition_members,
        "initial_checkpoint": (
            str(args.initial_checkpoint)
            if args.initial_checkpoint is not None
            else None
        ),
        "x_huber_weight": args.x_huber_weight,
        "x_huber_beta_m": args.x_huber_beta_m,
        "geometry_teacher_weight": args.geometry_teacher_weight,
        "candidate_ranking_weight": args.candidate_ranking_weight,
        "candidate_ranking_margin": args.candidate_ranking_margin,
        "evaluation_contract": (
            "checkpoint selection uses Train-internal sequence validation; "
            "formal held-out evaluation is separate"
        ),
    }
    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        output_dir=args.output_dir,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temporal_weight=0.0 if args.stage == "spatial" else 0.10,
        x_huber_weight=args.x_huber_weight,
        x_huber_beta_m=args.x_huber_beta_m,
        geometry_teacher_weight=args.geometry_teacher_weight,
        candidate_ranking_weight=args.candidate_ranking_weight,
        candidate_ranking_margin=args.candidate_ranking_margin,
        amp=not args.no_amp,
        patience=args.patience,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()



