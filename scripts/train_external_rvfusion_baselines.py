from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.evaluation import (  # noqa: E402
    evaluate_grouped_single_person_predictions,
    evaluate_single_person_predictions,
    extract_single_peak,
)
from radar_bev.losses import QualityAwareBEVLoss  # noqa: E402
from rvfusion.contracts import VisualEvidence  # noqa: E402
from rvfusion.data import ConditionTimeline, RVFusionCachedDataset, SequenceGroupedSampler  # noqa: E402
from rvfusion.public_official_fusion import build_public_official_baseline  # noqa: E402
from rvfusion.routed_radar_evidence import RoutedRadarEvidenceAdapter  # noqa: E402
from rvfusion.training import condition_from_collated  # noqa: E402
from scripts.train_fusion import (  # noqa: E402
    load_radar_adapter,
    route_expert_paths_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train source-backed public Radar-RGB fusion baselines")
    parser.add_argument("--baseline", choices=("rcrosnet", "crn", "rcbevdet"), required=True)
    parser.add_argument("--public-source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-cache-root", type=Path, required=True)
    conditions = parser.add_mutually_exclusive_group(required=True)
    conditions.add_argument("--condition-jsonl", nargs="+", type=Path)
    conditions.add_argument("--condition-predictions", nargs="+", type=Path)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-open-indoor-los", type=Path)
    parser.add_argument("--expert-indoor-clutter", type=Path)
    parser.add_argument("--expert-open-outdoor-weak-return", type=Path)
    parser.add_argument("--expert-outdoor-dry-complex", type=Path)
    parser.add_argument("--expert-outdoor-wet-return", type=Path)
    parser.add_argument("--gg-recovery-scale", type=float, default=1.15)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--radar-clip-length", type=int, default=16)
    parser.add_argument("--visual-clip-length", type=int, default=8)
    parser.add_argument("--sequence-cache-size", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
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


def visual_from_batch(batch: dict[str, object], device: torch.device) -> VisualEvidence:
    return VisualEvidence(
        boxes=batch["visual_boxes"].to(device, non_blocking=True),  # type: ignore[union-attr]
        raw_scores=batch["visual_raw_scores"].to(device, non_blocking=True),  # type: ignore[union-attr]
        raw_tokens=batch["visual_raw_tokens"].to(device, non_blocking=True),  # type: ignore[union-attr]
        temporal_tokens=batch["visual_temporal_tokens"].to(device, non_blocking=True),  # type: ignore[union-attr]
        temporal_support=batch["visual_temporal_support"].to(device, non_blocking=True),  # type: ignore[union-attr]
        valid_mask=batch["visual_valid_mask"].to(device, non_blocking=True),  # type: ignore[union-attr]
    )


def build_model(name: str, public_source_root: Path) -> nn.Module:
    return build_public_official_baseline(name, public_source_root)


def model_output(model: nn.Module, radar, visual):
    output = model(radar, visual)
    if hasattr(output, "fused_logits"):
        return output
    raise TypeError(f"unexpected baseline output: {type(output)!r}")


def run_epoch(
    model: nn.Module,
    radar: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    radar.eval()
    if isinstance(radar, RoutedRadarEvidenceAdapter):
        radar.reset_causal_state()
    loss_fn = QualityAwareBEVLoss().to(device)
    scores: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    routes: list[str] = []
    total_loss = 0.0
    samples = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        radar_window = batch["radar"].to(device, non_blocking=True)  # type: ignore[union-attr]
        target_xy = batch["target_xy_m"].to(device, non_blocking=True)  # type: ignore[union-attr]
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)  # type: ignore[union-attr]
        visual = visual_from_batch(batch, device)
        condition = condition_from_collated(batch["condition"])  # type: ignore[arg-type]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            if isinstance(radar, RoutedRadarEvidenceAdapter):
                sequences = tuple(str(value) for value in batch["sequence"])
                frame_ids = tuple(int(value) for value in batch["frame_id"].detach().cpu().tolist())  # type: ignore[union-attr]
                evidence = radar(
                    radar_window,
                    condition.global_route,
                    sequences=sequences,
                    frame_ids=frame_ids,
                )
            else:
                evidence = radar(radar_window)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model_output(model, evidence, visual)
            loss = loss_fn(output.fused_logits, target_xy, sample_weight)
        if training:
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if batch_index == 0:
                print(
                    json.dumps(
                        {
                            "event": "first_optimizer_step",
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "loss": float(loss.total.detach()),
                            "batch_size": int(target_xy.shape[0]),
                        }
                    ),
                    flush=True,
                )
        count = int(target_xy.shape[0])
        samples += count
        total_loss += float(loss.total.detach()) * count
        prediction, score = extract_single_peak(output.fused_logits.detach())
        predictions.append(prediction.cpu())
        scores.append(score.cpu())
        targets.append(target_xy.detach().cpu())
        routes.extend(condition.global_route)
    if not samples:
        raise ValueError("epoch produced no samples")
    score_tensor = torch.cat(scores)
    prediction_tensor = torch.cat(predictions)
    target_tensor = torch.cat(targets)
    metric = evaluate_single_person_predictions(
        score_tensor,
        prediction_tensor,
        target_tensor,
        distance_thresholds=(0.25, 0.30, 0.40, 0.50),
    )
    grouped = evaluate_grouped_single_person_predictions(
        score_tensor,
        prediction_tensor,
        target_tensor,
        routes,
        distance_thresholds=(0.25, 0.30, 0.40, 0.50),
    )
    return {"loss": total_loss / samples, "samples": samples, **metric, "per_route": grouped}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if args.condition_predictions is not None:
        timeline = ConditionTimeline.from_predictions_jsonl(args.condition_predictions)
        condition_paths = args.condition_predictions
        condition_source_format = "materialized_policy_predictions"
    else:
        timeline = ConditionTimeline.from_jsonl(args.condition_jsonl)
        condition_paths = args.condition_jsonl
        condition_source_format = "annotation"
    common = dict(
        data_root=args.data_root,
        manifest_path=args.manifest,
        visual_cache_root=args.visual_cache_root,
        condition_timeline=timeline,
        radar_clip_length=args.radar_clip_length,
        visual_clip_length=args.visual_clip_length,
        sequence_cache_size=args.sequence_cache_size,
    )
    train_set = RVFusionCachedDataset(split="Train", **common)
    eval_set = RVFusionCachedDataset(split="Development Val", **common)
    train_sequences = sorted(str(value) for value in train_set.sequences)
    eval_sequences = sorted(str(value) for value in eval_set.sequences)
    overlap = sorted(set(train_sequences).intersection(eval_sequences))
    if overlap:
        raise ValueError(f"sequence leakage between Train and Development Val: {overlap}")
    train_sampler = SequenceGroupedSampler(train_set, seed=args.seed, shuffle_sequences=True)
    eval_sampler = SequenceGroupedSampler(eval_set, seed=args.seed, shuffle_sequences=False)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_args)
    eval_loader = DataLoader(eval_set, sampler=eval_sampler, **loader_args)
    expert_paths = route_expert_paths_from_args(args)
    radar = load_radar_adapter(
        args.base_checkpoint,
        device,
        expert_paths=expert_paths,
        gg_recovery_scale=args.gg_recovery_scale,
    )
    model = build_model(args.baseline, args.public_source_root).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {
        "stage": "source_backed_public_fusion_baseline_development",
        "baseline": args.baseline,
        "architecture_source": getattr(model, "source_metadata", None),
        "task_adaptation": "single RGB evidence + RA/RA-MTI frozen DREAM-BEV evidence -> single-person 2D BEV",
        "official_3d_reproduction": False,
        "radar_anchor": "condition-routed expert-radar" if expert_paths is not None else "raw DREAM-BEV",
        "radar_experts_used": expert_paths is not None,
        "route_expert_checkpoints": (
            {
                "open_indoor_los": {
                    "path": str(expert_paths.open_indoor_los.resolve()),
                    "sha256": sha256(expert_paths.open_indoor_los),
                },
                "indoor_clutter": {
                    "path": str(expert_paths.indoor_clutter.resolve()),
                    "sha256": sha256(expert_paths.indoor_clutter),
                },
                "open_outdoor_weak_return": {
                    "path": str(expert_paths.open_outdoor_weak_return.resolve()),
                    "sha256": sha256(expert_paths.open_outdoor_weak_return),
                },
                "outdoor_dry_complex": {
                    "path": str(expert_paths.outdoor_dry_complex.resolve()),
                    "sha256": sha256(expert_paths.outdoor_dry_complex),
                },
                "outdoor_wet_return": {
                    "path": str(expert_paths.outdoor_wet_return.resolve()),
                    "sha256": sha256(expert_paths.outdoor_wet_return),
                },
            }
            if expert_paths is not None
            else None
        ),
        "gg_recovery_scale": args.gg_recovery_scale,
        "vlm_used": False,
        "depth_input": False,
        "range_preserving_constraint": False,
        "candidate_lattice": False,
        "identity_fallback": False,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "manifest": str(args.manifest.resolve()),
        "condition_source_format": condition_source_format,
        "condition_sources": [str(path.resolve()) for path in condition_paths],
        "visual_cache_root": str(args.visual_cache_root.resolve()),
        "split": "Train / Development Val only; Locked Test unopened",
        "train_sequences": train_sequences,
        "development_sequences": eval_sequences,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    history: list[dict[str, object]] = []
    best = -1.0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_metric = run_epoch(model, radar, train_loader, device, optimizer, args.max_train_batches)
        with torch.no_grad():
            eval_metric = run_epoch(model, radar, eval_loader, device, None, args.max_eval_batches)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train": train_metric,
            "development": eval_metric,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        current = float(eval_metric["ap_0p3m"])
        if current > best:
            best = current
            stale = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": eval_metric, "metadata": metadata}, args.output_dir / "best.pt")
            (args.output_dir / "best_metrics.json").write_text(json.dumps({"epoch": epoch, **eval_metric}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if stale >= 3:
            break
    (args.output_dir / "training_complete.json").write_text(json.dumps({"epochs": len(history), "best_ap_0p3m": best, "locked_test_used": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
