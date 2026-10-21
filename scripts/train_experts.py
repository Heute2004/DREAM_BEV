from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.data import RadarWindowDataset
from radar_bev.dry_complex_training import candidate_ranking_margin_loss
from radar_bev.recovery_training import recovery_loss_terms
from radar_bev.evaluation import evaluate_single_person_predictions, extract_single_peak
from radar_bev.geometry import BEVGeometry
from radar_bev.heterogeneous_training import heterogeneous_loss_terms
from radar_bev.protocol import (
    DEV_SPLIT,
    TRAIN_SPLIT,
    validate_training_manifest,
    validate_training_routing_manifest,
)
from radar_bev.losses import QualityAwareBEVLoss
from radar_bev.models import build_model
from radar_bev.models.recovery_experts import RECOVERY_EXPERT_KINDS
from radar_bev.models.heterogeneous_experts import EXPERT_KINDS
from radar_bev.training import has_converged_early_stop, manifest_sha256, seed_everything


ROUTE_TO_EXPERT = {
    "open_indoor_los": "clor",
    "indoor_clutter": "dcmd",
    "open_outdoor_weak_return": "patei",
    "outdoor_wet_return": "srre",
}

RECOVERY_ROUTE_TO_EXPERT = {
    "indoor_clutter": "cucre",
    "open_outdoor_weak_return": "ltge",
    "outdoor_wet_return": "gg_srre",
}

DRY_COMPLEX_ROUTE_TO_EXPERT = {
    "outdoor_dry_complex": "pb_mhbr",
}

DRY_COMPLEX_EXPERT_KINDS = tuple(DRY_COMPLEX_ROUTE_TO_EXPERT.values())
ALL_ROUTE_TO_EXPERT = {
    **ROUTE_TO_EXPERT,
    **RECOVERY_ROUTE_TO_EXPERT,
    **DRY_COMPLEX_ROUTE_TO_EXPERT,
}


class TaggedDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(self, dataset: RadarWindowDataset, identity_negative: bool) -> None:
        self.dataset = dataset
        self.identity_negative = identity_negative

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        item = dict(self.dataset[index])
        item["is_identity_negative"] = torch.tensor(self.identity_negative, dtype=torch.bool)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one frozen-DREAM route specialist")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--routing-manifest", required=True)
    parser.add_argument("--expert-family", choices=("local", "recovery", "dry_complex"), default="local")
    parser.add_argument(
        "--adapter-route",
        required=True,
        choices=tuple(sorted(ALL_ROUTE_TO_EXPERT)),
    )
    parser.add_argument(
        "--additional-train-route",
        action="append",
        default=[],
        choices=tuple(
            sorted(
                set(ALL_ROUTE_TO_EXPERT)
            )
        ),
        help=(
            "Additional Train-only route pooled into the target expert; "
            "Development Val remains adapter-route only."
        ),
    )
    parser.add_argument(
        "--expert-kind",
        choices=(*EXPERT_KINDS, *RECOVERY_EXPERT_KINDS, *DRY_COMPLEX_EXPERT_KINDS),
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eligibility", choices=("first", "later"), default="later")
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--logit-cap", type=float, default=3.0)
    parser.add_argument("--coordinate-weight", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=0.01)
    parser.add_argument("--residual-weight", type=float, default=1e-4)
    parser.add_argument("--regret-weight", type=float, default=0.05)
    parser.add_argument("--covariance-weight", type=float, default=0.02)
    parser.add_argument("--acceptance-weight", type=float, default=0.10)
    parser.add_argument("--polar-residual-weight", type=float, default=1e-4)
    parser.add_argument("--gain-weight", type=float, default=0.05)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--identity-weight", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.50)
    parser.add_argument("--acceptance-warmup-epochs", type=int, default=3)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--progress-every-batches", type=int, default=200)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ram-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--training-phase", choices=("joint", "recovery", "gain"), default="joint")
    parser.add_argument("--init-expert", help="recovery expert checkpoint used to initialize the current route expert")
    return parser.parse_args()


def _route_sequences(routing_manifest: Path, route: str, split: str) -> set[str]:
    sequences: set[str] = set()
    for raw_line in routing_manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if record.get("adapter_route") != route:
            continue
        if record.get("split", "").strip().lower() != split.lower():
            continue
        if bool(record.get("eligible_for_router_training", False)):
            sequences.add(str(record["sequence_name"]))
    if not sequences:
        raise ValueError(f"No {split} sequences are eligible for route={route!r}")
    return sequences


def _training_route_sequences(
    routing_manifest: Path,
    *,
    primary_route: str,
    additional_routes: tuple[str, ...] = (),
) -> set[str]:
    sequences: set[str] = set()
    for route in (primary_route, *additional_routes):
        sequences.update(
            _route_sequences(routing_manifest, route, TRAIN_SPLIT.lower())
        )
    return sequences


def _negative_sequences(
    routing_manifest: Path,
    excluded_routes: set[str],
    route_count: int,
    ratio: float,
    seed: int,
) -> set[str]:
    if ratio < 0.0:
        raise ValueError("negative_ratio must be non-negative")
    if ratio == 0.0:
        return set()
    candidates: set[str] = set()
    for raw_line in routing_manifest.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if record.get("adapter_route") in excluded_routes or record.get("split", "").strip().lower() != "train":
            continue
        if bool(record.get("eligible_for_router_training", False)):
            candidates.add(str(record["sequence_name"]))
    requested = min(len(candidates), max(1, round(route_count * ratio)))
    return set(random.Random(seed).sample(sorted(candidates), requested))


def _to_device(batch: dict[str, Tensor | str], device: torch.device) -> dict[str, Tensor | str]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _metric_record(output: dict[str, Tensor], batch: dict[str, Tensor | str], durations: list[float]) -> dict[str, Tensor | float]:
    prediction_xy, confidence = extract_single_peak(output["logits"], BEVGeometry())
    base_xy, base_confidence = extract_single_peak(output["base_logits"], BEVGeometry())
    return {
        "prediction_xy": prediction_xy.cpu(),
        "confidence": confidence.cpu(),
        "base_xy": base_xy.cpu(),
        "base_confidence": base_confidence.cpu(),
        "target_xy": batch["target_xy_m"].cpu(),
        "acceptance": output["acceptance"].detach().cpu(),
        "duration": sum(durations),
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: QualityAwareBEVLoss,
    device: torch.device,
    amp: bool,
    acceptance_enabled: bool,
    args: argparse.Namespace,
    max_batches: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    names = ("loss", "localization", "consistency", "residual_energy", "regret", "covariance_nll", "acceptance", "identity")
    if args.expert_family == "recovery":
        names = (*names[:-1], "polar_residual_energy", "gain", "edge", "identity")
    elif args.expert_family == "dry_complex":
        names = (*names, "ranking")
    totals = {name: 0.0 for name in names}
    samples = 0
    prediction_parts: list[Tensor] = []
    confidence_parts: list[Tensor] = []
    base_prediction_parts: list[Tensor] = []
    base_confidence_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    acceptance_parts: list[Tensor] = []
    durations: list[float] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        batch = _to_device(raw_batch, device)
        if not training and device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                output = model(batch["radar"])
                if args.expert_family == "recovery":
                    losses = recovery_loss_terms(
                        output=output,
                        target_xy_m=batch["target_xy_m"],
                        history_target_xy_m=batch["history_target_xy_m"],
                        history_target_valid=batch["history_target_valid"],
                        sample_weight=batch["sample_weight"],
                        identity_mask=batch["is_identity_negative"],
                        criterion=criterion,
                        acceptance_enabled=acceptance_enabled,
                        gain_enabled=args.training_phase == "gain",
                        consistency_weight=args.consistency_weight,
                        residual_weight=args.residual_weight,
                        polar_residual_weight=args.polar_residual_weight,
                        regret_weight=args.regret_weight,
                        covariance_weight=args.covariance_weight,
                        acceptance_weight=args.acceptance_weight,
                        gain_weight=args.gain_weight,
                        edge_weight=args.edge_weight,
                        identity_weight=args.identity_weight,
                    )
                else:
                    losses = heterogeneous_loss_terms(
                        output=output,
                        target_xy_m=batch["target_xy_m"],
                        sample_weight=batch["sample_weight"],
                        identity_mask=batch["is_identity_negative"],
                        criterion=criterion,
                        acceptance_enabled=acceptance_enabled,
                        consistency_weight=args.consistency_weight,
                        residual_weight=args.residual_weight,
                        regret_weight=args.regret_weight,
                        covariance_weight=args.covariance_weight,
                        acceptance_weight=args.acceptance_weight,
                        identity_weight=args.identity_weight,
                    )
                objective = losses.total
                ranking_loss = objective.detach() * 0.0
                if args.expert_family == "dry_complex":
                    ranking = candidate_ranking_margin_loss(
                        output["candidate_selector_logits"],
                        output["candidate_xy_m"],
                        batch["target_xy_m"],
                        margin=args.ranking_margin,
                    )
                    ranking_loss = ranking.loss
                    objective = objective + args.ranking_weight * ranking.loss
            if training:
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.expert_parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if not training and device.type == "cuda":
            torch.cuda.synchronize(device)
        duration = perf_counter() - start
        count = int(batch["radar"].shape[0])
        samples += count
        totals["loss"] += float(objective.detach().item()) * count
        for name in names[1:]:
            value = ranking_loss if name == "ranking" else getattr(losses, name)
            totals[name] += float(value.detach().item()) * count
        if not training:
            record = _metric_record(output, batch, [duration])
            prediction_parts.append(record["prediction_xy"])
            confidence_parts.append(record["confidence"])
            base_prediction_parts.append(record["base_xy"])
            base_confidence_parts.append(record["base_confidence"])
            target_parts.append(record["target_xy"])
            acceptance_parts.append(record["acceptance"])
            durations.append(float(record["duration"]))
        if training and args.progress_every_batches and batch_index % args.progress_every_batches == 0:
            print(json.dumps({"phase": "train", "batch": batch_index, "mean_loss": totals["loss"] / samples}), flush=True)
    result = {name: value / max(samples, 1) for name, value in totals.items()}
    result["samples"] = float(samples)
    if not training:
        score = torch.cat(confidence_parts)
        prediction = torch.cat(prediction_parts)
        target = torch.cat(target_parts)
        base_score = torch.cat(base_confidence_parts)
        base_prediction = torch.cat(base_prediction_parts)
        result.update(evaluate_single_person_predictions(score, prediction, target, distance_thresholds=(0.25, 0.3, 0.4, 0.5, 1.0)))
        result.update({f"base_{key}": value for key, value in evaluate_single_person_predictions(base_score, base_prediction, target, distance_thresholds=(0.25, 0.3, 0.4, 0.5, 1.0)).items()})
        acceptance = torch.cat(acceptance_parts)
        result["acceptance_mean"] = float(acceptance.mean().item())
        result["acceptance_coverage_0p5"] = float((acceptance >= 0.5).float().mean().item())
        result["fallback_rate_0p5"] = 1.0 - result["acceptance_coverage_0p5"]
        result["latency_ms_per_frame"] = 1000.0 * sum(durations) / max(samples, 1)
        result["fps"] = 1000.0 / max(result["latency_ms_per_frame"], 1e-6)
    return result


def main() -> None:
    args = parse_args()
    validate_training_manifest(Path(args.manifest), scope="base")
    validate_training_routing_manifest(Path(args.routing_manifest), scope="base")
    route_to_expert = {
        "local": ROUTE_TO_EXPERT,
        "recovery": RECOVERY_ROUTE_TO_EXPERT,
        "dry_complex": DRY_COMPLEX_ROUTE_TO_EXPERT,
    }[args.expert_family]
    if args.adapter_route not in route_to_expert:
        raise ValueError(f"route={args.adapter_route!r} is not available in expert family={args.expert_family!r}")
    expected_kind = route_to_expert[args.adapter_route]
    if args.expert_kind is not None and args.expert_kind != expected_kind:
        raise ValueError(f"{args.adapter_route} requires expert_kind={expected_kind}, not {args.expert_kind}")
    args.expert_kind = expected_kind
    if args.expert_family != "recovery" and args.training_phase != "joint":
        raise ValueError("--training-phase is only configurable for recovery experts")
    if args.early_stop_patience < 0 or args.acceptance_warmup_epochs < 0:
        raise ValueError("early-stop and acceptance warmup must be non-negative")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    routing_path = Path(args.routing_manifest)
    additional_train_routes = tuple(dict.fromkeys(args.additional_train_route))
    if args.adapter_route in additional_train_routes:
        raise ValueError("additional training routes must not repeat adapter-route")
    route_train_sequences = _training_route_sequences(
        routing_path,
        primary_route=args.adapter_route,
        additional_routes=additional_train_routes,
    )
    route_eval_sequences = _route_sequences(routing_path, args.adapter_route, DEV_SPLIT.lower())
    negative_sequences = _negative_sequences(
        routing_path,
        {args.adapter_route, *additional_train_routes},
        len(route_train_sequences),
        args.negative_ratio,
        args.seed,
    )
    common = dict(
        data_root=Path(args.data_root), manifest_path=Path(args.manifest), clip_length=args.clip_length,
        eligibility=args.eligibility, cache_in_memory=args.ram_cache,
    )
    route_train = RadarWindowDataset(split=TRAIN_SPLIT, allowed_sequences=route_train_sequences, **common)
    train_components: list[Dataset] = [TaggedDataset(route_train, identity_negative=False)]
    if negative_sequences:
        negative_train = RadarWindowDataset(split=TRAIN_SPLIT, allowed_sequences=negative_sequences, **common)
        train_components.append(TaggedDataset(negative_train, identity_negative=True))
    train_set: Dataset = ConcatDataset(train_components)
    eval_set = TaggedDataset(RadarWindowDataset(split=DEV_SPLIT, allowed_sequences=route_eval_sequences, **common), identity_negative=False)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0 if args.ram_cache else args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": (not args.ram_cache) and args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    eval_loader = DataLoader(eval_set, shuffle=False, **loader_options)
    if args.expert_family == "recovery":
        model = build_model(
            "dream_recovery_expert",
            expert_kind=args.expert_kind,
            logit_cap=args.logit_cap,
            candidate_count=args.candidate_count,
        )
    elif args.expert_family == "dry_complex":
        model = build_model("dream_dry_complex_expert", logit_cap=args.logit_cap)
    else:
        model = build_model(
            "dream_heterogeneous_expert",
            expert_kind=args.expert_kind,
            logit_cap=args.logit_cap,
            candidate_count=args.candidate_count,
        )
    base_sha256 = model.load_base_checkpoint(args.base_checkpoint)
    if args.init_expert:
        if args.expert_family != "recovery":
            raise ValueError("--init-expert is only valid for recovery experts")
        model.load_expert_checkpoint(args.init_expert)
    if args.expert_family == "recovery":
        model.set_training_phase(args.training_phase)
    model = model.to(device)
    criterion = QualityAwareBEVLoss(coordinate_weight=args.coordinate_weight).to(device)
    optimizer = torch.optim.AdamW(model.expert_parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "stage": f"route_expert_{args.expert_family}",
        "model": type(model).__name__,
        "expert_family": args.expert_family,
        "training_phase": args.training_phase,
        "adapter_route": args.adapter_route,
        "expert_kind": args.expert_kind,
        "base_model": "DREAM-BEV",
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": base_sha256,
        "manifest_sha256": manifest_sha256(args.manifest),
        "routing_manifest_sha256": hashlib.sha256(routing_path.read_bytes()).hexdigest(),
        "route_train_sequences": sorted(route_train_sequences),
        "primary_train_route": args.adapter_route,
        "additional_train_routes": list(additional_train_routes),
        "identity_negative_sequences": sorted(negative_sequences),
        "route_eval_sequences": sorted(route_eval_sequences),
        "route_train_samples": len(route_train),
        "total_train_samples": len(train_set),
        "eval_samples": len(eval_set),
        "trainable_parameters": sum(parameter.numel() for parameter in model.expert_parameters()),
        "frozen_backbone_parameters": sum(parameter.numel() for parameter in model.backbone.parameters()),
        "checkpoint_selection_metric": "ap_0p3m",
        "evaluation_protocol": "development selects checkpoints; locked test remains unopened",
        "args": vars(args),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    history: list[dict[str, float | int]] = []
    best: dict[str, float | int] = {"ap_0p3m": -1.0}
    for epoch in range(1, args.train_epochs + 1):
        acceptance_enabled = epoch > args.acceptance_warmup_epochs
        if args.expert_family == "recovery" and args.training_phase == "recovery":
            acceptance_enabled = False
        train_metrics = _run_epoch(
            model, train_loader, optimizer, criterion, device, not args.no_amp,
            acceptance_enabled=acceptance_enabled, args=args, max_batches=args.max_train_batches,
        )
        eval_metrics = _run_epoch(
            model, eval_loader, None, criterion, device, not args.no_amp,
            acceptance_enabled=acceptance_enabled, args=args, max_batches=args.max_eval_batches,
        )
        scheduler.step()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"eval_{key}": value for key, value in eval_metrics.items()}}
        history.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
        if float(eval_metrics["ap_0p3m"]) > float(best["ap_0p3m"]):
            best = {"epoch": epoch, **eval_metrics}
            torch.save(
                {
                    "model": type(model).__name__,
                    "expert_family": args.expert_family,
                    "adapter_route": args.adapter_route,
                    "expert_kind": args.expert_kind,
                    "base_checkpoint_sha256": base_sha256,
                    "expert_state": model.expert_state_dict(),
                    "metrics": best,
                },
                output_dir / "best_expert.pt",
            )
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_dir / "best_metrics.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        if epoch < args.train_epochs and has_converged_early_stop([float(item["eval_ap_0p3m"]) for item in history], args.early_stop_patience):
            break
    (output_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "requested_epochs": args.train_epochs,
                "completed_epochs": len(history),
                "best_epoch": best.get("epoch"),
                "best_metrics": best,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()




