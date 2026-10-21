from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from radar_bev.data import RadarWindowDataset
from radar_bev.evaluation import (
    evaluate_grouped_single_person_predictions,
    evaluate_single_person_predictions,
    extract_single_peak,
)
from radar_bev.geometry import BEVGeometry
from radar_bev.losses import QualityAwareBEVLoss
from radar_bev.models import build_model
from radar_bev.protocol import (
    DEV_SPLIT,
    TRAIN_SPLIT,
    validate_training_manifest,
    validate_training_routing_manifest,
)
from radar_bev.route_closure import (
    assert_development_only_paths,
    select_route_sequences,
    universal_capacity_profiles,
)
from radar_bev.training import has_converged_early_stop, manifest_sha256, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one route-agnostic homogeneous DREAM-BEV expert"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--routing-manifest", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--capacity-profile",
        choices=tuple(universal_capacity_profiles()),
        required=True,
    )
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coordinate-weight", type=float, default=0.05)
    parser.add_argument("--consistency-weight", type=float, default=0.01)
    parser.add_argument("--residual-weight", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ram-cache", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _to_device(batch: dict[str, Tensor | str], device: torch.device) -> dict[str, Tensor | str]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _loss(
    output: dict[str, Tensor],
    batch: dict[str, Tensor | str],
    criterion: QualityAwareBEVLoss,
    consistency_weight: float,
    residual_weight: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    localization = criterion(
        output["logits"], batch["target_xy_m"], batch["sample_weight"]
    )
    consistency = torch.nn.functional.mse_loss(
        output["logits"].sigmoid(), output["base_logits"].sigmoid()
    )
    residual = output["polar_adapter_residual"].square().mean()
    residual = residual + output["bev_adapter_residual"].square().mean()
    total = (
        localization.total
        + consistency_weight * consistency
        + residual_weight * residual
    )
    return total, {
        "heatmap": localization.heatmap.detach(),
        "coordinate": localization.coordinate.detach(),
        "consistency": consistency.detach(),
        "residual": residual.detach(),
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: QualityAwareBEVLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    *,
    amp: bool,
    consistency_weight: float,
    residual_weight: float,
    sequence_routes: dict[str, str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    training = optimizer is not None
    model.train(training)
    totals = defaultdict(float)
    samples = 0
    scores: list[Tensor] = []
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    groups: list[str] = []
    rows: list[dict[str, object]] = []
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                output = model(batch["radar"])
                total, parts = _loss(
                    output,
                    batch,
                    criterion,
                    consistency_weight,
                    residual_weight,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.expert_parameters(), 5.0)
                optimizer.step()
        count = int(batch["radar"].shape[0])
        samples += count
        totals["loss"] += float(total.detach().item()) * count
        for name, value in parts.items():
            totals[name] += float(value.item()) * count
        if training:
            continue
        prediction, confidence = extract_single_peak(output["logits"], BEVGeometry())
        prediction = prediction.detach().cpu()
        confidence = confidence.detach().cpu()
        target = batch["target_xy_m"].detach().cpu()
        sequences = [str(value) for value in batch["sequence"]]
        frame_ids = batch["frame_id"].detach().cpu().tolist()
        scores.append(confidence)
        predictions.append(prediction)
        targets.append(target)
        groups.extend(sequence_routes[sequence] for sequence in sequences)
        for index, sequence in enumerate(sequences):
            rows.append(
                {
                    "sequence": sequence,
                    "frame_id": int(frame_ids[index]),
                    "route": sequence_routes[sequence],
                    "target_x_m": float(target[index, 0]),
                    "target_y_m": float(target[index, 1]),
                    "prediction_x_m": float(prediction[index, 0]),
                    "prediction_y_m": float(prediction[index, 1]),
                    "score": float(confidence[index]),
                }
            )
    result: dict[str, object] = {
        name: value / max(samples, 1) for name, value in totals.items()
    }
    result["samples"] = samples
    if not training:
        all_scores = torch.cat(scores)
        all_predictions = torch.cat(predictions)
        all_targets = torch.cat(targets)
        result.update(
            evaluate_single_person_predictions(
                all_scores,
                all_predictions,
                all_targets,
                distance_thresholds=(0.25, 0.30, 0.50),
            )
        )
        result["per_route"] = evaluate_grouped_single_person_predictions(
            all_scores,
            all_predictions,
            all_targets,
            groups,
            distance_thresholds=(0.25, 0.30, 0.50),
        )
    return result, rows


def _sequence_routes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        sequence = str(record.get("sequence_name", ""))
        route = str(record.get("adapter_route", ""))
        if sequence and route:
            result[sequence] = route
    return result


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty prediction ledger")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    assert_development_only_paths(args.output_dir)
    if args.early_stop_patience < 0:
        raise ValueError("early-stop patience must be non-negative")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    validate_training_manifest(args.manifest)
    validate_training_routing_manifest(args.routing_manifest)
    routing_path = Path(args.routing_manifest)
    train_sequences = select_route_sequences(routing_path, TRAIN_SPLIT)
    dev_sequences = select_route_sequences(routing_path, DEV_SPLIT)
    sequence_routes = _sequence_routes(routing_path)
    profile = universal_capacity_profiles()[args.capacity_profile]
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    common = {
        "data_root": Path(args.data_root),
        "manifest_path": Path(args.manifest),
        "clip_length": args.clip_length,
        "eligibility": "later",
        "cache_in_memory": args.ram_cache,
    }
    train_set = RadarWindowDataset(
        split=TRAIN_SPLIT, allowed_sequences=train_sequences, **common
    )
    dev_set = RadarWindowDataset(
        split=DEV_SPLIT, allowed_sequences=dev_sequences, **common
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0 if args.ram_cache else args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": (not args.ram_cache) and args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    dev_loader = DataLoader(dev_set, shuffle=False, **loader_options)
    model = build_model(
        "dream_route_expert",
        polar_detail_channels=profile["polar_detail_channels"],
        bev_detail_channels=profile["bev_detail_channels"],
    )
    base_sha = model.load_base_checkpoint(args.base_checkpoint)
    model = model.to(device)
    trainable = sum(parameter.numel() for parameter in model.expert_parameters())
    if trainable != profile["trainable_parameters"]:
        raise RuntimeError(
            f"capacity profile mismatch: model={trainable}, profile={profile['trainable_parameters']}"
        )
    criterion = QualityAwareBEVLoss(
        coordinate_weight=args.coordinate_weight
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.expert_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_epochs
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment": "route_agnostic_universal_expert_control",
        "locked_test_used": False,
        "capacity_profile": args.capacity_profile,
        "capacity": profile,
        "trainable_parameters": trainable,
        "train_sequences": sorted(train_sequences),
        "development_sequences": sorted(dev_sequences),
        "train_samples": len(train_set),
        "development_samples": len(dev_set),
        "manifest_sha256": manifest_sha256(args.manifest),
        "routing_manifest_sha256": hashlib.sha256(routing_path.read_bytes()).hexdigest(),
        "base_checkpoint_sha256": base_sha,
        "args": vars(args),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    history: list[dict[str, object]] = []
    best_ap = -1.0
    best_epoch = 0
    for epoch in range(1, args.train_epochs + 1):
        train_metrics, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            amp=not args.no_amp,
            consistency_weight=args.consistency_weight,
            residual_weight=args.residual_weight,
            sequence_routes=sequence_routes,
        )
        dev_metrics, dev_rows = _run_epoch(
            model,
            dev_loader,
            criterion,
            device,
            None,
            amp=not args.no_amp,
            consistency_weight=args.consistency_weight,
            residual_weight=args.residual_weight,
            sequence_routes=sequence_routes,
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "development": dev_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=True), flush=True)
        current_ap = float(dev_metrics["ap_0p3m"])
        if current_ap > best_ap:
            best_ap = current_ap
            best_epoch = epoch
            torch.save(
                {
                    "model": "DREAMRouteExpertBEV",
                    "capacity_profile": args.capacity_profile,
                    "capacity": profile,
                    "base_checkpoint_sha256": base_sha,
                    "expert_state": model.expert_state_dict(),
                    "metrics": dev_metrics,
                },
                output_dir / "best_universal_expert.pt",
            )
            (output_dir / "best_development_metrics.json").write_text(
                json.dumps(dev_metrics, indent=2), encoding="utf-8"
            )
            _write_rows(output_dir / "best_development_predictions.csv", dev_rows)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epoch < args.train_epochs and has_converged_early_stop(
            [float(item["development"]["ap_0p3m"]) for item in history],
            args.early_stop_patience,
        ):
            break
    completion = {
        "completed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_ap_0p3m": best_ap,
    }
    (output_dir / "training_complete.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
