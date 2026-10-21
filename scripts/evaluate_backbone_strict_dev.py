"""Strict Development-only reload evaluation for frozen radar backbones."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_path in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from radar_bev.data import RadarWindowDataset
from radar_bev.frozen_backbone_evaluation import evaluate_frozen_backbone
from radar_bev.models import build_model
from radar_bev.protocol import DEV_SPLIT, validate_training_manifest
from radar_bev.public_full_adapted import PublicFullPolarBEVAdapter
from radar_bev.training import manifest_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=(
            "dream_bev",
            "erodnet_core",
            "trodnet_core",
            "t_rodnet_compact",
            "rodnet_cdc",
            "rodnet_hg",
            "fcn8s",
            "e_rodnet",
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--latency-warmup-batches", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model in {"dream_bev", "erodnet_core", "trodnet_core"}:
        return build_model(args.model)
    if args.source_root is None:
        raise ValueError(f"--source-root is required for {args.model}")
    return PublicFullPolarBEVAdapter(args.model, args.source_root)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {args.output_dir}")
    validate_training_manifest(args.manifest, scope="base")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if str(payload.get("model", "")).lower() != args.model:
        raise ValueError("checkpoint model does not match --model")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_model(args).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    dataset = RadarWindowDataset(
        args.data_root,
        args.manifest,
        DEV_SPLIT,
        clip_length=16,
        eligibility="later",
        cache_in_memory=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    result = evaluate_frozen_backbone(
        model,
        loader,
        device,
        latency_warmup_batches=args.latency_warmup_batches,
    )

    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "development_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.rows[0]))
        writer.writeheader()
        writer.writerows(result.rows)
    report = {
        "protocol": "private_train_development_backbone_comparison",
        "evaluation_split": DEV_SPLIT,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256(args.manifest),
        "frames": len(result.rows),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": result.metrics,
        "runtime": result.runtime,
        "predictions_csv": str(predictions_path),
        "locked_test_read": False,
    }
    report_path = args.output_dir / "strict_development_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
