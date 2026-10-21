from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rvfusion.data import ConditionTimeline, RVFusionCachedDataset, SequenceGroupedSampler
from rvfusion.frozen_checkpoint import load_frozen_model_state
from rvfusion.public_official_fusion import build_public_official_baseline
from scripts.train_fusion import load_radar_adapter, route_expert_paths_from_args
from scripts.train_external_rvfusion_baselines import run_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation-only replay for frozen source-backed public fusion baselines"
    )
    parser.add_argument("--baseline", choices=("rcrosnet", "crn", "rcbevdet"), required=True)
    parser.add_argument("--public-source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-cache-root", type=Path, required=True)
    parser.add_argument("--condition-predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-open-indoor-los", type=Path, required=True)
    parser.add_argument("--expert-indoor-clutter", type=Path, required=True)
    parser.add_argument("--expert-open-outdoor-weak-return", type=Path, required=True)
    parser.add_argument("--expert-outdoor-dry-complex", type=Path, required=True)
    parser.add_argument("--expert-outdoor-wet-return", type=Path, required=True)
    parser.add_argument("--gg-recovery-scale", type=float, default=1.15)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--radar-clip-length", type=int, default=16)
    parser.add_argument("--visual-clip-length", type=int, default=8)
    parser.add_argument("--sequence-cache-size", type=int, default=2)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    timeline = ConditionTimeline.from_predictions_jsonl(args.condition_predictions)
    dataset = RVFusionCachedDataset(
        split="Development Val",
        data_root=args.data_root,
        manifest_path=args.manifest,
        visual_cache_root=args.visual_cache_root,
        condition_timeline=timeline,
        radar_clip_length=args.radar_clip_length,
        visual_clip_length=args.visual_clip_length,
        sequence_cache_size=args.sequence_cache_size,
    )
    sampler = SequenceGroupedSampler(dataset, seed=0, shuffle_sequences=False)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    expert_paths = route_expert_paths_from_args(args)
    radar = load_radar_adapter(
        args.base_checkpoint,
        device,
        expert_paths=expert_paths,
        gg_recovery_scale=args.gg_recovery_scale,
    )
    model = build_public_official_baseline(args.baseline, args.public_source_root).to(device)
    checkpoint_metadata = load_frozen_model_state(model, args.checkpoint)
    model.eval()
    with torch.no_grad():
        metrics = run_epoch(model, radar, loader, device, None, args.max_eval_batches)
    stress_manifest = args.visual_cache_root / "visual_cache_stress_manifest.json"
    record = {
        "schema": "dream_bev_public_fusion_stress_replay_v1",
        "stage": "evaluation_only_development",
        "locked_test_used": False,
        "baseline": args.baseline,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "visual_cache_root": str(args.visual_cache_root.resolve()),
        "visual_stress_manifest": str(stress_manifest.resolve()) if stress_manifest.is_file() else None,
        "visual_stress_manifest_sha256": sha256(stress_manifest) if stress_manifest.is_file() else None,
        "development_sequences": sorted(str(value) for value in dataset.sequences),
        "metrics": metrics,
    }
    (args.output_dir / "evaluation_only.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
