from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rv_evidence_net.model import RVEvidenceNetSTMamba


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen RGB evidence per sequence")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Development Val"],
        help=(
            "Manifest splits to cache. Defaults intentionally exclude Locked Test "
            "so development-only preparation cannot read it by accident."
        ),
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_split_names(values: list[str]) -> set[str]:
    """Normalize shell-safe split aliases to manifest split names."""
    return {
        value.strip().lower().replace("_", " ")
        for value in values
        if value.strip()
    }


def load_spatial_model(checkpoint: Path, device: torch.device) -> RVEvidenceNetSTMamba:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    contract = payload.get("contract", {})
    source = payload.get("model", payload)
    # Older frozen spatial checkpoints predate the optional quality head and
    # do not record an explicit contract flag. Infer the setting from the
    # serialized keys so caching remains compatible with those checkpoints.
    quality_head = bool(contract.get("quality_head", False))
    if "quality_head" not in contract:
        quality_head = any(key.startswith("heads.0.quality.") for key in source)
    model = RVEvidenceNetSTMamba(
        width=float(contract.get("width", 1.25)),
        token_dim=int(contract.get("token_dim", 128)),
        temporal_model_dim=int(contract.get("temporal_model_dim", 128)),
        temporal_state_dim=int(contract.get("temporal_state_dim", 16)),
        temporal_layers=int(contract.get("temporal_layers", 2)),
        temporal_backend="selective_scan",
        top_k=int(contract.get("top_k", 3)),
        quality_head=quality_head,
    )
    spatial_prefixes = ("encoder.", "neck.", "heads.", "evidence_adapter.")
    spatial = {key: value for key, value in source.items() if key.startswith(spatial_prefixes)}
    missing, unexpected = model.load_state_dict(spatial, strict=False)
    missing_spatial = [key for key in missing if key.startswith(spatial_prefixes)]
    if missing_spatial or unexpected:
        raise ValueError(
            f"spatial checkpoint mismatch; missing={missing_spatial[:5]}, unexpected={unexpected[:5]}"
        )
    model.requires_grad_(False).eval().to(device)
    return model


def load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return transforms.to_tensor(
            transforms.resize(rgb, [image_size, image_size], antialias=True)
        )


def select_visual_sequences(
    manifest: Path,
    *,
    data_root: Path,
    allowed_splits: set[str],
) -> dict[str, list[tuple[int, Path]]]:
    """Select only explicitly allowed, non-excluded sequences from a manifest."""
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"].strip().lower()
            if split not in allowed_splits:
                continue
            if row.get("sequence_excluded", "False").strip().lower() in {
                "1",
                "true",
                "yes",
            }:
                continue
            grouped[row["sequence"].strip().lower()].append(
                (int(row["frame_id"]), data_root / row["rgb_path"])
            )
    if not grouped:
        raise ValueError("manifest selection produced no visual sequences")
    return grouped


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must lie in [0, shard_count)")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    allowed_splits = normalize_split_names(args.splits)
    grouped = select_visual_sequences(
        args.manifest,
        data_root=args.data_root,
        allowed_splits=allowed_splits,
    )
    selected = [
        sequence
        for index, sequence in enumerate(sorted(grouped))
        if index % args.shard_count == args.shard_index
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    model = load_spatial_model(args.checkpoint, device)
    completed = 0
    for sequence in selected:
        output_path = args.output_root / f"{sequence}.npz"
        if output_path.exists() and not args.overwrite:
            completed += 1
            continue
        rows = sorted(grouped[sequence])
        parts: dict[str, list[np.ndarray]] = defaultdict(list)
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            images = torch.stack(
                [load_image(path, args.image_size) for _, path in batch_rows]
            ).to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model.forward_frame(images)
            parts["boxes"].append(output.boxes.float().cpu().numpy())
            parts["raw_scores"].append(output.raw_scores.float().cpu().numpy())
            parts["raw_tokens"].append(output.raw_tokens.float().cpu().numpy())
            parts["valid_mask"].append(output.valid_mask.cpu().numpy())
        boxes = np.concatenate(parts["boxes"]).astype(np.float16)
        scores = np.concatenate(parts["raw_scores"]).astype(np.float16)
        raw_tokens = np.concatenate(parts["raw_tokens"]).astype(np.float16)
        valid = np.concatenate(parts["valid_mask"]).astype(bool)
        np.savez(
            output_path,
            frame_ids=np.asarray([frame for frame, _ in rows], dtype=np.int64),
            boxes=boxes,
            raw_scores=scores,
            raw_tokens=raw_tokens,
            # The formal visual temporal model is not promoted. Causal history
            # is modeled by RVFusion's KVMemory, so no random recovery token is used.
            temporal_tokens=raw_tokens,
            temporal_support=scores,
            valid_mask=valid,
        )
        completed += 1
        print(
            json.dumps(
                {
                    "sequence": sequence,
                    "frames": len(rows),
                    "completed": completed,
                    "assigned": len(selected),
                }
            ),
            flush=True,
        )
    metadata = {
        "cache_contract": "visual_evidence_cache",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "image_size": args.image_size,
        "top_k": model.top_k,
        "temporal_source": "raw_token_causal_kv_memory",
        "proposal_quality_used": False,
        "splits": sorted(allowed_splits),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sequences": len(selected),
    }
    (args.output_root / f"cache_metadata_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()



