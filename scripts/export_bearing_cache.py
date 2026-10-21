from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rv_bearing_net.evidence_initialized import (
    EvidenceInitializedRVBearingNet,
)
from rv_bearing_net.cached_proposals import validate_cached_proposal_source
from rv_bearing_net.model import RVBearingNet
from rv_bearing_net.monotonic_residual import (
    MonotonicResidualRVBearingNet,
)
from rv_bearing_net.proposal_geometry import (
    ProposalGeometryRVBearingNet,
)


def save_bearing_cache(
    path: str | Path,
    *,
    frame_ids: np.ndarray,
    theta: np.ndarray,
    sigma_theta: np.ndarray,
    reliability: np.ndarray,
    temporal_feature: np.ndarray,
    valid: np.ndarray,
) -> None:
    path = Path(path)
    arrays = {
        "frame_ids": np.asarray(frame_ids, dtype=np.int64),
        "theta": np.asarray(theta, dtype=np.float32),
        "sigma_theta": np.asarray(sigma_theta, dtype=np.float32),
        "reliability": np.asarray(reliability, dtype=np.float32),
        "temporal_feature": np.asarray(
            temporal_feature,
            dtype=np.float32,
        ),
        "valid": np.asarray(valid, dtype=np.bool_),
    }
    count = arrays["frame_ids"].size
    if arrays["frame_ids"].ndim != 1 or count == 0:
        raise ValueError("frame_ids must be a non-empty vector")
    if np.unique(arrays["frame_ids"]).size != count:
        raise ValueError("frame_ids must be unique")
    if arrays["theta"].shape != (count,):
        raise ValueError("theta must align with frame_ids")
    if arrays["sigma_theta"].shape != (count,):
        raise ValueError("sigma_theta must align with frame_ids")
    if arrays["reliability"].shape != (count,):
        raise ValueError("reliability must align with frame_ids")
    if arrays["valid"].shape != (count,):
        raise ValueError("valid must align with frame_ids")
    if (
        arrays["temporal_feature"].ndim != 2
        or arrays["temporal_feature"].shape[0] != count
    ):
        raise ValueError("temporal_feature must be [frames,channels]")
    for name in ("theta", "sigma_theta", "reliability", "temporal_feature"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be finite")
    if np.any(arrays["sigma_theta"] <= 0.0):
        raise ValueError("sigma_theta must be positive")
    if np.any((arrays["reliability"] < 0.0) | (arrays["reliability"] > 1.0)):
        raise ValueError("reliability must lie in [0,1]")
    order = np.argsort(arrays["frame_ids"], kind="stable")
    arrays = {name: values[order] for name, values in arrays.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
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


def resolve_export_runtime_contract(
    metadata: Mapping[str, object],
    *,
    clip_length: int | None,
    image_height: int | None,
    image_width: int | None,
) -> dict[str, int]:
    recorded_size = metadata.get("image_size", [256, 448])
    if (
        not isinstance(recorded_size, (list, tuple))
        or len(recorded_size) != 2
    ):
        raise ValueError("checkpoint image_size metadata is invalid")
    resolved = {
        "clip_length": int(
            clip_length
            if clip_length is not None
            else metadata.get("clip_length", 8)
        ),
        "image_height": int(
            image_height
            if image_height is not None
            else recorded_size[0]
        ),
        "image_width": int(
            image_width
            if image_width is not None
            else recorded_size[1]
        ),
    }
    if resolved["clip_length"] < 2:
        raise ValueError("clip_length must be at least 2")
    if resolved["image_height"] < 1 or resolved["image_width"] < 1:
        raise ValueError("image dimensions must be positive")
    return resolved


def _read_rgb_uint8(path: Path, height: int, width: int) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"missing RGB frame: {path}")
    with Image.open(path) as image:
        resized = transforms.resize(
            image.convert("RGB"),
            [height, width],
            antialias=True,
        )
        return transforms.pil_to_tensor(resized)


def build_model_from_checkpoint(
    payload: Mapping[str, object],
) -> tuple[RVBearingNet, str]:
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    backbone = str(metadata.get("backbone", "scratch"))
    if backbone == "scratch":
        model: RVBearingNet = RVBearingNet()
    elif backbone == "evidence_s1":
        model = EvidenceInitializedRVBearingNet()
    elif backbone == "proposal_geometry_s1":
        proposal_contract = metadata.get("proposal_contract", {})
        if not isinstance(proposal_contract, Mapping):
            raise TypeError("proposal contract must be a mapping")
        model = ProposalGeometryRVBearingNet.from_contract(
            proposal_contract
        )
    elif backbone == "monotonic_residual_s1":
        proposal_contract = metadata.get("proposal_contract", {})
        if not isinstance(proposal_contract, Mapping):
            raise TypeError("proposal contract must be a mapping")
        model = MonotonicResidualRVBearingNet.from_contract(
            proposal_contract
        )
    else:
        raise ValueError(f"unsupported RV-BearingNet backbone: {backbone}")
    raw_state = payload.get("model_state", payload)
    if not isinstance(raw_state, Mapping):
        raise TypeError("checkpoint must contain a model state mapping")
    model.load_state_dict(raw_state, strict=True)
    return model, backbone


def export_sequence(
    *,
    model: RVBearingNet,
    frame_ids: list[int],
    image_paths: list[Path],
    output_path: Path,
    device: torch.device,
    clip_length: int,
    batch_size: int,
    image_height: int,
    image_width: int,
    amp: bool,
) -> dict[str, object]:
    if len(frame_ids) != len(image_paths) or not frame_ids:
        raise ValueError("frame IDs and image paths must align and be non-empty")
    if clip_length < 2 or batch_size < 1:
        raise ValueError("clip_length must be >=2 and batch_size positive")
    images = torch.stack(
        [
            _read_rgb_uint8(path, image_height, image_width)
            for path in image_paths
        ]
    )
    theta: list[Tensor] = []
    sigma: list[Tensor] = []
    reliability: list[Tensor] = []
    features: list[Tensor] = []
    valid: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frame_ids), batch_size):
            positions = torch.arange(
                start,
                min(start + batch_size, len(frame_ids)),
            )
            offsets = torch.arange(clip_length) - (clip_length - 1)
            history = (positions[:, None] + offsets[None]).clamp_min(0)
            clips = images[history].float().div_(255.0).to(
                device,
                non_blocking=True,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                output = model(clips)
            theta.append(output.theta[:, 0].float().cpu())
            sigma.append(output.sigma_theta[:, 0].float().cpu())
            reliability.append(output.reliability[:, 0].float().cpu())
            features.append(output.temporal_feature.float().cpu())
            valid.append(output.valid.cpu())
    save_bearing_cache(
        output_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        theta=torch.cat(theta).numpy(),
        sigma_theta=torch.cat(sigma).numpy(),
        reliability=torch.cat(reliability).numpy(),
        temporal_feature=torch.cat(features).numpy(),
        valid=torch.cat(valid).numpy(),
    )
    return {
        "frames": len(frame_ids),
        "valid_frames": int(torch.cat(valid).sum()),
        "output": str(output_path),
    }


def export_sequence_from_proposals(
    *,
    model: object,
    frame_ids: list[int],
    proposal_cache_path: Path,
    output_path: Path,
    device: torch.device,
    clip_length: int,
    batch_size: int,
    image_height: int,
    image_width: int,
    amp: bool,
) -> dict[str, object]:
    if not frame_ids:
        raise ValueError("frame IDs must be non-empty")
    if clip_length < 2 or batch_size < 1:
        raise ValueError("clip_length must be >=2 and batch_size positive")
    required = {
        "frame_ids",
        "boxes",
        "raw_scores",
        "raw_tokens",
        "valid_mask",
    }
    with np.load(proposal_cache_path, allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise ValueError(
                f"proposal cache lacks {sorted(missing)}: "
                f"{proposal_cache_path}"
            )
        arrays = {
            name: np.array(payload[name], copy=True)
            for name in required
        }
    cached_frame_ids = arrays["frame_ids"].astype(np.int64, copy=False)
    frame_index = {
        int(frame_id): index
        for index, frame_id in enumerate(cached_frame_ids.tolist())
    }
    try:
        selected = np.asarray(
            [frame_index[int(frame_id)] for frame_id in frame_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise KeyError(
            f"proposal cache does not cover frame {int(error.args[0])}"
        ) from error
    tensors = {
        "boxes": torch.from_numpy(arrays["boxes"][selected]).float(),
        "scores": torch.from_numpy(
            arrays["raw_scores"][selected]
        ).float(),
        "tokens": torch.from_numpy(
            arrays["raw_tokens"][selected]
        ).float(),
        "valid_mask": torch.from_numpy(
            arrays["valid_mask"][selected]
        ).bool(),
    }
    forward_cached_pair = getattr(model, "forward_cached_pair", None)
    if forward_cached_pair is None:
        raise TypeError("selected backbone does not accept proposal caches")
    theta: list[Tensor] = []
    sigma: list[Tensor] = []
    reliability: list[Tensor] = []
    features: list[Tensor] = []
    valid: list[Tensor] = []
    model.eval()  # type: ignore[attr-defined]
    with torch.inference_mode():
        for start in range(0, len(frame_ids), batch_size):
            positions = torch.arange(
                start,
                min(start + batch_size, len(frame_ids)),
            )
            offsets = torch.arange(clip_length) - (clip_length - 1)
            history = (positions[:, None] + offsets[None]).clamp_min(0)
            current_inputs = {
                name: values[history].to(device, non_blocking=True)
                for name, values in tensors.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp and device.type == "cuda",
            ):
                _, output = forward_cached_pair(
                    **current_inputs,
                    image_height=image_height,
                    image_width=image_width,
                )
            theta.append(output.theta[:, 0].float().cpu())
            sigma.append(output.sigma_theta[:, 0].float().cpu())
            reliability.append(output.reliability[:, 0].float().cpu())
            features.append(output.temporal_feature.float().cpu())
            valid.append(output.valid.cpu())
    save_bearing_cache(
        output_path,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        theta=torch.cat(theta).numpy(),
        sigma_theta=torch.cat(sigma).numpy(),
        reliability=torch.cat(reliability).numpy(),
        temporal_feature=torch.cat(features).numpy(),
        valid=torch.cat(valid).numpy(),
    )
    return {
        "frames": len(frame_ids),
        "valid_frames": int(torch.cat(valid).sum()),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export frame-aligned RV-BearingNet evidence caches"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-evidence-cache-root", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Held-out Eval"],
    )
    parser.add_argument("--clip-length", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--start-sequence", type=int, default=1)
    parser.add_argument("--end-sequence", type=int, default=999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_splits = normalize_split_names(args.splits)
    grouped: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() not in allowed_splits:
                continue
            sequence = row["sequence"].strip().lower()
            sequence_id = int(sequence.split("_")[-1])
            if not args.start_sequence <= sequence_id <= args.end_sequence:
                continue
            grouped[sequence].append(
                (
                    int(row["frame_id"]),
                    args.data_root / row["rgb_path"],
                )
            )
    if not grouped:
        raise ValueError("manifest selection produced no sequences")
    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    model, backbone = build_model_from_checkpoint(payload)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    runtime_contract = resolve_export_runtime_contract(
        metadata,
        clip_length=args.clip_length,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    proposal_cache_contract: dict[str, object] | None = None
    if args.visual_evidence_cache_root is not None:
        if backbone not in (
            "proposal_geometry_s1",
            "monotonic_residual_s1",
        ):
            raise ValueError(
                "frozen proposal cache export requires a proposal backbone"
            )
        evidence_checkpoint = metadata.get("evidence_checkpoint")
        if not isinstance(evidence_checkpoint, str) or not evidence_checkpoint:
            raise ValueError(
                "checkpoint lacks the evidence checkpoint needed to "
                "validate frozen proposals"
            )
        proposal_cache_contract = validate_cached_proposal_source(
            cache_root=args.visual_evidence_cache_root,
            checkpoint=evidence_checkpoint,
            image_height=runtime_contract["image_height"],
            image_width=runtime_contract["image_width"],
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model.to(device)
    results: dict[str, object] = {}
    for sequence, rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item[0])
        common = {
            "model": model,
            "frame_ids": [item[0] for item in rows],
            "output_path": args.output_root / f"{sequence}.npz",
            "device": device,
            "clip_length": runtime_contract["clip_length"],
            "batch_size": args.batch_size,
            "image_height": runtime_contract["image_height"],
            "image_width": runtime_contract["image_width"],
            "amp": not args.no_amp,
        }
        if args.visual_evidence_cache_root is not None:
            results[sequence] = export_sequence_from_proposals(
                proposal_cache_path=(
                    args.visual_evidence_cache_root / f"{sequence}.npz"
                ),
                **common,
            )
        else:
            results[sequence] = export_sequence(
                image_paths=[item[1] for item in rows],
                **common,
            )
        print(json.dumps({sequence: results[sequence]}), flush=True)
    summary = {
        "cache_contract": "rv_bearing_evidence_cache",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "runtime_input": "causal_left_rgb_only",
        "export_input_source": (
            "checkpoint_bound_frozen_proposal_cache"
            if args.visual_evidence_cache_root is not None
            else "causal_left_rgb"
        ),
        "proposal_cache_contract": proposal_cache_contract,
        "backbone": backbone,
        "clip_length": runtime_contract["clip_length"],
        "image_size": [
            runtime_contract["image_height"],
            runtime_contract["image_width"],
        ],
        "sequences": results,
    }
    temporary = args.output_root / "cache_manifest.json.tmp"
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, args.output_root / "cache_manifest.json")


if __name__ == "__main__":
    main()



