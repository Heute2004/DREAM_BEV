from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for project_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))

from rvfusion.radar_health import compute_radar_health


@dataclass(frozen=True)
class RadarFrame:
    frame_id: int
    ra_path: Path
    ra_mti_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen DREAM-BEV Radar Health evidence by sequence."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--radar-clip-length", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--db-center", type=float, default=80.0)
    parser.add_argument("--db-scale", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Development Val"],
        help=(
            "Manifest splits to cache. Defaults intentionally exclude Locked Test "
            "so development-only preparation cannot read it by accident."
        ),
    )
    parser.add_argument("--start-sequence", type=int, default=1)
    parser.add_argument("--end-sequence", type=int, default=999)
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


def atomic_write_health_cache(
    path: Path,
    *,
    frame_ids: np.ndarray,
    health: np.ndarray,
) -> None:
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    health = np.asarray(health, dtype=np.float32)
    if frame_ids.ndim != 1 or health.shape != (frame_ids.size, 8):
        raise ValueError("health cache must contain frame_ids [N] and health [N,8]")
    if frame_ids.size == 0 or not np.isfinite(health).all():
        raise ValueError("health cache must be non-empty and finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, frame_ids=frame_ids, health=health)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _sequence_number(sequence: str) -> int:
    try:
        return int(sequence.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid sequence name: {sequence}") from error


def load_manifest_sequences(
    manifest_path: Path,
    *,
    data_root: Path,
    allowed_splits: set[str],
    start_sequence: int,
    end_sequence: int,
) -> dict[str, tuple[RadarFrame, ...]]:
    grouped: dict[str, list[RadarFrame]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"].strip().lower() not in allowed_splits:
                continue
            sequence = row["sequence"].strip().lower()
            sequence_number = _sequence_number(sequence)
            if not start_sequence <= sequence_number <= end_sequence:
                continue
            if _as_bool(row.get("sequence_excluded", "False")):
                continue
            grouped[sequence].append(
                RadarFrame(
                    frame_id=int(row["frame_id"]),
                    ra_path=data_root / row["ra_path"],
                    ra_mti_path=data_root / row["ra_mti_path"],
                )
            )
    sequences = {
        sequence: tuple(sorted(frames, key=lambda item: item.frame_id))
        for sequence, frames in sorted(grouped.items())
    }
    if not sequences:
        raise ValueError("manifest contains no selected non-excluded sequences")
    for sequence, frames in sequences.items():
        frame_ids = [frame.frame_id for frame in frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError(f"duplicate frame IDs in {sequence}")
        missing = [
            str(path)
            for frame in frames
            for path in (frame.ra_path, frame.ra_mti_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{sequence} has {len(missing)} missing radar files; first={missing[0]}"
            )
    return sequences


def _read_radar_map(
    path: Path,
    *,
    db_center: float,
    db_scale: float,
) -> Tensor:
    values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if values.shape != (128, 128):
        raise ValueError(f"radar map must be 128x128: {path}")
    values = np.nan_to_num(
        values,
        nan=db_center,
        posinf=db_center + db_scale,
        neginf=db_center - db_scale,
    )
    normalized = np.clip((values - db_center) / db_scale, -3.0, 3.0)
    return torch.from_numpy(normalized)


def _load_radar_adapter(checkpoint: Path, device: torch.device):
    from radar_bev.models.backbones import DREAMBEV, extract_dream_bev_state
    from rvfusion.adapters import FrozenRadarAdapter

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    backbone = DREAMBEV()
    backbone.load_state_dict(extract_dream_bev_state(payload), strict=True)
    adapter = FrozenRadarAdapter(backbone).to(device)
    adapter.eval()
    return adapter


def _decode_peak_xy(logits: Tensor) -> Tensor:
    probability = logits.float().flatten(1).softmax(dim=1)
    flat_index = probability.argmax(dim=1)
    x_index = torch.div(flat_index, 234, rounding_mode="floor")
    y_index = flat_index.remainder(234)
    x_m = -10.0 + x_index.to(logits.dtype) * 0.1
    y_m = y_index.to(logits.dtype) * 0.1
    return torch.stack((x_m, y_m), dim=1)


def cache_sequence(
    *,
    frames: tuple[RadarFrame, ...],
    adapter,
    device: torch.device,
    batch_size: int,
    radar_clip_length: int,
    db_center: float,
    db_scale: float,
    use_amp: bool,
) -> np.ndarray:
    radar = torch.stack(
        [
            torch.stack(
                (
                    _read_radar_map(
                        frame.ra_path,
                        db_center=db_center,
                        db_scale=db_scale,
                    ),
                    _read_radar_map(
                        frame.ra_mti_path,
                        db_center=db_center,
                        db_scale=db_scale,
                    ),
                )
            )
            for frame in frames
        ]
    )
    history_xy: list[Tensor] = []
    health_chunks: list[Tensor] = []
    amp_enabled = use_amp and device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, radar.shape[0], batch_size):
            end = min(start + batch_size, radar.shape[0])
            end_positions = torch.arange(start, end, dtype=torch.long)
            offsets = torch.arange(
                -radar_clip_length + 1,
                1,
                dtype=torch.long,
            )
            window_indices = (end_positions[:, None] + offsets[None]).clamp_min(0)
            window = radar[window_indices].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                evidence = adapter(window)
            logits = evidence.logits.float()
            current_xy = _decode_peak_xy(logits)
            previous_rows: list[Tensor] = []
            predicted_rows: list[Tensor] = []
            for row in current_xy:
                if history_xy:
                    previous = history_xy[-1]
                else:
                    previous = row
                if len(history_xy) >= 2:
                    predicted = history_xy[-1] + history_xy[-1] - history_xy[-2]
                else:
                    predicted = previous
                previous_rows.append(previous)
                predicted_rows.append(predicted)
                history_xy.append(row)
            health = compute_radar_health(
                logits,
                ra_current=window[:, -1, 0],
                ra_mti_current=window[:, -1, 1],
                previous_xy_m=torch.stack(previous_rows),
                predicted_xy_m=torch.stack(predicted_rows),
            )
            health_chunks.append(health.vector.cpu())
    result = torch.cat(health_chunks, dim=0).numpy().astype(np.float32, copy=False)
    if result.shape != (len(frames), 8) or not np.isfinite(result).all():
        raise RuntimeError("generated invalid Radar Health payload")
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.radar_clip_length < 1:
        raise ValueError("batch size and radar clip length must be positive")
    if args.db_scale <= 0:
        raise ValueError("db_scale must be positive")
    device = torch.device(args.device)
    sequences = load_manifest_sequences(
        args.manifest,
        data_root=args.data_root,
        allowed_splits=normalize_split_names(args.splits),
        start_sequence=args.start_sequence,
        end_sequence=args.end_sequence,
    )
    adapter = _load_radar_adapter(args.base_checkpoint, device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    for index, (sequence, frames) in enumerate(sequences.items(), start=1):
        output_path = args.output_root / f"{sequence}.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Radar Health cache already exists; use --overwrite: {output_path}"
            )
        health = cache_sequence(
            frames=frames,
            adapter=adapter,
            device=device,
            batch_size=args.batch_size,
            radar_clip_length=args.radar_clip_length,
            db_center=args.db_center,
            db_scale=args.db_scale,
            use_amp=not args.no_amp,
        )
        atomic_write_health_cache(
            output_path,
            frame_ids=np.asarray([frame.frame_id for frame in frames], dtype=np.int64),
            health=health,
        )
        completed.append(
            {
                "sequence": sequence,
                "frames": len(frames),
                "file": output_path.name,
            }
        )
        print(f"[{index}/{len(sequences)}] {sequence}: {len(frames)} frames")
    manifest_payload: dict[str, object] = {
        "schema": "RadarHealthCachecanonical",
        "checkpoint": str(args.base_checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.base_checkpoint),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "splits": sorted(normalize_split_names(args.splits)),
        "radar_clip_length": args.radar_clip_length,
        "health_dimensions": 8,
        "normalization": {
            "db_center": args.db_center,
            "db_scale": args.db_scale,
            "clip": [-3.0, 3.0],
        },
        "sequences": completed,
    }
    _atomic_write_json(args.output_root / "cache_manifest.json", manifest_payload)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "sequences": len(completed),
                "frames": sum(int(item["frames"]) for item in completed),
            }
        )
    )


if __name__ == "__main__":
    main()



