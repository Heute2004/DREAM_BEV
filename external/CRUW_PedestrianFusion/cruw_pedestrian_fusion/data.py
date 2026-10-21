"""Protocol-constrained CRUW pedestrian frame indexing."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .protocol import CRUWProtocol


CHIRP_IDS = (0, 64, 128, 192)


@dataclass(frozen=True)
class FrameRecord:
    sequence_id: int
    frame_id: int
    segment_start: int
    segment_end: int
    range_m: float
    angle_rad: float
    distractor_count: int
    image_path: Path
    sample_stride: int = 1


@dataclass(frozen=True)
class CandidateSegment:
    start_frame: int
    end_frame: int
    sample_stride: int


def _candidate_segments(path: Path) -> dict[int, list[CandidateSegment]]:
    segments: dict[int, list[CandidateSegment]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_stride = int(row.get("sample_stride") or 1)
            if sample_stride < 1:
                raise ValueError(f"invalid sample_stride: {sample_stride}")
            segments[int(row["sequence_id"])].append(
                CandidateSegment(int(row["start_frame"]), int(row["end_frame"]), sample_stride)
            )
    return {sequence_id: sorted(items, key=lambda item: item.start_frame) for sequence_id, items in segments.items()}


def _annotation_by_frame(path: Path) -> dict[int, list[tuple[float, float, str]]]:
    annotations: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.split()
            if len(parts) != 4:
                continue
            frame_id, range_m, angle_rad, category = parts
            annotations[int(frame_id)].append((float(range_m), float(angle_rad), category))
    return annotations


def _has_complete_modalities(data_root: Path, sequence_id: int, frame_id: int) -> tuple[bool, Path]:
    sequence_root = data_root / "sequences" / "train" / str(sequence_id)
    image_path = sequence_root / "IMAGES_0" / f"{frame_id:010d}.jpg"
    radar_root = sequence_root / "RADAR_RA_H"
    complete = image_path.is_file() and all(
        (radar_root / f"{frame_id:06d}_{chirp_id:04d}.npy").is_file() for chirp_id in CHIRP_IDS
    )
    return complete, image_path


def build_frame_index(
    data_root: str | Path,
    protocol: CRUWProtocol,
    candidates_csv: str | Path,
    split: str,
) -> list[FrameRecord]:
    """Build records from the public training tree without exposing CRUW test data."""

    root = Path(data_root)
    candidates = _candidate_segments(Path(candidates_csv))
    records: list[FrameRecord] = []
    for sequence_id in protocol.sequence_ids(split):
        annotations = _annotation_by_frame(root / "annotations" / "train" / f"{sequence_id}.txt")
        for segment in candidates.get(sequence_id, []):
            for frame_id in range(segment.start_frame, segment.end_frame + 1, segment.sample_stride):
                entries = annotations.get(frame_id, [])
                pedestrians = [(range_m, angle_rad) for range_m, angle_rad, category in entries if category == "pedestrian"]
                if len(pedestrians) != 1:
                    continue
                complete, image_path = _has_complete_modalities(root, sequence_id, frame_id)
                if not complete:
                    continue
                range_m, angle_rad = pedestrians[0]
                records.append(
                    FrameRecord(
                        sequence_id=sequence_id,
                        frame_id=frame_id,
                        segment_start=segment.start_frame,
                        segment_end=segment.end_frame,
                        range_m=range_m,
                        angle_rad=angle_rad,
                        distractor_count=len(entries) - 1,
                        image_path=image_path,
                        sample_stride=segment.sample_stride,
                    )
                )
    return records


class RadarWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal RA/RA-MTI windows backed by per-sequence amplitude caches."""

    def __init__(
        self,
        records: list[FrameRecord],
        cache_root: str | Path,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        clip_length: int = 8,
    ) -> None:
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        self.records = list(records)
        self.cache_root = Path(cache_root)
        self.range_grid = np.asarray(range_grid, dtype=np.float32)
        self.angle_grid = np.asarray(angle_grid, dtype=np.float32)
        self.clip_length = clip_length
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _sequence_cache(self, sequence_id: int) -> np.ndarray:
        if sequence_id not in self._cache:
            self._cache[sequence_id] = np.load(self.cache_root / f"{sequence_id}.npy", mmap_mode="r")
        return self._cache[sequence_id]

    def _target(self, record: FrameRecord) -> np.ndarray:
        ridx = int(np.argmin(np.abs(self.range_grid - record.range_m)))
        aidx = int(np.argmin(np.abs(self.angle_grid - record.angle_rad)))
        rows, columns = np.indices((128, 128), dtype=np.float32)
        squared = (rows - ridx) ** 2 + (columns - aidx) ** 2
        target = np.exp(-0.5 * squared / (1.5**2), dtype=np.float32)
        return target[None, ...]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        frame_ids = [
            max(record.segment_start, record.frame_id - (self.clip_length - 1 - offset) * record.sample_stride)
            for offset in range(self.clip_length)
        ]
        amplitude = np.asarray(self._sequence_cache(record.sequence_id)[frame_ids], dtype=np.float32)
        amplitude = (amplitude - float(amplitude.mean())) / (float(amplitude.std()) + 1e-6)
        mti = np.diff(amplitude, axis=0, prepend=amplitude[:1])
        mti = mti / (float(mti.std()) + 1e-6)
        radar = np.stack((amplitude, mti), axis=1).astype(np.float32, copy=False)
        return {
            "radar": torch.from_numpy(radar.copy()),
            "target": torch.from_numpy(self._target(record).copy()),
            "range_m": torch.tensor(record.range_m, dtype=torch.float32),
            "angle_rad": torch.tensor(record.angle_rad, dtype=torch.float32),
            "x_m": torch.tensor(record.range_m * np.sin(record.angle_rad), dtype=torch.float32),
            "y_m": torch.tensor(record.range_m * np.cos(record.angle_rad), dtype=torch.float32),
            "sequence_id": torch.tensor(record.sequence_id, dtype=torch.int64),
            "frame_id": torch.tensor(record.frame_id, dtype=torch.int64),
            "distractor_count": torch.tensor(record.distractor_count, dtype=torch.int64),
        }


class RGBWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal RGB history paired with the same frozen pedestrian records."""

    def __init__(
        self,
        records: list[FrameRecord],
        clip_length: int = 4,
        image_size: int = 224,
        image_root: str | Path | None = None,
    ) -> None:
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        self.records = list(records)
        self.clip_length = clip_length
        self.image_root = Path(image_root) if image_root is not None else None
        self.by_sequence_frame = {(record.sequence_id, record.frame_id): record for record in self.records}
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _image_path(self, record: FrameRecord) -> Path:
        if self.image_root is None:
            return record.image_path
        return self.image_root / str(record.sequence_id) / f"{record.frame_id:010d}.jpg"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        images: list[torch.Tensor] = []
        for offset in range(self.clip_length):
            frame_id = max(record.segment_start, record.frame_id - (self.clip_length - 1 - offset) * record.sample_stride)
            history = self.by_sequence_frame.get((record.sequence_id, frame_id), record)
            image_path = self._image_path(history)
            if not image_path.is_file():
                raise FileNotFoundError(f"missing declared RGB evidence frame: {image_path}")
            with Image.open(image_path) as image:
                images.append(self.transform(image.convert("RGB")))
        return {
            "images": torch.stack(images),
            "angle_rad": torch.tensor(record.angle_rad, dtype=torch.float32),
            "range_m": torch.tensor(record.range_m, dtype=torch.float32),
            "sequence_id": torch.tensor(record.sequence_id, dtype=torch.int64),
            "frame_id": torch.tensor(record.frame_id, dtype=torch.int64),
        }


class FusionDataset(Dataset[dict[str, torch.Tensor]]):
    """Single-index pairing of causal radar and RGB histories."""

    def __init__(
        self,
        records: list[FrameRecord],
        cache_root: str | Path,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        radar_clip_length: int = 8,
        rgb_clip_length: int = 4,
        image_size: int = 224,
        image_root: str | Path | None = None,
    ) -> None:
        self.radar = RadarWindowDataset(records, cache_root, range_grid, angle_grid, radar_clip_length)
        self.rgb = RGBWindowDataset(records, rgb_clip_length, image_size, image_root=image_root)

    def __len__(self) -> int:
        return len(self.radar)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        radar = self.radar[index]
        rgb = self.rgb[index]
        if radar["sequence_id"].item() != rgb["sequence_id"].item() or radar["frame_id"].item() != rgb["frame_id"].item():
            raise RuntimeError("radar/RGB record mismatch")
        radar["images"] = rgb["images"]
        return radar
