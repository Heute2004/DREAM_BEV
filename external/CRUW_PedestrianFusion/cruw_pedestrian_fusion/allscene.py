"""Full-scene CRUW indexing and canonical metric-BEV supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import CHIRP_IDS, _annotation_by_frame, _has_complete_modalities
from .protocol import CRUWProtocol


@dataclass(frozen=True)
class CRUWBEVGeometry:
    """The paper DREAM-BEV lattice, with tensor axes ordered as ``[x, y]``."""

    x_min_m: float = -10.0
    x_max_m: float = 10.0
    y_min_m: float = 0.0
    y_max_m: float = 23.4
    resolution_m: float = 0.1

    @property
    def x_bins(self) -> int:
        return int(round((self.x_max_m - self.x_min_m) / self.resolution_m))

    @property
    def y_bins(self) -> int:
        return int(round((self.y_max_m - self.y_min_m) / self.resolution_m))

    def contains(self, x_m: float, y_m: float) -> bool:
        return self.x_min_m <= x_m < self.x_max_m and self.y_min_m <= y_m < self.y_max_m

    def metric_to_index(self, x_m: float, y_m: float) -> tuple[int, int]:
        if not self.contains(x_m, y_m):
            raise ValueError(f"coordinate outside canonical BEV: ({x_m}, {y_m})")
        x_index = int(round((x_m - self.x_min_m) / self.resolution_m))
        y_index = int(round((y_m - self.y_min_m) / self.resolution_m))
        return min(x_index, self.x_bins - 1), min(y_index, self.y_bins - 1)


@dataclass(frozen=True)
class PedestrianTarget:
    range_m: float
    angle_rad: float

    @property
    def x_m(self) -> float:
        return float(self.range_m * np.sin(self.angle_rad))

    @property
    def y_m(self) -> float:
        return float(self.range_m * np.cos(self.angle_rad))


@dataclass(frozen=True)
class SceneFrameRecord:
    sequence_id: int
    frame_id: int
    sequence_start: int
    sequence_end: int
    pedestrians: tuple[PedestrianTarget, ...]
    distractor_count: int
    image_path: Path
    sample_stride: int

    @property
    def pedestrian_count(self) -> int:
        return len(self.pedestrians)


def build_all_scene_index(
    data_root: str | Path,
    protocol: CRUWProtocol,
    split: str,
    sample_stride: int = 3,
) -> list[SceneFrameRecord]:
    """Index every complete public-train frame without filtering by person count."""

    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    root = Path(data_root)
    records: list[SceneFrameRecord] = []
    for sequence_id in protocol.sequence_ids(split):
        sequence_root = root / "sequences" / "train" / str(sequence_id)
        image_root = sequence_root / "IMAGES_0"
        image_frames = sorted(int(path.stem) for path in image_root.glob("*.jpg"))
        if not image_frames:
            continue
        annotations = _annotation_by_frame(root / "annotations" / "train" / f"{sequence_id}.txt")
        sequence_start, sequence_end = image_frames[0], image_frames[-1]
        for frame_id in image_frames[::sample_stride]:
            complete, image_path = _has_complete_modalities(root, sequence_id, frame_id)
            if not complete:
                continue
            entries = annotations.get(frame_id, [])
            pedestrians = tuple(
                PedestrianTarget(range_m, angle_rad)
                for range_m, angle_rad, category in entries
                if category == "pedestrian"
            )
            records.append(
                SceneFrameRecord(
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    pedestrians=pedestrians,
                    distractor_count=sum(category != "pedestrian" for _, _, category in entries),
                    image_path=image_path,
                    sample_stride=sample_stride,
                )
            )
    return records


class RadarBEVWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal RA/RA-MTI windows with empty, single, or multi-person BEV targets."""

    def __init__(
        self,
        records: list[SceneFrameRecord],
        cache_root: str | Path,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        clip_length: int = 8,
        geometry: CRUWBEVGeometry | None = None,
        gaussian_sigma_cells: float = 1.5,
    ) -> None:
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        self.records = list(records)
        self.cache_root = Path(cache_root)
        self.range_grid = np.asarray(range_grid, dtype=np.float32)
        self.angle_grid = np.asarray(angle_grid, dtype=np.float32)
        self.clip_length = clip_length
        self.geometry = geometry or CRUWBEVGeometry()
        self.gaussian_sigma_cells = float(gaussian_sigma_cells)
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, sequence_id: int) -> Path:
        direct = self.cache_root / f"{sequence_id}.npy"
        public_train = self.cache_root / "train" / f"{sequence_id}.npy"
        if direct.is_file():
            return direct
        if public_train.is_file():
            return public_train
        raise FileNotFoundError(f"missing CRUW RA cache for sequence {sequence_id}")

    def _sequence_cache(self, sequence_id: int) -> np.ndarray:
        if sequence_id not in self._cache:
            self._cache[sequence_id] = np.load(self._cache_path(sequence_id), mmap_mode="r")
        return self._cache[sequence_id]

    def _target(self, record: SceneFrameRecord) -> tuple[np.ndarray, int]:
        target = np.zeros((self.geometry.x_bins, self.geometry.y_bins), dtype=np.float32)
        rows, columns = np.indices(target.shape, dtype=np.float32)
        valid_count = 0
        for pedestrian in record.pedestrians:
            if not self.geometry.contains(pedestrian.x_m, pedestrian.y_m):
                continue
            x_index, y_index = self.geometry.metric_to_index(pedestrian.x_m, pedestrian.y_m)
            squared = (rows - x_index) ** 2 + (columns - y_index) ** 2
            gaussian = np.exp(-0.5 * squared / (self.gaussian_sigma_cells**2), dtype=np.float32)
            target = np.maximum(target, gaussian)
            valid_count += 1
        return target[None, ...], valid_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        frame_ids = [
            max(record.sequence_start, record.frame_id - (self.clip_length - 1 - offset) * record.sample_stride)
            for offset in range(self.clip_length)
        ]
        amplitude = np.asarray(self._sequence_cache(record.sequence_id)[frame_ids], dtype=np.float32)
        if amplitude.ndim == 4:
            amplitude = amplitude.mean(axis=1)
        if amplitude.ndim != 3 or amplitude.shape[1:] != (128, 128):
            raise ValueError(f"unexpected cached RA shape: {amplitude.shape}")
        amplitude = (amplitude - float(amplitude.mean())) / (float(amplitude.std()) + 1e-6)
        mti = np.diff(amplitude, axis=0, prepend=amplitude[:1])
        mti = mti / (float(mti.std()) + 1e-6)
        radar = np.stack((amplitude, mti), axis=1).astype(np.float32, copy=False)
        target, valid_count = self._target(record)
        valid_pedestrians = [
            pedestrian
            for pedestrian in record.pedestrians
            if self.geometry.contains(pedestrian.x_m, pedestrian.y_m)
        ]
        single_x_m = valid_pedestrians[0].x_m if len(valid_pedestrians) == 1 else float("nan")
        single_y_m = valid_pedestrians[0].y_m if len(valid_pedestrians) == 1 else float("nan")
        return {
            "radar": torch.from_numpy(radar.copy()),
            "target": torch.from_numpy(target.copy()),
            "target_count": torch.tensor(valid_count, dtype=torch.int64),
            "raw_pedestrian_count": torch.tensor(record.pedestrian_count, dtype=torch.int64),
            "presence": torch.tensor(valid_count > 0, dtype=torch.float32),
            "single_x_m": torch.tensor(single_x_m, dtype=torch.float32),
            "single_y_m": torch.tensor(single_y_m, dtype=torch.float32),
            "sequence_id": torch.tensor(record.sequence_id, dtype=torch.int64),
            "frame_id": torch.tensor(record.frame_id, dtype=torch.int64),
            "distractor_count": torch.tensor(record.distractor_count, dtype=torch.int64),
        }
