"""CARRADA sequence indexing and canonical DREAM-BEV supervision."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .protocol import CARRADAProtocol

CARRADA_RANGE_RESOLUTION_M = 0.1953125
CARRADA_ANGLE_RESOLUTION_RAD = 0.01227184630308513
CARRADA_RANGE_BINS = 256
CARRADA_ANGLE_BINS = 256


@dataclass(frozen=True)
class CARRADABEVGeometry:
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
class Target:
    x_m: float
    y_m: float
    category: str
    instance_id: str | None = None

    @property
    def range_m(self) -> float:
        return float(math.hypot(self.x_m, self.y_m))

    @property
    def angle_rad(self) -> float:
        return float(math.atan2(self.x_m, self.y_m))


@dataclass(frozen=True)
class PolarTarget:
    range_m: float
    angle_rad: float

    @property
    def x_m(self) -> float:
        return float(self.range_m * math.sin(self.angle_rad))

    @property
    def y_m(self) -> float:
        return float(self.range_m * math.cos(self.angle_rad))


@dataclass(frozen=True)
class SceneFrameRecord:
    sequence_id: str
    frame_id: int
    targets: tuple[Target, ...]
    distractor_count: int
    image_path: Path
    episode_id: str | None = None
    episode_start_frame: int | None = None
    target_instance_id: str | None = None

    @property
    def pedestrian_count(self) -> int:
        return len(self.targets)

    @property
    def temporal_group(self) -> str:
        return self.episode_id or self.sequence_id


@dataclass(frozen=True)
class SingleTargetEpisode:
    episode_id: str
    sequence_id: str
    target_instance_id: str
    start_frame: int
    end_frame: int
    frame_count: int


def canonical_frame_files(directory: str | Path, suffix: str) -> dict[int, Path]:
    """Index only canonical six-digit CARRADA frame names."""

    paths: dict[int, Path] = {}
    for path in Path(directory).glob(f"*{suffix}"):
        if len(path.stem) != 6 or not path.stem.isdigit():
            continue
        frame_id = int(path.stem)
        if frame_id in paths:
            raise RuntimeError(f"duplicate canonical frame id {frame_id} in {directory}")
        paths[frame_id] = path
    return paths


def carrada_ra_grids() -> tuple[np.ndarray, np.ndarray]:
    raw_range = (CARRADA_RANGE_BINS - np.arange(CARRADA_RANGE_BINS, dtype=np.float32)) * CARRADA_RANGE_RESOLUTION_M
    raw_angle = (np.arange(CARRADA_ANGLE_BINS, dtype=np.float32) - CARRADA_ANGLE_BINS / 2) * CARRADA_ANGLE_RESOLUTION_RAD
    ranges = raw_range[::-1].reshape(128, 2).mean(axis=1)
    angles = raw_angle.reshape(128, 2).mean(axis=1)
    return ranges.astype(np.float32), angles.astype(np.float32)


def ra_index_to_metric(range_index: float, angle_index: float) -> PolarTarget:
    range_m = (CARRADA_RANGE_BINS - float(range_index)) * CARRADA_RANGE_RESOLUTION_M
    angle_rad = (float(angle_index) - CARRADA_ANGLE_BINS / 2) * CARRADA_ANGLE_RESOLUTION_RAD
    return PolarTarget(range_m=range_m, angle_rad=angle_rad)


def transform_ra(raw: np.ndarray, *, mean_db: float, std_db: float) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float32)
    if value.shape != (CARRADA_RANGE_BINS, CARRADA_ANGLE_BINS):
        raise ValueError(f"CARRADA RA must be 256x256, got {value.shape}")
    if std_db <= 0:
        raise ValueError("std_db must be positive")
    log_power = 20.0 * np.log10(np.maximum(value, 1e-6))
    pooled = log_power[::-1].reshape(128, 2, 128, 2).mean(axis=(1, 3))
    return ((pooled - float(mean_db)) / float(std_db)).astype(np.float32)


def _category(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"1", "pedestrian", "person"}:
        return "pedestrian"
    if normalized in {"2", "cyclist", "bicycle"}:
        return "cyclist"
    if normalized in {"3", "car", "vehicle"}:
        return "car"
    return normalized


def _targets_from_frame(frame: dict[str, object]) -> tuple[tuple[Target, ...], int]:
    people: list[Target] = []
    distractors = 0
    for instance_id, annotation in frame.items():
        if not isinstance(annotation, dict):
            continue
        ra = annotation.get("range_angle", {})
        if not isinstance(ra, dict):
            continue
        category = _category(ra.get("label", annotation.get("label", "unknown")))
        points = np.asarray(ra.get("sparse", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
            continue
        polar = ra_index_to_metric(float(points[:, 0].mean()), float(points[:, 1].mean()))
        target = Target(polar.x_m, polar.y_m, category, str(instance_id))
        if category == "pedestrian":
            people.append(target)
        else:
            distractors += 1
    return tuple(people), distractors


def derive_single_target_episodes(
    records: list[SceneFrameRecord],
    min_length_frames: int = 30,
    geometry: CARRADABEVGeometry | None = None,
) -> tuple[SingleTargetEpisode, ...]:
    """Derive strict continuous episodes containing one fixed pedestrian ID.

    Cars and cyclists remain in the records as distractors. A frame with zero
    pedestrians, multiple pedestrians, an out-of-BEV pedestrian, or a missing
    instance ID terminates the current episode.
    """

    if min_length_frames < 1:
        raise ValueError("min_length_frames must be positive")
    bev = geometry or CARRADABEVGeometry()
    grouped: dict[str, list[SceneFrameRecord]] = {}
    for record in records:
        grouped.setdefault(record.sequence_id, []).append(record)

    episodes: list[SingleTargetEpisode] = []
    for sequence_id, sequence_records in sorted(grouped.items()):
        current: list[SceneFrameRecord] = []

        def finalize() -> None:
            if len(current) < min_length_frames:
                return
            target_id = current[0].targets[0].instance_id
            if target_id is None:
                raise RuntimeError("single-target episode lost its instance ID")
            start = current[0].frame_id
            end = current[-1].frame_id
            episodes.append(
                SingleTargetEpisode(
                    episode_id=f"{sequence_id}__{target_id}__{start:06d}_{end:06d}",
                    sequence_id=sequence_id,
                    target_instance_id=target_id,
                    start_frame=start,
                    end_frame=end,
                    frame_count=len(current),
                )
            )

        for record in sorted(sequence_records, key=lambda item: item.frame_id):
            valid = len(record.targets) == 1 and bev.contains(record.targets[0].x_m, record.targets[0].y_m)
            target_id = record.targets[0].instance_id if valid else None
            continuous = (
                bool(current)
                and record.frame_id == current[-1].frame_id + 1
                and target_id == current[-1].targets[0].instance_id
            )
            if not valid:
                finalize()
                current = []
            elif continuous:
                current.append(record)
            else:
                finalize()
                current = [record]
        finalize()
    return tuple(episodes)


def apply_single_target_episode_manifest(
    records: list[SceneFrameRecord],
    episodes: tuple[SingleTargetEpisode, ...] | list[SingleTargetEpisode],
    geometry: CARRADABEVGeometry | None = None,
) -> list[SceneFrameRecord]:
    bev = geometry or CARRADABEVGeometry()
    by_key = {(record.sequence_id, record.frame_id): record for record in records}
    selected: list[SceneFrameRecord] = []
    claimed: set[tuple[str, int]] = set()
    for episode in episodes:
        expected = episode.end_frame - episode.start_frame + 1
        if expected != episode.frame_count:
            raise ValueError(f"non-contiguous episode manifest: {episode.episode_id}")
        for frame_id in range(episode.start_frame, episode.end_frame + 1):
            key = (episode.sequence_id, frame_id)
            if key in claimed:
                raise ValueError(f"overlapping episode frame: {key}")
            record = by_key.get(key)
            if record is None:
                raise ValueError(f"episode frame missing from synchronized records: {key}")
            valid = [target for target in record.targets if bev.contains(target.x_m, target.y_m)]
            if (
                len(record.targets) != 1
                or len(valid) != 1
                or valid[0].instance_id != episode.target_instance_id
            ):
                raise ValueError(f"episode target contract violated at {key}")
            selected.append(
                replace(
                    record,
                    episode_id=episode.episode_id,
                    episode_start_frame=episode.start_frame,
                    target_instance_id=episode.target_instance_id,
                )
            )
            claimed.add(key)
    return selected


def load_single_target_episode_manifest(
    path: str | Path,
    split: str,
) -> tuple[SingleTargetEpisode, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    key = "locked_test" if split in {"test", "locked_test"} else split
    rows = payload["splits"][key]
    return tuple(
        SingleTargetEpisode(
            episode_id=str(row["episode_id"]),
            sequence_id=str(row["sequence_id"]),
            target_instance_id=str(row["target_instance_id"]),
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            frame_count=int(row["frame_count"]),
        )
        for row in rows
    )


def build_all_scene_index(
    data_root: str | Path,
    protocol: CARRADAProtocol,
    split: str,
    annotations_path: str | Path | None = None,
) -> list[SceneFrameRecord]:
    root = Path(data_root)
    annotation_file = Path(annotations_path) if annotations_path is not None else root / "annotations_frame_oriented.json"
    annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
    records: list[SceneFrameRecord] = []
    for sequence_id in protocol.sequence_ids(split):
        sequence_root = root / sequence_id
        ra_by_frame = canonical_frame_files(sequence_root / "range_angle_numpy", ".npy")
        image_by_frame = canonical_frame_files(sequence_root / "camera_images", ".jpg")
        sequence_annotations = annotations.get(sequence_id, {})
        synchronized = sorted(set(ra_by_frame) & set(image_by_frame) & {int(value) for value in sequence_annotations})
        for frame_id in synchronized:
            targets, distractors = _targets_from_frame(sequence_annotations[f"{frame_id:06d}"])
            records.append(
                SceneFrameRecord(
                    sequence_id=sequence_id,
                    frame_id=frame_id,
                    targets=targets,
                    distractor_count=distractors,
                    image_path=image_by_frame[frame_id],
                )
            )
    return records


class RadarBEVWindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[SceneFrameRecord],
        cache_root: str | Path,
        clip_length: int = 8,
        geometry: CARRADABEVGeometry | None = None,
        gaussian_sigma_cells: float = 1.5,
    ) -> None:
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        self.records = list(records)
        self.cache_root = Path(cache_root)
        self.clip_length = int(clip_length)
        self.geometry = geometry or CARRADABEVGeometry()
        self.gaussian_sigma_cells = float(gaussian_sigma_cells)
        self._cache: dict[str, np.ndarray] = {}
        self._frame_to_index: dict[str, dict[int, int]] = {}
        self._sequence_frames: dict[str, list[int]] = {}
        for record in self.records:
            self._sequence_frames.setdefault(record.sequence_id, []).append(record.frame_id)
        for sequence_id, frame_ids in self._sequence_frames.items():
            ordered = sorted(set(frame_ids))
            cached_frames = np.load(self.cache_root / f"{sequence_id}_frames.npy").astype(np.int64).tolist()
            if not set(ordered).issubset(cached_frames):
                raise ValueError(f"record frames are not a subset of the cache for {sequence_id}")
            self._sequence_frames[sequence_id] = cached_frames
            self._frame_to_index[sequence_id] = {
                frame_id: index for index, frame_id in enumerate(cached_frames)
            }

    def __len__(self) -> int:
        return len(self.records)

    def _sequence_cache(self, sequence_id: str) -> np.ndarray:
        if sequence_id not in self._cache:
            self._cache[sequence_id] = np.load(self.cache_root / f"{sequence_id}.npy", mmap_mode="r")
        return self._cache[sequence_id]

    def _target(self, record: SceneFrameRecord) -> tuple[np.ndarray, list[Target]]:
        target = np.zeros((self.geometry.x_bins, self.geometry.y_bins), dtype=np.float32)
        rows, columns = np.indices(target.shape, dtype=np.float32)
        valid: list[Target] = []
        for person in record.targets:
            if not self.geometry.contains(person.x_m, person.y_m):
                continue
            x_index, y_index = self.geometry.metric_to_index(person.x_m, person.y_m)
            squared = (rows - x_index) ** 2 + (columns - y_index) ** 2
            gaussian = np.exp(-0.5 * squared / self.gaussian_sigma_cells**2).astype(np.float32)
            target = np.maximum(target, gaussian)
            valid.append(person)
        return target[None], valid

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        current = self._frame_to_index[record.sequence_id][record.frame_id]
        history_start = 0
        if record.episode_start_frame is not None:
            history_start = self._frame_to_index[record.sequence_id][record.episode_start_frame]
        positions = [
            max(history_start, current - (self.clip_length - 1 - offset))
            for offset in range(self.clip_length)
        ]
        amplitude = np.asarray(self._sequence_cache(record.sequence_id)[positions], dtype=np.float32)
        mti = np.diff(amplitude, axis=0, prepend=amplitude[:1])
        radar = np.stack((amplitude, mti), axis=1).astype(np.float32, copy=False)
        target, valid = self._target(record)
        single = valid[0] if len(valid) == 1 else None
        return {
            "radar": torch.from_numpy(radar.copy()),
            "target": torch.from_numpy(target.copy()),
            "target_count": torch.tensor(len(valid), dtype=torch.int64),
            "raw_pedestrian_count": torch.tensor(record.pedestrian_count, dtype=torch.int64),
            "presence": torch.tensor(bool(valid), dtype=torch.float32),
            "single_x_m": torch.tensor(single.x_m if single else float("nan"), dtype=torch.float32),
            "single_y_m": torch.tensor(single.y_m if single else float("nan"), dtype=torch.float32),
            "sequence_id": record.sequence_id,
            "frame_id": torch.tensor(record.frame_id, dtype=torch.int64),
            "distractor_count": torch.tensor(record.distractor_count, dtype=torch.int64),
        }


class AllSceneRGBDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[SceneFrameRecord],
        clip_length: int = 8,
        image_size: int = 192,
        angle_bins: int = 181,
        half_fov_rad: float = math.pi / 2.0,
        training: bool = False,
        geometry: CARRADABEVGeometry | None = None,
        context_records: list[SceneFrameRecord] | None = None,
    ) -> None:
        self.records = list(records)
        self.clip_length = int(clip_length)
        self.geometry = geometry or CARRADABEVGeometry()
        self.angle_centers = torch.linspace(-half_fov_rad, half_fov_rad, angle_bins)
        steps: list[object] = [transforms.Resize((image_size, image_size))]
        if training:
            steps.append(transforms.ColorJitter(brightness=0.20, contrast=0.15, saturation=0.10, hue=0.02))
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.transform = transforms.Compose(steps)
        self._sequence_records: dict[str, list[SceneFrameRecord]] = {}
        for record in context_records if context_records is not None else self.records:
            self._sequence_records.setdefault(record.temporal_group, []).append(record)
        self._position: dict[tuple[str, int], int] = {}
        for temporal_group, sequence_records in self._sequence_records.items():
            sequence_records.sort(key=lambda item: item.frame_id)
            for position, record in enumerate(sequence_records):
                self._position[(temporal_group, record.frame_id)] = position

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        sequence_records = self._sequence_records[record.temporal_group]
        current = self._position[(record.temporal_group, record.frame_id)]
        images: list[torch.Tensor] = []
        for offset in range(self.clip_length):
            position = max(0, current - (self.clip_length - 1 - offset))
            with Image.open(sequence_records[position].image_path) as image:
                images.append(self.transform(image.convert("RGB")))
        valid = [target for target in record.targets if self.geometry.contains(target.x_m, target.y_m)]
        dense = torch.zeros_like(self.angle_centers)
        for target in valid:
            dense = torch.maximum(dense, torch.exp(-0.5 * ((self.angle_centers - target.angle_rad) / math.radians(1.0)) ** 2))
        bearing_target = dense / dense.sum().clamp_min(1e-8)
        single = valid[0] if len(valid) == 1 else None
        return {
            "images": torch.stack(images),
            "frame_valid": torch.ones(self.clip_length, dtype=torch.bool),
            "person_u_target": dense,
            "bearing_target": bearing_target,
            "valid_bearing_count": torch.tensor(len(valid), dtype=torch.int64),
            "theta_rad": torch.tensor(single.angle_rad if single else float("nan"), dtype=torch.float32),
            "x_m": torch.tensor(single.x_m if single else float("nan"), dtype=torch.float32),
            "y_m": torch.tensor(single.y_m if single else float("nan"), dtype=torch.float32),
            "sequence_id": record.sequence_id,
            "frame_id": torch.tensor(record.frame_id, dtype=torch.int64),
        }


class RadarRGBFusionDataset(Dataset[dict[str, object]]):
    """Paired causal radar/RGB evidence over the exact same frame records."""

    def __init__(
        self,
        records: list[SceneFrameRecord],
        cache_root: str | Path,
        radar_clip_length: int = 8,
        rgb_clip_length: int = 8,
        image_size: int = 192,
        training: bool = False,
        geometry: CARRADABEVGeometry | None = None,
        context_records: list[SceneFrameRecord] | None = None,
    ) -> None:
        self.radar = RadarBEVWindowDataset(records, cache_root, radar_clip_length, geometry)
        self.rgb = AllSceneRGBDataset(
            records,
            rgb_clip_length,
            image_size,
            training=training,
            geometry=geometry,
            context_records=context_records,
        )
        if len(self.radar) != len(self.rgb):
            raise RuntimeError("radar and RGB datasets must align")

    def __len__(self) -> int:
        return len(self.radar)

    def __getitem__(self, index: int) -> dict[str, object]:
        radar = self.radar[index]
        rgb = self.rgb[index]
        if int(radar["frame_id"]) != int(rgb["frame_id"]) or radar["sequence_id"] != rgb["sequence_id"]:
            raise RuntimeError("radar/RGB sample mismatch")
        return {**radar, "images": rgb["images"], "frame_valid": rgb["frame_valid"]}


def visual_checkpoint_score(stage: str, metrics: dict[str, float]) -> float:
    if stage == "pretrain":
        auroc = float(metrics["presence_auroc"])
        return 1.0 - auroc if np.isfinite(auroc) else float("inf")
    if stage == "adapt":
        return float(metrics["bearing_mae_deg"])
    raise ValueError(f"unsupported visual stage: {stage}")
