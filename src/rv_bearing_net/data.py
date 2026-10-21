from __future__ import annotations

import csv
import hashlib
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as transforms

from .training_teacher import load_teacher_sequence


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def stable_sequence_partition(
    sequence: str,
    *,
    validation_modulo: int = 10,
    validation_bucket: int = 0,
) -> str:
    if validation_modulo < 2:
        raise ValueError("validation_modulo must be at least two")
    digest = hashlib.sha256(sequence.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % validation_modulo
    return "InternalVal" if bucket == validation_bucket else "Train"


class ConditionRecordLike(Protocol):
    global_route: str
    visual_condition: str
    visual_reliability: str
    weather_surface: str


class ConditionTimelineLike(Protocol):
    def lookup(self, sequence: str, frame_id: int) -> ConditionRecordLike: ...


@dataclass(frozen=True)
class BearingFrameRecord:
    sequence: str
    frame_id: int
    rgb_path: Path
    target_xy_m: tuple[float, float]
    label_valid: bool
    source_is_radar: bool
    person_label_path: Path | None


def coordinate_supervision_weight(condition: ConditionRecordLike) -> float:
    visual_condition = str(condition.visual_condition).strip().lower()
    reliability = str(condition.visual_reliability).strip().lower()
    if visual_condition not in {"normal", "normal_visibility"}:
        return 0.0
    if reliability == "usable":
        return 1.0
    if reliability == "cautious":
        return 0.5
    return 0.0


class RGBBearingClipDataset(Dataset[dict[str, object]]):
    """Causal RGB clips joined to frozen BEV and visual-evidence person labels."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        manifest_path: str | Path,
        visual_teacher_root: str | Path,
        visual_manifest_path: str | Path,
        condition_timeline: ConditionTimelineLike,
        partition: str,
        clip_length: int = 8,
        frame_stride: int = 1,
        image_height: int = 256,
        image_width: int = 448,
        validation_modulo: int = 10,
        validation_bucket: int = 0,
        excluded_sequences: tuple[str, ...] = (),
        train_augmentation: str = "mild_photometric",
        preload_rgb: bool = False,
        rgb_cache_size: int = 256,
        training_teacher_cache_root: str | Path | None = None,
        teacher_cache_size: int = 2,
    ) -> None:
        if partition not in {"Train", "InternalVal", "Eval"}:
            raise ValueError("partition must be Train, InternalVal, or Eval")
        if clip_length < 1 or frame_stride < 1:
            raise ValueError("clip_length and frame_stride must be positive")
        if image_height < 16 or image_width < 16:
            raise ValueError("image dimensions must be at least 16")
        if train_augmentation not in {"none", "mild_photometric"}:
            raise ValueError(
                "train_augmentation must be none or mild_photometric"
            )
        if rgb_cache_size < 0:
            raise ValueError("rgb_cache_size must be non-negative")
        if teacher_cache_size < 1:
            raise ValueError("teacher_cache_size must be positive")
        self.partition = partition
        self.clip_length = int(clip_length)
        self.frame_stride = int(frame_stride)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.condition_timeline = condition_timeline
        self.train_augmentation = train_augmentation
        self.preload_rgb = bool(preload_rgb)
        self.rgb_cache_size = int(rgb_cache_size)
        self.training_teacher_cache_root = (
            Path(training_teacher_cache_root)
            if training_teacher_cache_root is not None
            else None
        )
        self.teacher_cache_size = int(teacher_cache_size)
        self._training_teacher_cache: OrderedDict[
            str, tuple[dict[str, object], dict[int, int]]
        ] = OrderedDict()
        data_root = Path(data_root)
        visual_teacher_root = Path(visual_teacher_root)
        excluded = {value.strip().lower() for value in excluded_sequences}

        person_labels: dict[tuple[str, int], Path] = {}
        with Path(visual_manifest_path).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                person_labels[
                    (
                        row["sequence_name"].strip().lower(),
                        int(row["frame_index"]),
                    )
                ] = visual_teacher_root / row["yolo_label_path"]

        grouped: dict[str, list[BearingFrameRecord]] = defaultdict(list)
        with Path(manifest_path).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            for row in csv.DictReader(handle):
                sequence = row["sequence"].strip().lower()
                if sequence in excluded or _as_bool(
                    row.get("sequence_excluded", False)
                ):
                    continue
                source_split = row["split"].strip().lower()
                if partition == "Eval":
                    if source_split not in {
                        "eval",
                        "held-out eval",
                        "held_out_eval",
                        "heldout eval",
                    }:
                        continue
                else:
                    if source_split != "train":
                        continue
                    if stable_sequence_partition(
                        sequence,
                        validation_modulo=validation_modulo,
                        validation_bucket=validation_bucket,
                    ) != partition:
                        continue
                frame_id = int(row["frame_id"])
                grouped[sequence].append(
                    BearingFrameRecord(
                        sequence=sequence,
                        frame_id=frame_id,
                        rgb_path=data_root / row["rgb_path"],
                        target_xy_m=(
                            float(row["final_bev_x_m"]),
                            float(row["final_bev_y_m"]),
                        ),
                        label_valid=_as_bool(
                            row.get("final_bev_label_valid", False)
                        ),
                        source_is_radar=_as_bool(
                            row.get("final_bev_source_is_radar", False)
                        ),
                        person_label_path=person_labels.get(
                            (sequence, frame_id)
                        ),
                    )
                )
        self.sequences = {
            sequence: tuple(
                sorted(records, key=lambda record: record.frame_id)
            )
            for sequence, records in sorted(grouped.items())
        }
        self.samples: list[tuple[str, int]] = []
        for sequence, records in self.sequences.items():
            for position, record in enumerate(records):
                if record.label_valid:
                    self.samples.append((sequence, position))
        if not self.samples:
            raise ValueError(
                f"no valid RGB bearing samples for partition={partition}"
            )
        self._preloaded_images: dict[Path, Tensor] = {}
        self._image_cache: OrderedDict[Path, Tensor] = OrderedDict()
        if self.preload_rgb:
            unique_paths = {
                record.rgb_path
                for records in self.sequences.values()
                for record in records
            }
            for path in sorted(unique_paths):
                self._preloaded_images[path] = self._read_image_uint8(path)

    def __len__(self) -> int:
        return len(self.samples)

    def _teacher_fields(
        self, sequence: str, frame_id: int
    ) -> dict[str, Tensor]:
        root = self.training_teacher_cache_root
        if root is None:
            return {}
        cached = self._training_teacher_cache.get(sequence)
        if cached is None:
            cached = load_teacher_sequence(root / f"{sequence}.npz")
            self._training_teacher_cache[sequence] = cached
            while len(self._training_teacher_cache) > self.teacher_cache_size:
                self._training_teacher_cache.popitem(last=False)
        else:
            self._training_teacher_cache.move_to_end(sequence)
        values, frame_index = cached
        if frame_id not in frame_index:
            raise KeyError(
                f"teacher cache does not cover {sequence} frame {frame_id}"
            )
        index = frame_index[frame_id]
        return {
            "geometry_theta_rad": torch.tensor(
                values["geometry_theta_rad"][index], dtype=torch.float32
            ),
            "geometry_sigma_rad": torch.tensor(
                values["geometry_sigma_rad"][index], dtype=torch.float32
            ),
            "geometry_valid": torch.tensor(
                values["geometry_valid"][index], dtype=torch.bool
            ),
            "radar_candidate_theta_rad": torch.as_tensor(
                values["candidate_theta_rad"][index], dtype=torch.float32
            ),
            "radar_candidate_support": torch.as_tensor(
                values["candidate_support"][index], dtype=torch.float32
            ),
            "radar_candidate_valid": torch.as_tensor(
                values["candidate_valid"][index], dtype=torch.bool
            ),
            "radar_candidate_positive_index": torch.tensor(
                values["candidate_positive_index"][index], dtype=torch.int64
            ),
        }

    def _history_positions(self, end_position: int) -> list[int]:
        first = end_position - (
            self.clip_length - 1
        ) * self.frame_stride
        return [
            max(0, first + offset * self.frame_stride)
            for offset in range(self.clip_length)
        ]

    def _read_image_uint8(self, path: Path) -> Tensor:
        if not path.is_file():
            raise FileNotFoundError(f"missing RGB frame: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            resized = transforms.resize(
                rgb,
                [self.image_height, self.image_width],
                antialias=True,
            )
            return transforms.pil_to_tensor(resized)

    def _read_image(self, path: Path) -> Tensor:
        values = self._preloaded_images.get(path)
        if values is None:
            values = self._image_cache.get(path)
            if values is not None:
                self._image_cache.move_to_end(path)
            else:
                values = self._read_image_uint8(path)
                if self.rgb_cache_size:
                    self._image_cache[path] = values
                    while len(self._image_cache) > self.rgb_cache_size:
                        self._image_cache.popitem(last=False)
        return values.float().div_(255.0)

    @staticmethod
    def _read_person_u(path: Path | None) -> tuple[float, bool]:
        if path is None or not path.is_file():
            return 0.5, False
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            tokens = line.split()
            if len(tokens) == 5 and tokens[0] == "0":
                center_u = min(1.0, max(0.0, float(tokens[1])))
                return center_u, True
        return 0.5, False

    def _augment_clip(self, images: Tensor) -> Tensor:
        if (
            self.partition != "Train"
            or self.train_augmentation == "none"
        ):
            return images
        brightness = float(torch.empty(()).uniform_(0.85, 1.15))
        contrast = float(torch.empty(()).uniform_(0.85, 1.15))
        saturation = float(torch.empty(()).uniform_(0.90, 1.10))
        return torch.stack(
            [
                transforms.adjust_saturation(
                    transforms.adjust_contrast(
                        transforms.adjust_brightness(image, brightness),
                        contrast,
                    ),
                    saturation,
                )
                for image in images
            ]
        )

    def __getitem__(self, item: int) -> dict[str, object]:
        sequence, end_position = self.samples[item]
        records = self.sequences[sequence]
        positions = self._history_positions(end_position)
        history = [records[position] for position in positions]
        current = records[end_position]
        condition = self.condition_timeline.lookup(
            sequence,
            current.frame_id,
        )
        person_targets = [
            self._read_person_u(record.person_label_path)
            for record in history
        ]
        xy_history = torch.tensor(
            [record.target_xy_m for record in history],
            dtype=torch.float32,
        )
        theta_history = torch.atan2(
            xy_history[:, 0],
            xy_history[:, 1].clamp_min(1e-3),
        )
        label_valid_history = torch.tensor(
            [record.label_valid for record in history],
            dtype=torch.bool,
        )
        coordinate_weight = coordinate_supervision_weight(condition)
        return {
            "images": self._augment_clip(
                torch.stack(
                    [self._read_image(record.rgb_path) for record in history]
                )
            ),
            "target_xy_history_m": xy_history,
            "target_theta_history": theta_history,
            "target_valid_history": label_valid_history,
            "person_u_history": torch.tensor(
                [value for value, _ in person_targets],
                dtype=torch.float32,
            ),
            "person_valid_history": torch.tensor(
                [valid for _, valid in person_targets],
                dtype=torch.bool,
            ),
            "coordinate_weight": torch.tensor(
                coordinate_weight,
                dtype=torch.float32,
            ),
            "sample_weight": torch.tensor(
                0.75 if current.source_is_radar else 1.0,
                dtype=torch.float32,
            ),
            "history_frame_ids": torch.tensor(
                [record.frame_id for record in history],
                dtype=torch.int64,
            ),
            "frame_id": torch.tensor(
                current.frame_id,
                dtype=torch.int64,
            ),
            "sequence": sequence,
            "condition": {
                "global_route": condition.global_route,
                "visual_condition": condition.visual_condition,
                "visual_reliability": condition.visual_reliability,
                "weather_surface": condition.weather_surface,
            },
            **self._teacher_fields(sequence, current.frame_id),
        }


class SequenceGroupedSampler(Sampler[int]):
    def __init__(
        self,
        dataset: RGBBearingClipDataset,
        *,
        seed: int,
        shuffle_sequences: bool,
    ) -> None:
        self.seed = int(seed)
        self.shuffle_sequences = bool(shuffle_sequences)
        self.epoch = 0
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, (sequence, _) in enumerate(dataset.samples):
            grouped[sequence].append(index)
        self.grouped_indices = {
            sequence: tuple(indices)
            for sequence, indices in sorted(grouped.items())
        }

    def __iter__(self):
        sequences = list(self.grouped_indices)
        if self.shuffle_sequences:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(
                len(sequences),
                generator=generator,
            ).tolist()
            sequences = [sequences[index] for index in order]
        for sequence in sequences:
            yield from self.grouped_indices[sequence]

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.grouped_indices.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)



