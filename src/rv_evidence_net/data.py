from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import functional as transforms

from .protocol import DEV_SPLIT, TRAIN_SPLIT, validate_base_train_dev_manifest


@dataclass(frozen=True)
class FrameRecord:
    sequence_id: str
    sequence_name: str
    frame_index: int
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class SequenceRecords:
    sequence_id: str
    sequence_name: str
    frames: tuple[FrameRecord, ...]
    partition: str | None = None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def stable_sequence_partition(sequence_id: str, validation_modulo: int = 10, validation_bucket: int = 0) -> str:
    """Return a deterministic sequence-level Train/InternalVal assignment."""

    if validation_modulo < 2:
        raise ValueError("validation_modulo must be at least two")
    digest = hashlib.sha256(sequence_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % validation_modulo
    return "InternalVal" if bucket == validation_bucket else "Train"


def load_student_sequences(
    dataset_root: Path,
    manifest_path: Path | None = None,
    profile_path: Path | None = None,
    validation_modulo: int = 10,
    validation_bucket: int = 0,
    protocol_manifest_path: Path | None = None,
    protocol_path: Path | None = None,
) -> tuple[SequenceRecords, ...]:
    """Load only frozen Train rows eligible for first-stage visual student training."""

    manifest_path = manifest_path or dataset_root / "visual_teacher_manifest.csv"
    profile_path = profile_path or dataset_root / "visual_teacher_profile.csv"
    profile_rows = _read_csv(profile_path)
    eligible_keys = {
        (row["sequence_name"], int(row["frame_index"]))
        for row in profile_rows
        if _is_true(row.get("eligible_for_visual_teacher_train", ""))
    }
    explicit_partitions = None
    if protocol_manifest_path is not None or protocol_path is not None:
        if protocol_manifest_path is None or protocol_path is None:
            raise ValueError("protocol manifest and protocol file must be provided together")
        explicit_partitions = validate_base_train_dev_manifest(protocol_manifest_path, protocol_path)
    grouped: dict[tuple[str, str, str | None], list[FrameRecord]] = defaultdict(list)
    for row in _read_csv(manifest_path):
        sequence_name = row["sequence_name"].strip().lower()
        if explicit_partitions is None and row.get("split", "").strip().lower() != "train":
            continue
        explicit_split = explicit_partitions.get(sequence_name) if explicit_partitions is not None else None
        if explicit_partitions is not None and explicit_split is None:
            continue
        frame_index = int(row["frame_index"])
        if (sequence_name, frame_index) not in eligible_keys and (explicit_split is None or explicit_split == TRAIN_SPLIT):
            continue
        sequence_id = f"{int(row['sequence_id']):03d}"
        partition = "Train" if explicit_split == TRAIN_SPLIT else "InternalVal" if explicit_split == DEV_SPLIT else None
        grouped[(sequence_id, sequence_name, partition)].append(
            FrameRecord(
                sequence_id=sequence_id,
                sequence_name=sequence_name,
                frame_index=frame_index,
                image_path=dataset_root / row["image_path"],
                label_path=dataset_root / row["yolo_label_path"],
            )
        )
    sequences = [
        SequenceRecords(sequence_id, sequence_name, tuple(sorted(frames, key=lambda frame: frame.frame_index)), partition)
        for (sequence_id, sequence_name, partition), frames in grouped.items()
    ]
    if not sequences:
        raise ValueError("No eligible student training sequences were found")
    sequences.sort(key=lambda sequence: int(sequence.sequence_id))
    return tuple(sequences)


class RGBClipDataset(Dataset[dict[str, Tensor | str]]):
    """Causal clips from one physical sequence, never crossing a sequence boundary."""

    def __init__(
        self,
        sequences: Iterable[SequenceRecords],
        partition: str,
        clip_length: int = 8,
        frame_stride: int = 1,
        image_size: int = 640,
        validation_modulo: int = 10,
        validation_bucket: int = 0,
        train_augmentation: str = "none",
    ) -> None:
        if clip_length < 1 or frame_stride < 1:
            raise ValueError("clip_length and frame_stride must both be positive")
        if partition not in {"Train", "InternalVal"}:
            raise ValueError("partition must be 'Train' or 'InternalVal'")
        if train_augmentation not in {"none", "mild_photometric"}:
            raise ValueError("train_augmentation must be 'none' or 'mild_photometric'")
        self.partition = partition
        self.train_augmentation = train_augmentation
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.samples: list[tuple[SequenceRecords, int]] = []
        required_history = (clip_length - 1) * frame_stride
        for sequence in sequences:
            assigned_partition = sequence.partition or stable_sequence_partition(sequence.sequence_id, validation_modulo, validation_bucket)
            if assigned_partition != partition:
                continue
            for end_index in range(required_history, len(sequence.frames)):
                self.samples.append((sequence, end_index))
        if not self.samples:
            raise ValueError(f"No {partition} clips available with the requested temporal configuration")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _read_box(label_path: Path, image_size: int) -> tuple[Tensor, bool]:
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            return torch.zeros(4, dtype=torch.float32), False
        tokens = content.splitlines()[0].split()
        if len(tokens) != 5 or tokens[0] != "0":
            raise ValueError(f"Expected one-class YOLO person label in {label_path}")
        center_x, center_y, width, height = (float(value) for value in tokens[1:])
        box = torch.tensor(
            (
                (center_x - width * 0.5) * image_size,
                (center_y - height * 0.5) * image_size,
                (center_x + width * 0.5) * image_size,
                (center_y + height * 0.5) * image_size,
            ),
            dtype=torch.float32,
        )
        return box, True

    def _read_image(self, image_path: Path) -> Tensor:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            resized = transforms.resize(rgb, [self.image_size, self.image_size], antialias=True)
            return transforms.to_tensor(resized)

    def _augment_clip(self, images: Tensor) -> Tensor:
        if self.partition != "Train" or self.train_augmentation == "none":
            return images
        brightness = float(torch.empty(()).uniform_(0.85, 1.15))
        contrast = float(torch.empty(()).uniform_(0.85, 1.15))
        saturation = float(torch.empty(()).uniform_(0.90, 1.10))
        return torch.stack(
            [
                transforms.adjust_saturation(
                    transforms.adjust_contrast(transforms.adjust_brightness(image, brightness), contrast), saturation
                )
                for image in images
            ]
        )

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sequence, end_index = self.samples[index]
        first_index = end_index - (self.clip_length - 1) * self.frame_stride
        frame_indices = range(first_index, end_index + 1, self.frame_stride)
        frames = [sequence.frames[frame_index] for frame_index in frame_indices]
        images = self._augment_clip(torch.stack([self._read_image(frame.image_path) for frame in frames]))
        labels = [self._read_box(frame.label_path, self.image_size) for frame in frames]
        boxes = torch.stack([label[0] for label in labels]).unsqueeze(1)
        valid = torch.tensor([label[1] for label in labels], dtype=torch.bool).unsqueeze(1)
        return {
            "images": images,
            "boxes": boxes,
            "valid_mask": valid,
            "sequence_id": sequence.sequence_id,
            "sequence_name": sequence.sequence_name,
            "end_frame_index": torch.tensor(frames[-1].frame_index, dtype=torch.int64),
        }


def build_sequence_balanced_sampler(dataset: RGBClipDataset) -> WeightedRandomSampler:
    """Reduce domination by long, highly correlated recordings during an epoch."""

    counts: dict[str, int] = defaultdict(int)
    for sequence, _ in dataset.samples:
        counts[sequence.sequence_id] += 1
    weights = torch.tensor([1.0 / counts[sequence.sequence_id] for sequence, _ in dataset.samples], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)



