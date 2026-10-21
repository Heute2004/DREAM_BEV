"""Causal RGB windows for the paper RV-BearingNet on CRUW scenes."""

from __future__ import annotations

import math
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as transform_functional

from .allscene import CRUWBEVGeometry, SceneFrameRecord


def visual_checkpoint_score(
    stage: str,
    metrics: dict[str, float],
    selection_metric: str | None = None,
) -> float:
    """Return the validation quantity supervised by the active training stage."""
    if stage == "pretrain":
        return float(metrics["loss"])
    if stage == "adapt":
        metric = selection_metric or "bearing_mae"
        metric_keys = {
            "bearing_mae": "bearing_mae_deg",
            "lateral_mae": "lateral_mae_m",
            "lateral_p90": "lateral_p90_m",
        }
        if metric not in metric_keys:
            raise ValueError(f"unsupported adaptation selection metric: {metric}")
        return float(metrics[metric_keys[metric]])
    raise ValueError(f"unsupported visual training stage: {stage}")


class AllSceneRGBDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[SceneFrameRecord],
        clip_length: int = 8,
        image_size: int = 192,
        image_height: int | None = None,
        angle_bins: int = 181,
        half_fov_rad: float = 0.80,
        target_sigma_rad: float = math.radians(1.0),
        training: bool = False,
        augmentation_mode: str = "independent",
        geometry: CRUWBEVGeometry | None = None,
    ) -> None:
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        self.records = list(records)
        self.clip_length = clip_length
        self.angle_bins = angle_bins
        self.half_fov_rad = half_fov_rad
        self.target_sigma_rad = target_sigma_rad
        self.geometry = geometry or CRUWBEVGeometry()
        if augmentation_mode not in {"independent", "clip_consistent", "none"}:
            raise ValueError(f"unsupported augmentation mode: {augmentation_mode}")
        self.training = bool(training)
        self.augmentation_mode = augmentation_mode
        resize_height = image_size if image_height is None else int(image_height)
        augmentation: list[object] = [transforms.Resize((resize_height, image_size))]
        if training and augmentation_mode == "independent":
            augmentation.append(
                transforms.ColorJitter(brightness=0.20, contrast=0.15, saturation=0.10, hue=0.02)
            )
        augmentation.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.transform = transforms.Compose(augmentation)
        self.angle_centers = torch.linspace(-half_fov_rad, half_fov_rad, angle_bins)

    @staticmethod
    def _sample_clip_jitter() -> list[tuple[str, float]]:
        operations = [
            ("brightness", random.uniform(0.80, 1.20)),
            ("contrast", random.uniform(0.85, 1.15)),
            ("saturation", random.uniform(0.90, 1.10)),
            ("hue", random.uniform(-0.02, 0.02)),
        ]
        random.shuffle(operations)
        return operations

    @staticmethod
    def _apply_clip_jitter(image: Image.Image, operations: list[tuple[str, float]]) -> Image.Image:
        adjusted = image
        functions = {
            "brightness": transform_functional.adjust_brightness,
            "contrast": transform_functional.adjust_contrast,
            "saturation": transform_functional.adjust_saturation,
            "hue": transform_functional.adjust_hue,
        }
        for name, factor in operations:
            adjusted = functions[name](adjusted, factor)
        return adjusted

    def __len__(self) -> int:
        return len(self.records)

    def _history_path(self, record: SceneFrameRecord, frame_id: int):
        return record.image_path.parent / f"{frame_id:010d}.jpg"

    def _target(self, record: SceneFrameRecord) -> tuple[torch.Tensor, list[float]]:
        dense = torch.zeros(self.angle_bins, dtype=torch.float32)
        valid_angles: list[float] = []
        for pedestrian in record.pedestrians:
            if not self.geometry.contains(pedestrian.x_m, pedestrian.y_m):
                continue
            if not -self.half_fov_rad <= pedestrian.angle_rad <= self.half_fov_rad:
                continue
            gaussian = torch.exp(-0.5 * ((self.angle_centers - pedestrian.angle_rad) / self.target_sigma_rad) ** 2)
            gaussian = gaussian / gaussian.max().clamp_min(1e-8)
            dense = torch.maximum(dense, gaussian)
            valid_angles.append(pedestrian.angle_rad)
        return dense, valid_angles

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        images: list[torch.Tensor] = []
        clip_jitter = (
            self._sample_clip_jitter()
            if self.training and self.augmentation_mode == "clip_consistent"
            else None
        )
        for offset in range(self.clip_length):
            frame_id = max(
                record.sequence_start,
                record.frame_id - (self.clip_length - 1 - offset) * record.sample_stride,
            )
            path = self._history_path(record, frame_id)
            if not path.is_file():
                raise FileNotFoundError(f"missing CRUW RGB history frame: {path}")
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if clip_jitter is not None:
                    rgb = self._apply_clip_jitter(rgb, clip_jitter)
                images.append(self.transform(rgb))
        person_u_target, angles = self._target(record)
        single = len(angles) == 1
        theta = angles[0] if single else float("nan")
        matching = [
            pedestrian
            for pedestrian in record.pedestrians
            if single and abs(pedestrian.angle_rad - theta) < 1e-8
        ]
        x_m = matching[0].x_m if len(matching) == 1 else float("nan")
        y_m = matching[0].y_m if len(matching) == 1 else float("nan")
        bearing_target = person_u_target / person_u_target.sum().clamp_min(1e-8)
        return {
            "images": torch.stack(images),
            "frame_valid": torch.ones(self.clip_length, dtype=torch.bool),
            "person_u_target": person_u_target,
            "bearing_target": bearing_target,
            "valid_bearing_count": torch.tensor(len(angles), dtype=torch.int64),
            "theta_rad": torch.tensor(theta, dtype=torch.float32),
            "x_m": torch.tensor(x_m, dtype=torch.float32),
            "y_m": torch.tensor(y_m, dtype=torch.float32),
            "sequence_id": torch.tensor(record.sequence_id, dtype=torch.int64),
            "frame_id": torch.tensor(record.frame_id, dtype=torch.int64),
        }
