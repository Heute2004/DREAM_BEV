from __future__ import annotations

import bisect
import csv
import json
import random
import re
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from .rv_bearing_cache import load_rv_bearing_cache


_FRAME_PATTERN = re.compile(r"(seq_\d{3}).*?frame_(\d{4})", re.IGNORECASE)


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def training_eligibility_from_row(row: dict[str, str]) -> bool:
    """Read the canonical eligibility flag with the IF manifest fallback."""
    if "radar_training_eligible" in row:
        return _as_bool(row["radar_training_eligible"])
    if "m3_later_stage_eligible" in row:
        return _as_bool(row["m3_later_stage_eligible"])
    raise KeyError(
        "manifest row must provide radar_training_eligible or "
        "m3_later_stage_eligible"
    )


def _normalized_visual_condition(value: str) -> str:
    aliases = {
        "normal_visibility": "normal",
        "normal_visible": "normal",
        "rain_or_wet": "rain_wet",
        "wet_or_rain": "rain_wet",
    }
    value = str(value).strip().lower()
    return aliases.get(value, value or "unknown")


def _route_with_surface(route: str, surface: str) -> str:
    route = route.strip().lower()
    surface = surface.strip().lower()
    if route != "outdoor_adverse":
        return route
    if surface == "wet_return":
        return "outdoor_wet_return"
    if surface == "dry":
        return "outdoor_dry_complex"
    return route


@dataclass(frozen=True)
class ConditionRecord:
    sequence: str
    frame_id: int
    global_route: str
    visual_condition: str
    visual_reliability: str
    weather_surface: str


class ConditionTimeline:
    """Hold the most recent causal 2 Hz VLM condition at each 10 Hz frame."""

    def __init__(self, records: Iterable[ConditionRecord]) -> None:
        grouped: dict[str, list[ConditionRecord]] = defaultdict(list)
        for record in records:
            grouped[record.sequence].append(record)
        self.records = {
            sequence: tuple(sorted(values, key=lambda item: item.frame_id))
            for sequence, values in grouped.items()
        }
        if not self.records:
            raise ValueError("condition timeline is empty")

    @staticmethod
    def _response_to_record(
        image: str,
        response: dict[str, object],
    ) -> ConditionRecord | None:
        if "augmentized_data" in image.lower():
            return None
        match = _FRAME_PATTERN.search(image.replace("\\", "/"))
        if match is None:
            return None
        surface = str(response.get("weather_surface_prior", "unknown"))
        # New RGB policy predictions emit the executable route directly.
        # global_route remains an archived-prediction fallback only.
        raw_route = response.get("expert_route", response.get("global_route", "unknown"))
        route = _route_with_surface(str(raw_route), surface)
        return ConditionRecord(
            sequence=match.group(1).lower(),
            frame_id=int(match.group(2)),
            global_route=route,
            visual_condition=_normalized_visual_condition(
                str(response.get("visual_condition", "unknown"))
            ),
            visual_reliability=str(
                response.get("visual_reliability_prior", "unknown")
            )
            .strip()
            .lower(),
            weather_surface=surface.strip().lower() or "unknown",
        )

    @classmethod
    def _annotation_to_record(
        cls,
        annotation: dict[str, object],
    ) -> ConditionRecord | None:
        image = str(annotation.get("image", ""))
        conversations = annotation.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            return None
        assistant = conversations[-1]
        if not isinstance(assistant, dict):
            return None
        response = json.loads(str(assistant.get("value", "{}")))
        if not isinstance(response, dict):
            return None
        return cls._response_to_record(image, response)

    @classmethod
    def _prediction_to_record(
        cls,
        prediction_row: dict[str, object],
    ) -> ConditionRecord | None:
        image = str(prediction_row.get("image", ""))
        prediction = prediction_row.get("prediction")
        if isinstance(prediction, str):
            prediction = json.loads(prediction)
        if not isinstance(prediction, dict):
            return None
        return cls._response_to_record(image, prediction)

    @classmethod
    def from_jsonl(cls, paths: Iterable[str | Path]) -> "ConditionTimeline":
        records: list[ConditionRecord] = []
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = cls._annotation_to_record(json.loads(line))
                    if record is not None:
                        records.append(record)
        return cls(records)

    @classmethod
    def from_predictions_jsonl(
        cls,
        paths: Iterable[str | Path],
    ) -> "ConditionTimeline":
        records: list[ConditionRecord] = []
        for path in paths:
            with Path(path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = cls._prediction_to_record(json.loads(line))
                    if record is not None:
                        records.append(record)
        return cls(records)

    @classmethod
    def from_zip(cls, zip_path: str | Path, members: Iterable[str]) -> "ConditionTimeline":
        records: list[ConditionRecord] = []
        with zipfile.ZipFile(zip_path) as archive:
            for member in members:
                with archive.open(member) as handle:
                    for raw_line in handle:
                        record = cls._annotation_to_record(json.loads(raw_line))
                        if record is not None:
                            records.append(record)
        return cls(records)

    def lookup(self, sequence: str, frame_id: int) -> ConditionRecord:
        values = self.records.get(sequence)
        if values is None:
            raise KeyError(f"no VLM condition records for {sequence}")
        frame_ids = [item.frame_id for item in values]
        index = bisect.bisect_right(frame_ids, int(frame_id)) - 1
        if index < 0:
            index = 0
        return values[index]


@dataclass(frozen=True)
class _FrameRecord:
    sequence: str
    frame_id: int
    rgb_path: Path
    ra_path: Path
    ra_mti_path: Path
    target_xy_m: tuple[float, float]
    source_is_radar: bool
    input_clean: bool
    eligible: bool


class RVFusionCachedDataset(Dataset[dict[str, object]]):
    """Causal radar clips joined to frozen RGB evidence caches."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        manifest_path: str | Path,
        visual_cache_root: str | Path,
        condition_timeline: ConditionTimeline,
        rv_bearing_cache_root: str | Path | None = None,
        secondary_rv_bearing_cache_root: str | Path | None = None,
        radar_health_cache_root: str | Path | None = None,
        split: str,
        radar_clip_length: int = 16,
        visual_clip_length: int = 8,
        db_center: float = 80.0,
        db_scale: float = 30.0,
        sequence_cache_size: int = 1,
        return_radar: bool = True,
        return_rgb: bool = False,
        rgb_image_size: int = 384,
        sample_stride: int = 1,
        excluded_sequences: Iterable[str] = (),
    ) -> None:
        if radar_clip_length < 1 or visual_clip_length < 1:
            raise ValueError("clip lengths must be positive")
        if sequence_cache_size < 1:
            raise ValueError("sequence_cache_size must be positive")
        if rgb_image_size < 32 or sample_stride < 1:
            raise ValueError("rgb_image_size and sample_stride must be positive")
        self.data_root = Path(data_root)
        self.visual_cache_root = Path(visual_cache_root)
        self.rv_bearing_cache_root = (
            None
            if rv_bearing_cache_root is None
            else Path(rv_bearing_cache_root)
        )
        self.secondary_rv_bearing_cache_root = (
            None
            if secondary_rv_bearing_cache_root is None
            else Path(secondary_rv_bearing_cache_root)
        )
        self.radar_health_cache_root = (
            None
            if radar_health_cache_root is None
            else Path(radar_health_cache_root)
        )
        self.condition_timeline = condition_timeline
        self.radar_clip_length = radar_clip_length
        self.visual_clip_length = visual_clip_length
        self.db_center = db_center
        self.db_scale = db_scale
        self.sequence_cache_size = sequence_cache_size
        self.return_radar = bool(return_radar)
        self.return_rgb = bool(return_rgb)
        self.rgb_image_size = int(rgb_image_size)
        self.sample_stride = int(sample_stride)
        excluded = {
            str(sequence).strip().lower()
            for sequence in excluded_sequences
            if str(sequence).strip()
        }
        grouped: dict[str, list[_FrameRecord]] = defaultdict(list)
        with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"].strip().lower() != split.strip().lower():
                    continue
                if row["sequence"].strip().lower() in excluded:
                    continue
                input_clean = (
                    not _as_bool(row["sequence_excluded"])
                    and _as_bool(row["radar_input_range_eligible"])
                    and _as_bool(row["radar_quality_valid"])
                    and not _as_bool(row["red_band_anomaly"])
                    and not _as_bool(row["blue_dropout_anomaly"])
                )
                label_valid = _as_bool(row["final_bev_label_valid"])
                grouped[row["sequence"]].append(
                    _FrameRecord(
                        sequence=row["sequence"],
                        frame_id=int(row["frame_id"]),
                        rgb_path=self.data_root / row["rgb_path"],
                        ra_path=self.data_root / row["ra_path"],
                        ra_mti_path=self.data_root / row["ra_mti_path"],
                        target_xy_m=(float(row["final_bev_x_m"]), float(row["final_bev_y_m"])),
                        source_is_radar=_as_bool(row["final_bev_source_is_radar"]),
                        input_clean=input_clean,
                        eligible=input_clean
                        and label_valid
                        and training_eligibility_from_row(row),
                    )
                )
        self.sequences = {
            sequence: tuple(sorted(records, key=lambda item: item.frame_id))
            for sequence, records in grouped.items()
        }
        self.samples: list[tuple[str, int]] = []
        for sequence, records in sorted(self.sequences.items()):
            for index, record in enumerate(records):
                radar_positions = self._history_positions(index, self.radar_clip_length)
                if (
                    index % self.sample_stride == 0
                    and record.eligible
                    and all(records[position].input_clean for position in radar_positions)
                ):
                    self.samples.append((sequence, index))
        if not self.samples:
            raise ValueError(f"no eligible RVFusion samples for split={split}")
        self._visual_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._rv_bearing_cache: OrderedDict[
            str,
            dict[str, np.ndarray],
        ] = OrderedDict()
        self._secondary_rv_bearing_cache: OrderedDict[
            str,
            dict[str, np.ndarray],
        ] = OrderedDict()
        self._radar_cache: OrderedDict[str, Tensor] = OrderedDict()
        self._health_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _read_radar(self, path: Path) -> Tensor:
        values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if values.shape != (128, 128):
            raise ValueError(f"radar map must be 128x128: {path}")
        values = np.nan_to_num(
            values,
            nan=self.db_center,
            posinf=self.db_center + self.db_scale,
            neginf=self.db_center - self.db_scale,
        )
        return torch.from_numpy(np.clip((values - self.db_center) / self.db_scale, -3.0, 3.0))

    def _visual_for_sequence(self, sequence: str) -> dict[str, np.ndarray]:
        cached = self._visual_cache.get(sequence)
        if cached is not None:
            self._visual_cache.move_to_end(sequence)
            return cached
        path = self.visual_cache_root / f"{sequence}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen visual evidence cache: {path}")
        with np.load(path, allow_pickle=False) as payload:
            cached = {name: payload[name] for name in payload.files}
        required = {
            "frame_ids",
            "boxes",
            "raw_scores",
            "raw_tokens",
            "temporal_tokens",
            "temporal_support",
            "valid_mask",
        }
        if not required.issubset(cached):
            raise ValueError(f"visual cache lacks {sorted(required - set(cached))}: {path}")
        frame_ids = cached["frame_ids"].astype(np.int64, copy=False)
        frame_index = np.full(int(frame_ids.max()) + 1, -1, dtype=np.int64)
        frame_index[frame_ids] = np.arange(frame_ids.size, dtype=np.int64)
        cached["_frame_index"] = frame_index
        self._visual_cache[sequence] = cached
        while len(self._visual_cache) > self.sequence_cache_size:
            self._visual_cache.popitem(last=False)
        return cached

    def _radar_for_sequence(self, sequence: str) -> Tensor:
        cached = self._radar_cache.get(sequence)
        if cached is not None:
            self._radar_cache.move_to_end(sequence)
            return cached
        records = self.sequences[sequence]
        cached = torch.stack(
            [
                torch.stack((self._read_radar(row.ra_path), self._read_radar(row.ra_mti_path)))
                for row in records
            ]
        )
        self._radar_cache[sequence] = cached
        while len(self._radar_cache) > self.sequence_cache_size:
            self._radar_cache.popitem(last=False)
        return cached

    def _rv_bearing_for_sequence(
        self,
        sequence: str,
    ) -> dict[str, np.ndarray]:
        if self.rv_bearing_cache_root is None:
            raise RuntimeError("RV-BearingNet cache is not configured")
        cached = self._rv_bearing_cache.get(sequence)
        if cached is not None:
            self._rv_bearing_cache.move_to_end(sequence)
            return cached
        cached = load_rv_bearing_cache(
            self.rv_bearing_cache_root / f"{sequence}.npz"
        )
        self._rv_bearing_cache[sequence] = cached
        while len(self._rv_bearing_cache) > self.sequence_cache_size:
            self._rv_bearing_cache.popitem(last=False)
        return cached

    def _secondary_rv_bearing_for_sequence(
        self,
        sequence: str,
    ) -> dict[str, np.ndarray]:
        if self.secondary_rv_bearing_cache_root is None:
            raise RuntimeError("secondary RV-BearingNet cache is not configured")
        cached = self._secondary_rv_bearing_cache.get(sequence)
        if cached is not None:
            self._secondary_rv_bearing_cache.move_to_end(sequence)
            return cached
        cached = load_rv_bearing_cache(
            self.secondary_rv_bearing_cache_root / f"{sequence}.npz"
        )
        self._secondary_rv_bearing_cache[sequence] = cached
        while len(self._secondary_rv_bearing_cache) > self.sequence_cache_size:
            self._secondary_rv_bearing_cache.popitem(last=False)
        return cached

    @staticmethod
    def _append_bearing_sample(
        sample: dict[str, object],
        *,
        bearing_cache: dict[str, np.ndarray],
        sequence: str,
        frame_id: int,
        prefix: str,
    ) -> None:
        frame_index = bearing_cache["_frame_index"]
        bearing_index = (
            int(frame_index[frame_id])
            if frame_id < frame_index.size
            else -1
        )
        if bearing_index < 0:
            raise KeyError(
                f"{prefix} cache does not cover requested frame "
                f"{sequence}:{frame_id}"
            )
        sample.update(
            {
                f"{prefix}_theta": torch.tensor(
                    bearing_cache["theta"][bearing_index],
                    dtype=torch.float32,
                ),
                f"{prefix}_sigma_theta": torch.tensor(
                    bearing_cache["sigma_theta"][bearing_index],
                    dtype=torch.float32,
                ),
                f"{prefix}_reliability": torch.tensor(
                    bearing_cache["reliability"][bearing_index],
                    dtype=torch.float32,
                ),
                f"{prefix}_temporal_feature": torch.from_numpy(
                    bearing_cache["temporal_feature"][bearing_index]
                ).float(),
                f"{prefix}_valid": torch.tensor(
                    bool(bearing_cache["valid"][bearing_index]),
                    dtype=torch.bool,
                ),
            }
        )

    def _health_for_sequence(self, sequence: str) -> dict[str, np.ndarray]:
        if self.radar_health_cache_root is None:
            raise RuntimeError("Radar Health cache is not configured")
        cached = self._health_cache.get(sequence)
        if cached is not None:
            self._health_cache.move_to_end(sequence)
            return cached
        path = self.radar_health_cache_root / f"{sequence}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing Radar Health cache: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if not {"frame_ids", "health"}.issubset(payload.files):
                raise ValueError(f"invalid Radar Health cache: {path}")
            frame_ids = payload["frame_ids"].astype(np.int64, copy=False)
            health = payload["health"].astype(np.float32, copy=False)
        if (
            frame_ids.ndim != 1
            or frame_ids.size == 0
            or health.shape != (frame_ids.size, 8)
            or np.any(frame_ids < 0)
            or np.unique(frame_ids).size != frame_ids.size
            or not np.isfinite(health).all()
        ):
            raise ValueError(f"invalid Radar Health cache: {path}")
        frame_index = np.full(int(frame_ids.max()) + 1, -1, dtype=np.int64)
        frame_index[frame_ids] = np.arange(frame_ids.size, dtype=np.int64)
        cached = {"health": health, "_frame_index": frame_index}
        self._health_cache[sequence] = cached
        while len(self._health_cache) > self.sequence_cache_size:
            self._health_cache.popitem(last=False)
        return cached

    def _read_rgb(self, path: Path) -> Tensor:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize(
                (self.rgb_image_size, self.rgb_image_size),
                resample=Image.Resampling.BILINEAR,
            )
            values = np.asarray(rgb, dtype=np.uint8).copy()
        return torch.from_numpy(values).permute(2, 0, 1).float().div_(255.0)

    @staticmethod
    def _history_positions(end: int, length: int) -> list[int]:
        return [max(0, end - length + 1 + offset) for offset in range(length)]

    def __getitem__(self, item: int) -> dict[str, object]:
        sequence, end_position = self.samples[item]
        records = self.sequences[sequence]
        radar_positions = self._history_positions(end_position, self.radar_clip_length)
        visual_positions = self._history_positions(end_position, self.visual_clip_length)
        radar_history = [records[position] for position in radar_positions]
        target = records[end_position]
        cache = self._visual_for_sequence(sequence)
        frame_index = cache["_frame_index"]
        visual_indices = [int(frame_index[records[position].frame_id]) for position in visual_positions]
        if any(index < 0 for index in visual_indices):
            raise KeyError(f"visual cache does not cover requested history in {sequence}")
        condition = self.condition_timeline.lookup(sequence, target.frame_id)
        sample: dict[str, object] = {
            "visual_boxes": torch.from_numpy(cache["boxes"][visual_indices]).float(),
            "visual_raw_scores": torch.from_numpy(cache["raw_scores"][visual_indices]).float(),
            "visual_raw_tokens": torch.from_numpy(cache["raw_tokens"][visual_indices]).float(),
            "visual_temporal_tokens": torch.from_numpy(cache["temporal_tokens"][visual_indices]).float(),
            "visual_temporal_support": torch.from_numpy(cache["temporal_support"][visual_indices]).float(),
            "visual_valid_mask": torch.from_numpy(cache["valid_mask"][visual_indices]).bool(),
            "target_xy_m": torch.tensor(target.target_xy_m, dtype=torch.float32),
            "sample_weight": torch.tensor(0.75 if target.source_is_radar else 1.0),
            "history_frame_ids": torch.tensor(
                [row.frame_id for row in radar_history], dtype=torch.int64
            ),
            "frame_id": torch.tensor(target.frame_id, dtype=torch.int64),
            "sequence": sequence,
            "condition": {
                "global_route": condition.global_route,
                "visual_condition": condition.visual_condition,
                "visual_reliability": condition.visual_reliability,
                "weather_surface": condition.weather_surface,
            },
        }
        if self.return_radar:
            sample["radar"] = self._radar_for_sequence(sequence)[radar_positions]
        if self.radar_health_cache_root is not None:
            health_cache = self._health_for_sequence(sequence)
            health_frame_index = health_cache["_frame_index"]
            history_frame_ids = [row.frame_id for row in radar_history]
            health_indices = [
                (
                    int(health_frame_index[frame_id])
                    if frame_id < health_frame_index.size
                    else -1
                )
                for frame_id in history_frame_ids
            ]
            if any(index < 0 for index in health_indices):
                raise KeyError(
                    f"Radar Health cache does not cover requested history in {sequence}"
                )
            sample["radar_health_history"] = torch.from_numpy(
                health_cache["health"][health_indices]
            ).float()
        if self.rv_bearing_cache_root is not None:
            bearing_cache = self._rv_bearing_for_sequence(sequence)
            self._append_bearing_sample(
                sample,
                bearing_cache=bearing_cache,
                sequence=sequence,
                frame_id=target.frame_id,
                prefix="rv_bearing",
            )
        if self.secondary_rv_bearing_cache_root is not None:
            secondary_cache = self._secondary_rv_bearing_for_sequence(sequence)
            self._append_bearing_sample(
                sample,
                bearing_cache=secondary_cache,
                sequence=sequence,
                frame_id=target.frame_id,
                prefix="secondary_rv_bearing",
            )
        if self.return_rgb:
            sample["rgb_clip"] = torch.stack(
                [self._read_rgb(records[position].rgb_path) for position in visual_positions]
            )
        return sample


class SequenceGroupedSampler(Sampler[int]):
    """Shuffle complete sequences while keeping frames contiguous within each sequence."""

    def __init__(
        self,
        dataset: RVFusionCachedDataset,
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
            sequence: tuple(indices) for sequence, indices in sorted(grouped.items())
        }

    def __iter__(self):
        sequences = list(self.grouped_indices)
        if self.shuffle_sequences:
            random.Random(self.seed + self.epoch).shuffle(sequences)
        for sequence in sequences:
            yield from self.grouped_indices[sequence]

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.grouped_indices.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)



