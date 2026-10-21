from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


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


@dataclass(frozen=True)
class RadarFrameRecord:
    sequence: str
    frame_id: int
    ra_path: Path
    ra_mti_path: Path
    rd_path: Path
    rd_mti_path: Path
    ad_path: Path
    target_xy_m: tuple[float, float]
    source_is_radar: bool
    input_clean: bool
    label_valid: bool
    target_eligible: bool


class RadarWindowDataset(Dataset[dict[str, Tensor | str]]):
    """Causal RA/RA-MTI clips with optional aligned RD/RD-MIT/AD side views."""

    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        split: str,
        clip_length: int = 16,
        eligibility: str = "later",
        db_center: float = 80.0,
        db_scale: float = 30.0,
        cache_in_memory: bool = False,
        allowed_sequences: Collection[str] | None = None,
        include_aux_views: bool = False,
        aux_view_history_frames: int | None = None,
        cache_primary_in_memory: bool = False,
        aux_views: Collection[str] | None = None,
    ) -> None:
        if clip_length < 1:
            raise ValueError("clip_length must be positive")
        if eligibility not in {"first", "later"}:
            raise ValueError("eligibility must be 'first' or 'later'")
        self.data_root = Path(data_root)
        self.clip_length = clip_length
        self.db_center = db_center
        self.db_scale = db_scale
        self.cache_in_memory = cache_in_memory
        self.include_aux_views = include_aux_views
        requested_aux = () if not include_aux_views else (
            ("rd", "rd_mti", "ad") if aux_views is None else tuple(sorted({str(view) for view in aux_views}))
        )
        invalid_aux = set(requested_aux) - {"rd", "rd_mti", "ad"}
        if invalid_aux:
            raise ValueError(f"unsupported aux_views: {sorted(invalid_aux)}")
        if requested_aux and not include_aux_views:
            raise ValueError("include_aux_views=True is required when aux_views are requested")
        self.aux_views = requested_aux if include_aux_views else ()
        if aux_view_history_frames is not None and aux_view_history_frames < 1:
            raise ValueError("aux_view_history_frames must be positive when specified")
        self.aux_view_history_frames = aux_view_history_frames
        self.cache_primary_in_memory = cache_primary_in_memory
        self._map_cache: dict[Path, Tensor] = {}
        allowed = None if allowed_sequences is None else {str(value) for value in allowed_sequences}
        sequence_rows: dict[str, list[RadarFrameRecord]] = defaultdict(list)
        with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"].strip().lower() != split.strip().lower():
                    continue
                if allowed is not None and row["sequence"] not in allowed:
                    continue
                input_clean = (
                    not _as_bool(row["sequence_excluded"])
                    and _as_bool(row["radar_input_range_eligible"])
                    and _as_bool(row["radar_quality_valid"])
                    and not _as_bool(row["red_band_anomaly"])
                    and not _as_bool(row["blue_dropout_anomaly"])
                )
                required = input_clean and _as_bool(row["final_bev_label_valid"])
                sequence_rows[row["sequence"]].append(
                    RadarFrameRecord(
                        sequence=row["sequence"],
                        frame_id=int(row["frame_id"]),
                        ra_path=self.data_root / row["ra_path"],
                        ra_mti_path=self.data_root / row["ra_mti_path"],
                        rd_path=self._auxiliary_path(row, "rd_path", "RD"),
                        rd_mti_path=self._auxiliary_path(row, "rd_mti_path", "RD-MIT"),
                        ad_path=self._auxiliary_path(row, "ad_path", "AD"),
                        target_xy_m=(float(row["final_bev_x_m"]), float(row["final_bev_y_m"])),
                        source_is_radar=_as_bool(row["final_bev_source_is_radar"]),
                        input_clean=input_clean,
                        label_valid=required,
                        target_eligible=required and training_eligibility_from_row(row),
                    )
                )

        self.sequences: dict[str, tuple[RadarFrameRecord, ...]] = {
            sequence: tuple(sorted(rows, key=lambda item: item.frame_id)) for sequence, rows in sequence_rows.items()
        }
        self.samples: list[tuple[str, int]] = []
        for sequence, rows in sorted(self.sequences.items()):
            for position, row in enumerate(rows):
                history_positions = [max(0, position - self.clip_length + 1 + offset) for offset in range(self.clip_length)]
                if row.target_eligible and all(rows[history_position].input_clean for history_position in history_positions):
                    self.samples.append((sequence, position))
        if not self.samples:
            raise ValueError(f"No eligible rows found for split={split!r}, eligibility={eligibility!r}")
        if self.include_aux_views:
            self._validate_auxiliary_alignment()
        if self.cache_in_memory or self.cache_primary_in_memory:
            for records in self.sequences.values():
                for record in records:
                    self._read_map(record.ra_path, (128, 128), cache_primary=True)
                    self._read_map(record.ra_mti_path, (128, 128), cache_primary=True)
                    if self.cache_in_memory:
                        if "rd" in self.aux_views:
                            self._read_map(record.rd_path, (128, 256))
                        if "rd_mti" in self.aux_views:
                            self._read_map(record.rd_mti_path, (128, 256))
                        if "ad" in self.aux_views:
                            self._read_map(record.ad_path, (128, 256))

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def cached_map_count(self) -> int:
        return len(self._map_cache)

    def _auxiliary_path(self, row: dict[str, str], column: str, folder: str) -> Path:
        explicit = row.get(column, "").strip()
        if explicit:
            return self.data_root / explicit
        ra_path = self.data_root / row["ra_path"]
        return ra_path.parent.parent / folder / ra_path.name

    def _validate_auxiliary_alignment(self) -> None:
        for records in self.sequences.values():
            for record in records:
                candidates = {
                    "rd": (record.rd_path, "RD"),
                    "rd_mti": (record.rd_mti_path, "RD-MIT"),
                    "ad": (record.ad_path, "AD"),
                }
                for view in self.aux_views:
                    path, name = candidates[view]
                    if path.name != record.ra_path.name or not path.is_file():
                        raise ValueError(
                            f"{record.sequence} frame={record.frame_id}: {name} must be an existing same-name RA side view; got {path}"
                        )

    def _read_map(self, path: Path, expected_shape: tuple[int, int], cache_primary: bool = False) -> Tensor:
        cached = self._map_cache.get(path)
        if cached is not None:
            return cached
        values = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        if values.shape != expected_shape:
            raise ValueError(f"Expected {expected_shape} radar map, got {values.shape} at {path}")
        values = np.nan_to_num(values, nan=self.db_center, posinf=self.db_center + self.db_scale, neginf=self.db_center - self.db_scale)
        tensor = torch.from_numpy(np.clip((values - self.db_center) / self.db_scale, -3.0, 3.0))
        if self.cache_in_memory or (cache_primary and self.cache_primary_in_memory):
            self._map_cache[path] = tensor
        return tensor

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        sequence, end_position = self.samples[index]
        rows = self.sequences[sequence]
        history_positions = [max(0, end_position - self.clip_length + 1 + offset) for offset in range(self.clip_length)]
        history = [rows[position] for position in history_positions]
        target = rows[end_position]
        radar = torch.stack(
            [
                torch.stack((self._read_map(row.ra_path, (128, 128)), self._read_map(row.ra_mti_path, (128, 128))), dim=0)
                for row in history
            ],
            dim=0,
        )
        # Radar-assisted labels are intentionally softer/lower-weight in this first stage.
        sample_weight = 0.75 if target.source_is_radar else 1.0
        item: dict[str, Tensor | str] = {
            "radar": radar,
            "target_xy_m": torch.tensor(target.target_xy_m, dtype=torch.float32),
            "history_target_xy_m": torch.tensor([row.target_xy_m for row in history], dtype=torch.float32),
            "history_target_valid": torch.tensor([row.label_valid for row in history], dtype=torch.bool),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "history_frame_ids": torch.tensor([row.frame_id for row in history], dtype=torch.int64),
            "frame_id": torch.tensor(target.frame_id, dtype=torch.int64),
            "sequence": sequence,
        }
        if self.include_aux_views:
            aux_history = history if self.aux_view_history_frames is None else history[-self.aux_view_history_frames :]
            if "rd" in self.aux_views:
                item["rd"] = torch.stack([self._read_map(row.rd_path, (128, 256)) for row in aux_history], dim=0)
            if "rd_mti" in self.aux_views:
                item["rd_mti"] = torch.stack([self._read_map(row.rd_mti_path, (128, 256)) for row in aux_history], dim=0)
            if "ad" in self.aux_views:
                item["ad"] = torch.stack([self._read_map(row.ad_path, (128, 256)) for row in aux_history], dim=0)
        return item



