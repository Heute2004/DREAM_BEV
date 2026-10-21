from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .data import RGBBearingClipDataset, coordinate_supervision_weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cached_proposal_source(
    *,
    cache_root: str | Path,
    checkpoint: str | Path,
    image_height: int,
    image_width: int,
) -> dict[str, object]:
    cache_root = Path(cache_root)
    checkpoint = Path(checkpoint)
    metadata_paths = sorted(
        cache_root.glob("cache_metadata_shard_*.json")
    )
    if not metadata_paths:
        raise FileNotFoundError(
            f"proposal cache metadata not found: {cache_root}"
        )
    metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in metadata_paths
    ]
    if not all(isinstance(item, dict) for item in metadata):
        raise TypeError("proposal cache metadata must be mappings")
    checkpoint_sha = _sha256(checkpoint)
    recorded_shas = {
        str(item.get("checkpoint_sha256", ""))
        for item in metadata
    }
    if recorded_shas != {checkpoint_sha}:
        raise ValueError(
            "proposal cache checkpoint does not match "
            f"{checkpoint}: recorded={sorted(recorded_shas)}"
        )
    if image_height != image_width:
        raise ValueError("frozen proposal cache requires square images")
    recorded_sizes = {
        int(item.get("image_size", -1)) for item in metadata
    }
    if recorded_sizes != {int(image_height)}:
        raise ValueError(
            "proposal cache image size does not match training contract"
        )
    recorded_top_k = {
        int(item.get("top_k", -1)) for item in metadata
    }
    if len(recorded_top_k) != 1:
        raise ValueError("proposal cache shards disagree on top_k")
    quality_used = {
        bool(item.get("proposal_quality_used", False))
        for item in metadata
    }
    if quality_used != {False}:
        raise ValueError(
            "cached proposal training requires quality-free evidence"
        )
    manifest_shas = {
        str(item.get("manifest_sha256", ""))
        for item in metadata
        if item.get("manifest_sha256") is not None
    }
    if len(manifest_shas) > 1:
        raise ValueError("proposal cache shards disagree on manifest")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "image_size": image_height,
        "top_k": next(iter(recorded_top_k)),
        "proposal_quality_used": False,
        "manifest_sha256": (
            next(iter(manifest_shas)) if manifest_shas else None
        ),
        "metadata_files": [str(path) for path in metadata_paths],
    }


class CachedProposalBearingDataset(RGBBearingClipDataset):
    """Bearing supervision over frozen, frame-aligned visual proposals."""

    def __init__(
        self,
        *,
        visual_evidence_cache_root: str | Path,
        proposal_cache_size: int = 2,
        **kwargs,
    ) -> None:
        if proposal_cache_size < 1:
            raise ValueError("proposal_cache_size must be positive")
        kwargs["preload_rgb"] = False
        kwargs["rgb_cache_size"] = 0
        kwargs["train_augmentation"] = "none"
        super().__init__(**kwargs)
        self.visual_evidence_cache_root = Path(
            visual_evidence_cache_root
        )
        self.proposal_cache_size = int(proposal_cache_size)
        self._proposal_cache: OrderedDict[
            str,
            tuple[dict[str, np.ndarray], dict[int, int]],
        ] = OrderedDict()

    def _load_sequence_cache(
        self,
        sequence: str,
    ) -> tuple[dict[str, np.ndarray], dict[int, int]]:
        cached = self._proposal_cache.get(sequence)
        if cached is not None:
            self._proposal_cache.move_to_end(sequence)
            return cached
        path = self.visual_evidence_cache_root / f"{sequence}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing frozen proposal cache: {path}"
            )
        required = {
            "frame_ids",
            "boxes",
            "raw_scores",
            "raw_tokens",
            "valid_mask",
        }
        with np.load(path, allow_pickle=False) as payload:
            missing = required - set(payload.files)
            if missing:
                raise ValueError(
                    f"proposal cache lacks {sorted(missing)}: {path}"
                )
            values = {
                key: np.array(payload[key], copy=True)
                for key in required
            }
        frame_ids = values["frame_ids"].astype(np.int64, copy=False)
        if frame_ids.ndim != 1 or len(set(frame_ids.tolist())) != len(
            frame_ids
        ):
            raise ValueError(
                f"proposal frame IDs must be unique and one-dimensional: {path}"
            )
        if any(value.shape[0] != frame_ids.shape[0] for value in values.values()):
            raise ValueError(f"proposal arrays are not frame aligned: {path}")
        frame_index = {
            int(frame_id): index
            for index, frame_id in enumerate(frame_ids.tolist())
        }
        cached = values, frame_index
        self._proposal_cache[sequence] = cached
        while len(self._proposal_cache) > self.proposal_cache_size:
            self._proposal_cache.popitem(last=False)
        return cached

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
        cache, frame_index = self._load_sequence_cache(sequence)
        try:
            indices = [
                frame_index[record.frame_id] for record in history
            ]
        except KeyError as error:
            raise KeyError(
                f"proposal cache does not cover {sequence} frame "
                f"{int(error.args[0])}"
            ) from error
        coordinate_weight = coordinate_supervision_weight(condition)
        return {
            "proposal_boxes": torch.from_numpy(
                cache["boxes"][indices]
            ).float(),
            "proposal_raw_scores": torch.from_numpy(
                cache["raw_scores"][indices]
            ).float(),
            "proposal_raw_tokens": torch.from_numpy(
                cache["raw_tokens"][indices]
            ).float(),
            "proposal_valid_mask": torch.from_numpy(
                cache["valid_mask"][indices]
            ).bool(),
            "proposal_image_height": torch.tensor(
                self.image_height,
                dtype=torch.int64,
            ),
            "proposal_image_width": torch.tensor(
                self.image_width,
                dtype=torch.int64,
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



