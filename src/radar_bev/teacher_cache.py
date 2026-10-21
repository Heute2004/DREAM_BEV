from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


ALLOWED_TEACHER_SPLITS = ("Train", "Development Val")


def normalize_teacher_splits(splits: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(split.strip() for split in splits)
    forbidden = sorted(set(normalized) - set(ALLOWED_TEACHER_SPLITS))
    if forbidden:
        raise ValueError(
            "frozen teacher cache may only contain Train or Development Val; "
            f"forbidden split(s): {', '.join(forbidden)}"
        )
    if not normalized:
        raise ValueError("at least one non-test teacher split is required")
    return normalized


def teacher_cache_path(root: str | Path, split: str, sequence: str, frame_id: int) -> Path:
    safe_split = split.replace(" ", "_")
    if not sequence.startswith("seq_"):
        raise ValueError("teacher cache sequence must use canonical seq_* naming")
    return Path(root) / safe_split / sequence / f"frame_{frame_id:04d}.pt"


class TeacherCacheDataset(Dataset[dict[str, Tensor | str]]):
    """Attach a frozen teacher target that is keyed to one canonical frame."""

    def __init__(self, base: Dataset[dict[str, Tensor | str]], cache_root: str | Path, split: str) -> None:
        self.base = base
        self.cache_root = Path(cache_root)
        self.split = split

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        item = dict(self.base[index])
        sequence = item.get("sequence")
        frame_id = item.get("frame_id")
        if not isinstance(sequence, str) or not isinstance(frame_id, Tensor) or frame_id.numel() != 1:
            raise TypeError("teacher cache requires scalar frame_id and canonical sequence")
        path = teacher_cache_path(self.cache_root, self.split, sequence, int(frame_id.item()))
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen teacher target: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"teacher cache payload must be a mapping: {path}")
        logits = payload.get("logits")
        peak = payload.get("peak_xy_m")
        if not isinstance(logits, Tensor) or logits.shape != (1, 200, 234):
            raise ValueError(f"teacher logits must be [1,200,234]: {path}")
        if not isinstance(peak, Tensor) or peak.shape != (2,):
            raise ValueError(f"teacher peak must be [2]: {path}")
        item["teacher_logits"] = logits.float()
        item["teacher_peak_xy_m"] = peak.float()
        return item
