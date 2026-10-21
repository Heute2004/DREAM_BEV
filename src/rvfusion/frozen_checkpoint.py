"""Small, explicit checkpoint loader for evaluation-only public-model replays."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


def load_frozen_model_state(model: nn.Module, checkpoint_path: str | Path) -> dict[str, object]:
    """Restore a frozen training checkpoint without mutating its metadata."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model_state" not in payload:
        raise ValueError(f"checkpoint lacks model_state: {path}")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping):
        raise ValueError(f"checkpoint model_state is invalid: {path}")
    model.load_state_dict(model_state, strict=True)
    return {str(name): value for name, value in payload.items() if name != "model_state"}
