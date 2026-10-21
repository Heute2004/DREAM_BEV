from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .contracts import TemporalBearingEvidence


def load_rv_bearing_cache(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing RV-BearingNet cache: {path}")
    with np.load(path, allow_pickle=False) as payload:
        cache = {name: payload[name] for name in payload.files}
    required = {
        "frame_ids",
        "theta",
        "sigma_theta",
        "reliability",
        "temporal_feature",
        "valid",
    }
    if not required.issubset(cache):
        raise ValueError(
            f"RV-BearingNet cache lacks {sorted(required - set(cache))}: "
            f"{path}"
        )
    frame_ids = cache["frame_ids"].astype(np.int64, copy=False)
    count = frame_ids.size
    if (
        frame_ids.ndim != 1
        or count == 0
        or np.any(frame_ids < 0)
        or np.unique(frame_ids).size != count
        or np.any(frame_ids[1:] <= frame_ids[:-1])
    ):
        raise ValueError(f"invalid sorted unique frame IDs: {path}")
    expected_vectors = {
        "theta": (count,),
        "sigma_theta": (count,),
        "reliability": (count,),
        "valid": (count,),
    }
    for name, shape in expected_vectors.items():
        if cache[name].shape != shape:
            raise ValueError(f"invalid {name} shape in {path}")
    if (
        cache["temporal_feature"].ndim != 2
        or cache["temporal_feature"].shape[0] != count
    ):
        raise ValueError(f"invalid temporal_feature shape in {path}")
    for name in ("theta", "sigma_theta", "reliability", "temporal_feature"):
        cache[name] = cache[name].astype(np.float32, copy=False)
        if not np.isfinite(cache[name]).all():
            raise ValueError(f"non-finite {name} in {path}")
    if np.any(cache["sigma_theta"] <= 0.0):
        raise ValueError(f"non-positive sigma_theta in {path}")
    if np.any(
        (cache["reliability"] < 0.0)
        | (cache["reliability"] > 1.0)
    ):
        raise ValueError(f"reliability outside [0,1] in {path}")
    cache["valid"] = cache["valid"].astype(np.bool_, copy=False)
    frame_index = np.full(int(frame_ids.max()) + 1, -1, dtype=np.int64)
    frame_index[frame_ids] = np.arange(count, dtype=np.int64)
    cache["frame_ids"] = frame_ids
    cache["_frame_index"] = frame_index
    return cache


def _batch_tensor(
    batch: dict[str, object],
    name: str,
    device: torch.device,
) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise KeyError(f"batch lacks tensor field {name}")
    return value.to(device, non_blocking=True)


def bearing_from_collated_cache(
    batch: dict[str, object],
    device: torch.device,
    *,
    prefix: str = "rv_bearing",
) -> TemporalBearingEvidence:
    theta = _batch_tensor(batch, f"{prefix}_theta", device).float()
    sigma = _batch_tensor(
        batch,
        f"{prefix}_sigma_theta",
        device,
    ).float()
    reliability = _batch_tensor(
        batch,
        f"{prefix}_reliability",
        device,
    ).float()
    temporal_feature = _batch_tensor(
        batch,
        f"{prefix}_temporal_feature",
        device,
    ).float()
    valid = _batch_tensor(
        batch,
        f"{prefix}_valid",
        device,
    ).bool()
    if theta.ndim == 1:
        theta = theta[:, None]
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    if reliability.ndim == 1:
        reliability = reliability[:, None]
    result = TemporalBearingEvidence(
        theta=theta,
        sigma_theta=sigma,
        temporal_feature=temporal_feature,
        reliability=reliability,
        valid=valid,
    )
    result.validate()
    return result



