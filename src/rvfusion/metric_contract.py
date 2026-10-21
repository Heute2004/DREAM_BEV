from __future__ import annotations

import torch
from torch import Tensor

from .contracts import RadarEvidence


def frozen_radar_confidence(logits: Tensor) -> Tensor:
    """Return the immutable radar-expert single-person detection confidence."""

    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("radar logits must be [batch,1,x_bins,y_bins]")
    return torch.sigmoid(logits.float().flatten(1).amax(dim=1))


def frozen_radar_position(logits: Tensor) -> Tensor:
    """Decode the immutable DREAM-BEV peak using the canonical radar-stage geometry."""

    if logits.ndim != 4 or tuple(logits.shape[1:]) != (1, 200, 234):
        raise ValueError("radar logits must be [batch,1,200,234]")
    flat_index = logits.float().flatten(1).argmax(dim=1)
    x_index = torch.div(flat_index, 234, rounding_mode="floor")
    y_index = flat_index.remainder(234)
    x_m = -10.0 + x_index.float() * 0.1
    y_m = y_index.float() * 0.1
    return torch.stack((x_m, y_m), dim=1)


def resolved_radar_position(evidence: RadarEvidence) -> Tensor:
    """Decode radar position while honoring an explicitly valid causal override."""

    evidence.validate()
    position = frozen_radar_position(evidence.logits)
    if evidence.position_override_xy_m is None:
        return position
    assert evidence.position_override_mask is not None
    return torch.where(
        evidence.position_override_mask[:, None],
        evidence.position_override_xy_m.to(
            device=position.device,
            dtype=position.dtype,
        ),
        position,
    )



