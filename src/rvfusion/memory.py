from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contracts import VisualEvidence


@dataclass(frozen=True)
class KVMemoryOutput:
    keys: Tensor
    values: Tensor
    valid_mask: Tensor
    has_evidence: Tensor
    source_time_index: Tensor


class VisualKVMemory(nn.Module):
    """Compact causal memory over frozen proposal evidence tokens."""

    def __init__(
        self,
        token_dim: int = 128,
        memory_dim: int = 64,
        max_entries: int = 16,
        score_threshold: float = 0.05,
        image_size: float = 640.0,
    ) -> None:
        super().__init__()
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self.score_threshold = score_threshold
        self.image_size = image_size
        input_dim = token_dim * 2 + 8
        self.key_projection = nn.Sequential(nn.Linear(input_dim, memory_dim), nn.SiLU(), nn.LayerNorm(memory_dim))
        self.value_projection = nn.Sequential(nn.Linear(input_dim, memory_dim), nn.SiLU(), nn.LayerNorm(memory_dim))

    def forward(self, visual: VisualEvidence, trust_cap: Tensor) -> KVMemoryOutput:
        visual.validate()
        batch, time_steps, proposals = visual.raw_scores.shape
        if tuple(trust_cap.shape) != (batch, 1):
            raise ValueError("trust_cap must be [batch,1]")
        boxes = visual.boxes / self.image_size
        age = torch.linspace(1.0, 0.0, time_steps, device=boxes.device, dtype=boxes.dtype)
        age = age.view(1, time_steps, 1, 1).expand(batch, time_steps, proposals, 1)
        features = torch.cat(
            (
                visual.raw_tokens,
                visual.temporal_tokens,
                boxes,
                visual.raw_scores.unsqueeze(-1),
                visual.temporal_support.unsqueeze(-1),
                age,
                trust_cap[:, None, None, :].expand(batch, time_steps, proposals, 1),
            ),
            dim=-1,
        )
        valid = (
            visual.valid_mask
            & (visual.raw_scores >= self.score_threshold)
            & (trust_cap[:, 0].view(batch, 1, 1) > 0.0)
        )
        flat_features = features.reshape(batch, time_steps * proposals, -1)
        flat_valid = valid.reshape(batch, time_steps * proposals)
        flat_time = (
            torch.arange(time_steps, device=boxes.device)
            .view(1, time_steps, 1)
            .expand(batch, time_steps, proposals)
            .reshape(batch, -1)
        )

        selected_features = features.new_zeros(batch, self.max_entries, features.shape[-1])
        selected_valid = torch.zeros(batch, self.max_entries, device=boxes.device, dtype=torch.bool)
        selected_time = torch.full(
            (batch, self.max_entries), -1, device=boxes.device, dtype=torch.long
        )
        for batch_index in range(batch):
            indices = flat_valid[batch_index].nonzero(as_tuple=False).flatten()
            indices = indices[-self.max_entries :]
            count = int(indices.numel())
            if count:
                destination = slice(self.max_entries - count, self.max_entries)
                selected_features[batch_index, destination] = flat_features[batch_index, indices]
                selected_valid[batch_index, destination] = True
                selected_time[batch_index, destination] = flat_time[batch_index, indices]

        mask = selected_valid.unsqueeze(-1)
        keys = self.key_projection(selected_features) * mask
        values = self.value_projection(selected_features) * mask
        return KVMemoryOutput(
            keys=keys,
            values=values,
            valid_mask=selected_valid,
            has_evidence=selected_valid.any(dim=1),
            source_time_index=selected_time,
        )



