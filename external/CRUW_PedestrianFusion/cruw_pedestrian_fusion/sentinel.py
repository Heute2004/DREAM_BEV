"""Presence-only Empty-Scene Sentinel for the frozen DREAM-BEV anchor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class EmptySceneSentinel(nn.Module):
    """Predict whether current radar evidence contains any in-FOV pedestrian.

    The sentinel never creates or relocates a peak. Its only allowed operation is
    a spatially uniform confidence suppression when the causal state is empty.
    """

    diagnostic_dim = 6

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(128 + self.diagnostic_dim, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(96, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
        )

    @staticmethod
    def features(
        shared_feature: torch.Tensor,
        base_logits: torch.Tensor,
        radar: torch.Tensor,
    ) -> torch.Tensor:
        if shared_feature.ndim != 4 or shared_feature.shape[1:] != (128, 50, 59):
            raise ValueError("shared_feature must be [batch,128,50,59]")
        if base_logits.ndim != 4 or base_logits.shape[1:] != (1, 200, 234):
            raise ValueError("base_logits must be [batch,1,200,234]")
        if radar.ndim != 5 or radar.shape[2:] != (2, 128, 128):
            raise ValueError("radar must be [batch,time,2,128,128]")
        pooled = shared_feature.mean(dim=(2, 3))
        confidence = torch.sigmoid(base_logits).flatten(1)
        latest = radar[:, -1]
        diagnostics = torch.stack(
            (
                confidence.amax(dim=1),
                confidence.mean(dim=1),
                latest[:, 0].mean(dim=(1, 2)),
                latest[:, 0].std(dim=(1, 2), unbiased=False),
                latest[:, 1].abs().mean(dim=(1, 2)),
                (latest[:, 0] - latest[:, 1]).abs().mean(dim=(1, 2)),
            ),
            dim=1,
        )
        return torch.cat((pooled, diagnostics), dim=1)

    def forward(
        self,
        shared_feature: torch.Tensor,
        base_logits: torch.Tensor,
        radar: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.features(shared_feature, base_logits, radar))

    @staticmethod
    def suppress(
        base_logits: torch.Tensor,
        presence_logit: torch.Tensor,
        *,
        threshold: float,
        suppression_logit: float = 8.0,
    ) -> torch.Tensor:
        if presence_logit.shape != (base_logits.shape[0], 1):
            raise ValueError("presence_logit must be [batch,1]")
        empty = torch.sigmoid(presence_logit) < threshold
        return base_logits - empty[:, :, None, None].to(base_logits.dtype) * suppression_logit


@dataclass
class SentinelHysteresis:
    empty_enter_frames: int = 4
    present_exit_frames: int = 2
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.empty_enter_frames < 1 or self.present_exit_frames < 1:
            raise ValueError("hysteresis frame counts must be positive")
        self.is_empty = False
        self._empty_count = 0
        self._present_count = 0

    def reset(self) -> None:
        self.is_empty = False
        self._empty_count = 0
        self._present_count = 0

    def update(self, presence_probability: float) -> bool:
        if presence_probability < self.threshold:
            self._empty_count += 1
            self._present_count = 0
            if self._empty_count >= self.empty_enter_frames:
                self.is_empty = True
        else:
            self._present_count += 1
            self._empty_count = 0
            if self._present_count >= self.present_exit_frames:
                self.is_empty = False
        return self.is_empty
