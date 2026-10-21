"""Causal radar--vision disagreement guard for frozen CRUW fusion outputs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CausalLateralConsistencyGuard:
    """Keep a signed innovation EMA separately for each ordered sequence."""

    ema_alpha: float
    ema_threshold: float
    _ema_by_sequence: dict[int, float] = field(default_factory=dict, init=False)
    _last_frame_by_sequence: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < float(self.ema_alpha) <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if float(self.ema_threshold) <= 0.0:
            raise ValueError("ema_threshold must be positive")

    def step(
        self,
        *,
        sequence_id: int,
        frame_id: int,
        signed_innovation: float,
        eligible: bool,
    ) -> tuple[bool, float]:
        """Return the causal visual admission and current sequence EMA.

        A no-op posterior is not visual evidence and therefore neither updates
        state nor bypasses the exact radar identity fallback.
        """

        previous = self._last_frame_by_sequence.get(sequence_id)
        if previous is not None and frame_id <= previous:
            raise ValueError("frame ids must increase strictly within a sequence")
        self._last_frame_by_sequence[sequence_id] = frame_id
        current = self._ema_by_sequence.get(sequence_id, 0.0)
        if not eligible:
            return False, current
        current = (1.0 - self.ema_alpha) * current + self.ema_alpha * signed_innovation
        self._ema_by_sequence[sequence_id] = current
        return abs(current) <= self.ema_threshold, current
