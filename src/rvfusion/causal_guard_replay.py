"""Causal composition of a frozen visual candidate with radar identity fallback."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

import torch

from .cross_modal_consistency import CausalSignedInnovationGuard


@dataclass(frozen=True)
class CausalGuardReplayFrame:
    """One frozen radar/candidate record in causal sequence order."""

    sequence: str
    frame_id: int
    policy_admitted: bool
    radar_x_m: float
    radar_y_m: float
    candidate_x_m: float
    target_x_m: float
    target_y_m: float
    score: float
    signed_innovation: float


@dataclass(frozen=True)
class GuardedReplayFrame(CausalGuardReplayFrame):
    """Frozen record after the fast evidence-consistency decision."""

    visual_accepted: bool
    fused_x_m: float
    fused_y_m: float
    innovation_ema: float


AdmissionMode = Literal["learned", "always-admit", "always-identity"]


def override_policy_admission(
    frames: Iterable[CausalGuardReplayFrame],
    *,
    mode: AdmissionMode,
    warmup_frames: int = 32,
) -> list[CausalGuardReplayFrame]:
    """Return a deterministic Development-only policy-admission control.

    ``always-admit`` preserves the causal warm-up identity interval in every
    sequence and then makes visual evidence eligible on every frame.  It does
    not alter the learned expert route, candidate coordinates, or guard.
    ``always-identity`` leaves every frame at the exact radar identity.
    """

    if mode not in {"learned", "always-admit", "always-identity"}:
        raise ValueError(f"unsupported admission mode: {mode}")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")

    source = list(frames)
    if mode == "learned":
        return source
    if mode == "always-identity":
        return [replace(frame, policy_admitted=False) for frame in source]
    return [
        replace(frame, policy_admitted=frame.frame_id >= warmup_frames)
        for frame in source
    ]


def replay_guarded_frames(
    frames: Iterable[CausalGuardReplayFrame],
    *,
    ema_alpha: float,
    ema_threshold: float,
    instantaneous_threshold: float | None = None,
    guard_enabled: bool = True,
) -> list[GuardedReplayFrame]:
    """Replay records in sequence order with only past signed innovations.

    The returned point keeps the radar range in every case. A rejected visual
    candidate is composed as an exact radar identity, rather than a weighted
    blend or a reduced correction.
    """

    guard = (
        CausalSignedInnovationGuard(
            ema_alpha=ema_alpha,
            ema_threshold=ema_threshold,
            instantaneous_threshold=instantaneous_threshold,
        )
        if guard_enabled
        else None
    )
    result: list[GuardedReplayFrame] = []
    for frame in frames:
        if guard is None:
            passed = True
            innovation_ema = 0.0
        else:
            passed = guard.step(
                sequences=(frame.sequence,),
                frame_ids=(frame.frame_id,),
                signed_innovation=torch.tensor(
                    [[frame.signed_innovation]],
                    dtype=torch.float32,
                ),
                eligible=torch.tensor([frame.policy_admitted], dtype=torch.bool),
            ).item()
            innovation_ema = guard.current_ema(frame.sequence)
        visual_accepted = bool(frame.policy_admitted and passed)
        result.append(
            GuardedReplayFrame(
                **frame.__dict__,
                visual_accepted=visual_accepted,
                fused_x_m=(
                    frame.candidate_x_m if visual_accepted else frame.radar_x_m
                ),
                fused_y_m=frame.radar_y_m,
                innovation_ema=innovation_ema,
            )
        )
    return result


def bootstrap_rows_from_guarded_replay(
    frames: Iterable[GuardedReplayFrame],
) -> list[dict[str, float | bool | str]]:
    """Expose final compositor outputs to the shared sequence bootstrap."""

    return [
        {
            "sequence": frame.sequence,
            "target_x_m": frame.target_x_m,
            "target_y_m": frame.target_y_m,
            "radar_x_m": frame.radar_x_m,
            "radar_y_m": frame.radar_y_m,
            "candidate_x_m": frame.fused_x_m,
            "candidate_y_m": frame.fused_y_m,
            "radar_score": frame.score,
            "candidate_score": frame.score,
            "accepted": True,
        }
        for frame in frames
    ]
