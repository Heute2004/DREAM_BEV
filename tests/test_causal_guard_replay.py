from __future__ import annotations

from dataclasses import replace

from rvfusion.causal_guard_replay import (
    CausalGuardReplayFrame,
    bootstrap_rows_from_guarded_replay,
    override_policy_admission,
    replay_guarded_frames,
)


def _frame(
    *,
    frame_id: int,
    visual_x_m: float,
    innovation: float,
) -> CausalGuardReplayFrame:
    return CausalGuardReplayFrame(
        sequence="seq_a",
        frame_id=frame_id,
        policy_admitted=True,
        radar_x_m=0.0,
        radar_y_m=4.0,
        candidate_x_m=visual_x_m,
        target_x_m=0.0,
        target_y_m=4.0,
        score=0.9,
        signed_innovation=innovation,
    )


def test_replay_preserves_candidate_until_persistent_signed_bias_requires_identity() -> None:
    result = replay_guarded_frames(
        [
            _frame(frame_id=0, visual_x_m=0.1, innovation=0.2),
            _frame(frame_id=1, visual_x_m=0.1, innovation=0.2),
            _frame(frame_id=2, visual_x_m=0.5, innovation=3.0),
            _frame(frame_id=3, visual_x_m=0.5, innovation=3.0),
            _frame(frame_id=4, visual_x_m=0.5, innovation=3.0),
        ],
        ema_alpha=0.1,
        ema_threshold=0.8,
    )

    assert [frame.visual_accepted for frame in result] == [True, True, True, True, False]
    assert result[-1].fused_x_m == result[-1].radar_x_m
    assert result[-1].fused_y_m == result[-1].radar_y_m


def test_replay_respects_policy_identity_without_updating_temporal_guard() -> None:
    result = replay_guarded_frames(
        [
            _frame(frame_id=0, visual_x_m=0.5, innovation=3.0),
            CausalGuardReplayFrame(
                sequence="seq_a",
                frame_id=1,
                policy_admitted=False,
                radar_x_m=0.0,
                radar_y_m=4.0,
                candidate_x_m=0.5,
                target_x_m=0.0,
                target_y_m=4.0,
                score=0.9,
                signed_innovation=3.0,
            ),
            _frame(frame_id=2, visual_x_m=0.1, innovation=0.2),
        ],
        ema_alpha=0.1,
        ema_threshold=0.8,
    )

    assert [frame.visual_accepted for frame in result] == [True, False, True]


def test_guard_off_keeps_each_policy_admitted_candidate() -> None:
    frames = [
        _frame(frame_id=0, visual_x_m=0.5, innovation=3.0),
        _frame(frame_id=1, visual_x_m=0.5, innovation=3.0),
        _frame(frame_id=2, visual_x_m=0.5, innovation=3.0),
    ]

    result = replay_guarded_frames(
        frames,
        ema_alpha=0.1,
        ema_threshold=0.8,
        guard_enabled=False,
    )

    assert [frame.visual_accepted for frame in result] == [True, True, True]
    assert all(frame.fused_x_m == 0.5 for frame in result)


def test_admission_overrides_preserve_warmup_and_identity_contract() -> None:
    frames = [
        _frame(frame_id=0, visual_x_m=0.2, innovation=0.1),
        _frame(frame_id=31, visual_x_m=0.2, innovation=0.1),
        replace(
            _frame(frame_id=32, visual_x_m=0.2, innovation=0.1),
            policy_admitted=False,
        ),
    ]

    always_admit = override_policy_admission(
        frames,
        mode="always-admit",
        warmup_frames=32,
    )
    always_identity = override_policy_admission(
        frames,
        mode="always-identity",
        warmup_frames=32,
    )

    assert [frame.policy_admitted for frame in always_admit] == [False, False, True]
    assert not any(frame.policy_admitted for frame in always_identity)


def test_guarded_replay_exports_final_compositor_rows_for_sequence_bootstrap() -> None:
    replay = replay_guarded_frames(
        [_frame(frame_id=0, visual_x_m=0.25, innovation=0.1)],
        ema_alpha=0.1,
        ema_threshold=0.8,
    )

    rows = bootstrap_rows_from_guarded_replay(replay)

    assert rows == [
        {
            "sequence": "seq_a",
            "target_x_m": 0.0,
            "target_y_m": 4.0,
            "radar_x_m": 0.0,
            "radar_y_m": 4.0,
            "candidate_x_m": 0.25,
            "candidate_y_m": 4.0,
            "radar_score": 0.9,
            "candidate_score": 0.9,
            "accepted": True,
        }
    ]
