#!/usr/bin/env python3
"""Freeze the 24/6 CARRADA Train/Validation and strict single-target episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from carrada_dream_bev.data import build_all_scene_index, derive_single_target_episodes
from carrada_dream_bev.protocol import CARRADAProtocol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_row(episode) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "sequence_id": episode.sequence_id,
        "target_instance_id": episode.target_instance_id,
        "start_frame": episode.start_frame,
        "end_frame": episode.end_frame,
        "frame_count": episode.frame_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path, required=True)
    parser.add_argument("--episode-output", type=Path, required=True)
    parser.add_argument("--min-episode-frames", type=int, default=30)
    args = parser.parse_args()

    reference_path = args.data_root / "data_seq_ref.json"
    annotations_path = args.data_root / "annotations_frame_oriented.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    validation = sorted(
        sequence for sequence, item in reference.items() if item["split"].lower() == "validation"
    )
    train = sorted(sequence for sequence in reference if sequence not in set(validation))
    protocol = CARRADAProtocol(tuple(train), tuple(validation), ())
    if len(train) != 24 or len(validation) != 6 or len(protocol.all_sequences) != 30:
        raise RuntimeError("CARRADA Train/Validation protocol must contain 24/6 complete sequences")

    train_records = build_all_scene_index(args.data_root, protocol, "train")
    validation_records = build_all_scene_index(args.data_root, protocol, "validation")
    train_episodes = derive_single_target_episodes(
        train_records,
        min_length_frames=args.min_episode_frames,
    )
    validation_episodes = derive_single_target_episodes(
        validation_records,
        min_length_frames=args.min_episode_frames,
    )
    train_human_sequences = sorted({episode.sequence_id for episode in train_episodes})
    validation_human_sequences = sorted({episode.sequence_id for episode in validation_episodes})
    if set(train_human_sequences) & set(validation_human_sequences):
        raise RuntimeError("single-target sequence leakage")

    protocol_payload = {
        "dataset": "CARRADA",
        "purpose": "complete-sequence Train/Validation external transfer study",
        "selection": "official Validation retained; official Train and Test merged into Train",
        "train_sequences": train,
        "validation_sequences": validation,
        "test_sequences": [],
        "sequence_overlap": [],
        "train_sequence_count": len(train),
        "validation_sequence_count": len(validation),
        "annotations_sha256": sha256(annotations_path),
        "data_seq_ref_sha256": sha256(reference_path),
    }
    episode_payload = {
        "dataset": "CARRADA",
        "contract": "strict continuous single-pedestrian episode with one fixed official instance ID",
        "min_episode_frames": args.min_episode_frames,
        "cars_and_cyclists_retained_as_distractors": True,
        "multi_pedestrian_frames_excluded": True,
        "history_clamped_at_episode_start": True,
        "splits": {
            "fit": [episode_row(value) for value in train_episodes],
            "development": [episode_row(value) for value in validation_episodes],
        },
        "summary": {
            "train_human_sequences": train_human_sequences,
            "validation_human_sequences": validation_human_sequences,
            "train_episode_count": len(train_episodes),
            "validation_episode_count": len(validation_episodes),
            "train_episode_frames": sum(value.frame_count for value in train_episodes),
            "validation_episode_frames": sum(value.frame_count for value in validation_episodes),
        },
    }
    args.protocol_output.parent.mkdir(parents=True, exist_ok=True)
    args.episode_output.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_output.write_text(json.dumps(protocol_payload, indent=2), encoding="utf-8")
    args.episode_output.write_text(json.dumps(episode_payload, indent=2), encoding="utf-8")
    print(json.dumps({**protocol_payload, **episode_payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
