#!/usr/bin/env python3
"""Replay frozen CARRADA predictions on the fixed-identity episode manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from carrada_dream_bev.metrics import localization_metrics


HEADS = {
    "anchor": ("anchor_x_m", "anchor_y_m"),
    "expert": ("expert_x_m", "expert_y_m"),
    "fusion": ("fusion_x_m", "fusion_y_m"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_frames(episodes_path: Path, split: str) -> tuple[list[dict[str, object]], dict[tuple[str, int], str]]:
    payload = json.loads(episodes_path.read_text(encoding="utf-8"))
    episodes = payload.get("splits", {}).get(split)
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"episode manifest has no {split!r} episodes")
    frame_to_episode: dict[tuple[str, int], str] = {}
    for episode in episodes:
        sequence_id = str(episode["sequence_id"])
        episode_id = str(episode["episode_id"])
        start = int(episode["start_frame"])
        end = int(episode["end_frame"])
        declared_count = int(episode["frame_count"])
        if end < start or end - start + 1 != declared_count:
            raise RuntimeError(f"invalid frame range in {episode_id}")
        for frame_id in range(start, end + 1):
            key = (sequence_id, frame_id)
            if key in frame_to_episode:
                raise RuntimeError(f"overlapping episode frame: {key}")
            frame_to_episode[key] = episode_id
    return episodes, frame_to_episode


def replay_episode_predictions(
    predictions_path: Path,
    episodes_path: Path,
    split: str = "development",
) -> dict[str, object]:
    """Filter one frozen prediction table to the exact declared episode frames."""

    episodes, frame_to_episode = _episode_frames(episodes_path, split)
    selected: dict[tuple[str, int], dict[str, str]] = {}
    required = {"sequence_id", "frame_id", "target_x_m", "target_y_m", "radar_score"}
    required.update(column for pair in HEADS.values() for column in pair)
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise RuntimeError(f"prediction table missing required columns: {missing}")
        for row in reader:
            key = (str(row["sequence_id"]), int(row["frame_id"]))
            if key in frame_to_episode:
                if key in selected:
                    raise RuntimeError(f"duplicate prediction frame: {key}")
                selected[key] = row
    missing = sorted(set(frame_to_episode).difference(selected))
    if missing:
        raise RuntimeError(f"missing prediction frames for fixed episodes: {missing[:5]}")

    ordered = [selected[key] for key in sorted(frame_to_episode)]
    targets_x = np.asarray([float(row["target_x_m"]) for row in ordered], dtype=np.float64)
    targets_y = np.asarray([float(row["target_y_m"]) for row in ordered], dtype=np.float64)
    scores = np.asarray([float(row["radar_score"]) for row in ordered], dtype=np.float64)
    metrics = {
        name: localization_metrics(
            np.asarray([float(row[x_key]) for row in ordered], dtype=np.float64),
            np.asarray([float(row[y_key]) for row in ordered], dtype=np.float64),
            targets_x,
            targets_y,
            scores,
        )
        for name, (x_key, y_key) in HEADS.items()
    }
    return {
        "split": split,
        "episode_count": len(episodes),
        "frame_count": len(ordered),
        "episode_ids": [str(episode["episode_id"]) for episode in episodes],
        "metrics": metrics,
        "rows": ordered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "fit"), default="development")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = replay_episode_predictions(args.predictions, args.episodes, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = result.pop("rows")
    contract = {
        "purpose": "prediction-only replay on the fixed-identity CARRADA episode manifest",
        "predictions_sha256": sha256(args.predictions),
        "episodes_sha256": sha256(args.episodes),
        "split": args.split,
        "model_or_checkpoint_loaded": False,
    }
    (args.output_dir / "replay_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "episode_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
