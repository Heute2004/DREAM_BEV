#!/usr/bin/env python3
"""Export a frozen CARRADA radar checkpoint for episode-only replay.

This utility never trains or selects a checkpoint. Every exported model uses
the frozen public 24/6 Development protocol and writes the common prediction
schema consumed by ``replay_fixed_identity_episodes.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from carrada_dream_bev.canonical_dream_bev import DREAMBEV
from carrada_dream_bev.data import (
    CARRADABEVGeometry,
    RadarBEVWindowDataset,
    apply_single_target_episode_manifest,
    build_all_scene_index,
    carrada_ra_grids,
    load_single_target_episode_manifest,
)
from carrada_dream_bev.metrics import decode_bev_logits
from carrada_dream_bev.protocol import load_protocol
from carrada_dream_bev.public_fcn_baseline import CARRADAFCN8sBEV
from carrada_dream_bev.public_rodnet import PublicRODNetBEV
from carrada_dream_bev.public_temporal_baselines import PublicTemporalBEV
from carrada_dream_bev.public_temporal_baselines import T_RODNET_REFERENCE_CONFIGURATION


MODEL_CHOICES = (
    "dream_bev",
    "rodnet_cdc",
    "rodnet_hg",
    "rodnet_hgwi",
    "t_rodnet",
    "fcn8s",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_prediction_rows(
    batch: dict[str, Any],
    decoded: dict[str, Any],
) -> list[dict[str, object]]:
    """Materialize one radar-only head under all common replay columns."""

    rows: list[dict[str, object]] = []
    batch_size = len(batch["sequence_id"])
    for index in range(batch_size):
        x_m = float(decoded["x_m"][index])
        y_m = float(decoded["y_m"][index])
        rows.append(
            {
                "sequence_id": str(batch["sequence_id"][index]),
                "frame_id": int(batch["frame_id"][index]),
                "target_count": int(batch["target_count"][index]),
                "target_x_m": float(batch["single_x_m"][index]),
                "target_y_m": float(batch["single_y_m"][index]),
                "radar_score": float(decoded["score"][index]),
                "anchor_x_m": x_m,
                "anchor_y_m": y_m,
                "expert_x_m": x_m,
                "expert_y_m": y_m,
                "fusion_x_m": x_m,
                "fusion_y_m": y_m,
            }
        )
    return rows


def episode_development_records(records, manifest_path: Path):
    """Select fixed-identity Development episodes and preserve causal reset markers."""

    episodes = load_single_target_episode_manifest(manifest_path, "development")
    selected = apply_single_target_episode_manifest(records, episodes)
    if not selected:
        raise RuntimeError("development episode manifest selected no synchronized records")
    return selected


def build_model(
    name: str,
    *,
    public_root: Path | None,
    geometry: CARRADABEVGeometry,
    t_rodnet_configuration: dict[str, int] | None = None,
) -> torch.nn.Module:
    range_grid, angle_grid = carrada_ra_grids()
    if name == "dream_bev":
        return DREAMBEV(range_grid, angle_grid, geometry)
    if name in {"rodnet_cdc", "rodnet_hg", "rodnet_hgwi"}:
        return PublicRODNetBEV(name, public_root, range_grid, angle_grid, geometry)
    if name == "t_rodnet":
        return PublicTemporalBEV(
            name,
            public_root,
            range_grid,
            angle_grid,
            geometry,
            t_rodnet_configuration=t_rodnet_configuration,
        )
    if name == "fcn8s":
        return CARRADAFCN8sBEV(public_root, range_grid, angle_grid, geometry)
    raise ValueError(f"unsupported frozen model: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--development-episode-manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--public-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development",), default="development")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--t-rodnet-embed-dim", type=int)
    parser.add_argument("--t-rodnet-win-size", type=int)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if args.model != "dream_bev" and args.public_root is None:
        raise ValueError("public-root is required for a public architecture")

    protocol = load_protocol(args.protocol)
    records = build_all_scene_index(args.data_root, protocol, args.split)
    expected_sequences = set(protocol.sequence_ids(args.split))
    if {record.sequence_id for record in records} != expected_sequences:
        raise RuntimeError("records do not exactly match the frozen Development sequences")
    if args.development_episode_manifest is not None:
        records = episode_development_records(records, args.development_episode_manifest)
    geometry = CARRADABEVGeometry()
    dataset = RadarBEVWindowDataset(records, args.cache_root, args.clip_length, geometry)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda")
    t_rodnet_configuration = None
    if args.model == "t_rodnet" and (
        args.t_rodnet_embed_dim is not None or args.t_rodnet_win_size is not None
    ):
        t_rodnet_configuration = dict(T_RODNET_REFERENCE_CONFIGURATION)
        if args.t_rodnet_embed_dim is not None:
            t_rodnet_configuration["embed_dim"] = args.t_rodnet_embed_dim
        if args.t_rodnet_win_size is not None:
            t_rodnet_configuration["win_size"] = args.t_rodnet_win_size
    model = build_model(
        args.model,
        public_root=args.public_root,
        geometry=geometry,
        t_rodnet_configuration=t_rodnet_configuration,
    ).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    rows: list[dict[str, object]] = []
    for batch in loader:
        logits = model(batch["radar"].to(device, non_blocking=True)).logits
        rows.extend(frozen_prediction_rows(batch, decode_bev_logits(logits, geometry)))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = list(rows[0])
    predictions_path = args.output_dir / "frame_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "purpose": "frozen CARRADA radar checkpoint prediction export for fixed-identity replay",
        "model": args.model,
        "split": args.split,
        "frame_count": len(rows),
        "checkpoint_sha256": sha256(args.checkpoint),
        "protocol_sha256": sha256(args.protocol),
        "locked_test_read": False,
        "training_or_selection_performed": False,
        "development_evaluation_contract": (
            "fixed_identity_episode_manifest"
            if args.development_episode_manifest is not None
            else "complete_development_sequences"
        ),
        "development_episode_manifest_sha256": (
            sha256(args.development_episode_manifest)
            if args.development_episode_manifest is not None
            else None
        ),
    }
    if args.model == "t_rodnet":
        metadata["source_trainer_configuration"] = model.public_source[
            "source_trainer_configuration"
        ]
    (args.output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
