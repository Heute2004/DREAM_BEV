#!/usr/bin/env python3
"""Independently replay one frozen paper expert on CRUW Development records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cruw_pedestrian_fusion.allscene import (
    CRUWBEVGeometry,
    RadarBEVWindowDataset,
    build_all_scene_index,
)
from cruw_pedestrian_fusion.canonical_dream_bev import DREAMBEV
from cruw_pedestrian_fusion.geometry import load_cruw_grids
from cruw_pedestrian_fusion.paper_experts import PAPER_EXPERT_KINDS, CRUWPaperExpert
from cruw_pedestrian_fusion.protocol import load_protocol
from scripts.train_cruw_paper_expert import evaluate, select_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cruw-devkit-root", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-kind", choices=PAPER_EXPERT_KINDS, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--identity-negative-ratio", type=float, default=0.20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    geometry = CRUWBEVGeometry()
    protocol = load_protocol(args.protocol)
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    development_sequences = {
        int(sequence_id) for sequence_id in payload["single_person_validation_sequences"]
    }
    range_grid, angle_grid = load_cruw_grids(args.data_root, args.cruw_devkit_root)
    records = build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride)
    selected, positive_records, negative_records = select_records(
        records, development_sequences, geometry, args.identity_negative_ratio
    )
    loader = DataLoader(
        RadarBEVWindowDataset(
            selected, args.cache_root, range_grid, angle_grid, args.clip_length, geometry
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    anchor = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    anchor.load_state_dict(
        torch.load(args.anchor_checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    expert = CRUWPaperExpert(args.expert_kind, geometry).to(device)
    expert.load_state_dict(
        torch.load(args.expert_checkpoint, map_location=device, weights_only=False)["model"],
        strict=True,
    )
    report = {
        "stage": "independent_frozen_paper_expert_development_replay",
        "expert_kind": args.expert_kind,
        "development_sequences": sorted(development_sequences),
        "selected_records": len(selected),
        "positive_records": positive_records,
        "negative_records": negative_records,
        "metrics": evaluate(anchor, expert, loader, device),
        "anchor_sha256": sha256(args.anchor_checkpoint),
        "expert_sha256": sha256(args.expert_checkpoint),
        "protocol_sha256": sha256(args.protocol),
        "official_test_read": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
