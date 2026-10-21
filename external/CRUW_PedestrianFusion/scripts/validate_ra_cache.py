#!/usr/bin/env python3
"""Validate an existing protocol-only CRUW RA cache and write its manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cruw_pedestrian_fusion.data import build_frame_index
from cruw_pedestrian_fusion.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    fit_records = build_frame_index(args.data_root, protocol, args.candidates, "fit")
    validation_records = build_frame_index(args.data_root, protocol, args.candidates, "validation")
    if not fit_records or not validation_records:
        raise RuntimeError("protocol index is empty")
    cache_records = []
    for sequence_id in protocol.all_sequences:
        path = args.cache_root / f"{sequence_id}.npy"
        array = np.load(path, mmap_mode="r")
        if array.ndim != 3 or array.shape[1:] != (128, 128):
            raise ValueError(f"invalid cache shape at {path}: {array.shape}")
        cache_records.append({"sequence_id": sequence_id, "frame_count": int(array.shape[0]), "path": str(path)})
    payload = {
        "protocol": str(args.protocol),
        "source_partition_read": "train",
        "official_test_read": False,
        "fit_records": len(fit_records),
        "validation_records": len(validation_records),
        "eligible_records": len(fit_records) + len(validation_records),
        "sequences": cache_records,
    }
    (args.cache_root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
