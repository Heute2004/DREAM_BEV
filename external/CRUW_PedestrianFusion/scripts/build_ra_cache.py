#!/usr/bin/env python3
"""Cache four-chirp CRUW RA amplitudes for the frozen pedestrian-only protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cruw_pedestrian_fusion.data import CHIRP_IDS, build_frame_index
from cruw_pedestrian_fusion.protocol import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def build_sequence(data_root: Path, output_root: Path, sequence_id: int) -> dict[str, int | str]:
    radar_root = data_root / "sequences" / "train" / str(sequence_id) / "RADAR_RA_H"
    frame_ids = sorted({int(path.name.split("_")[0]) for path in radar_root.glob("*.npy")})
    if not frame_ids or frame_ids != list(range(frame_ids[-1] + 1)):
        raise ValueError(f"non-contiguous RA frame ids in {radar_root}")
    output_path = output_root / f"{sequence_id}.npy"
    packed = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(frame_ids), 128, 128),
    )
    for frame_id in frame_ids:
        amplitudes = []
        for chirp_id in CHIRP_IDS:
            source = np.load(radar_root / f"{frame_id:06d}_{chirp_id:04d}.npy", allow_pickle=False)
            if source.shape != (128, 128, 2):
                raise ValueError(f"unexpected RA shape {source.shape} at sequence {sequence_id}, frame {frame_id}")
            amplitudes.append(np.log1p(np.hypot(source[..., 0], source[..., 1])))
        packed[frame_id] = np.mean(amplitudes, axis=0, dtype=np.float32).astype(np.float16)
    packed.flush()
    return {"sequence_id": sequence_id, "frame_count": len(frame_ids), "path": str(output_path)}


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"cache directory is not empty: {args.output_root}")
    protocol = load_protocol(args.protocol)
    records = build_frame_index(args.data_root, protocol, args.candidates, "fit")
    records += build_frame_index(args.data_root, protocol, args.candidates, "validation")
    args.output_root.mkdir(parents=True, exist_ok=False)
    outputs = [build_sequence(args.data_root, args.output_root, sequence_id) for sequence_id in protocol.all_sequences]
    (args.output_root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": str(args.protocol),
                "source_partition_read": "train",
                "official_test_read": False,
                "chirp_aggregation": "mean(log1p(hypot(real,imag))) over chirps 0,64,128,192",
                "eligible_records": len(records),
                "sequences": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "sequences": len(outputs), "eligible_records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
