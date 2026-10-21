#!/usr/bin/env python3
"""Build fit-normalized CARRADA RA caches without opening the lockbox."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from carrada_dream_bev.data import build_all_scene_index, transform_ra
from carrada_dream_bev.protocol import load_protocol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pooled_db(path: Path) -> np.ndarray:
    return transform_ra(np.load(path), mean_db=0.0, std_db=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--include-locked-test",
        action="store_true",
        help="Append the official locked test only after every model and evaluator is frozen.",
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.include_locked_test:
        manifest_path = args.output_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("build Fit/Development cache before appending locked test")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "test" in manifest.get("splits", {}):
            raise FileExistsError("locked test cache already exists")
        if manifest.get("protocol_sha256") != sha256(args.protocol):
            raise RuntimeError("protocol changed after the development cache was frozen")
        mean_db = float(manifest["mean_db"])
        std_db = float(manifest["std_db"])
        test_records = build_all_scene_index(args.data_root, protocol, "test")
        split_manifest: dict[str, object] = {}
        for sequence_id in sorted({record.sequence_id for record in test_records}):
            frame_ids = [record.frame_id for record in test_records if record.sequence_id == sequence_id]
            cache = np.stack(
                [
                    transform_ra(
                        np.load(args.data_root / sequence_id / "range_angle_numpy" / f"{frame_id:06d}.npy"),
                        mean_db=mean_db,
                        std_db=std_db,
                    )
                    for frame_id in frame_ids
                ]
            ).astype(np.float16)
            np.save(args.output_root / f"{sequence_id}.npy", cache)
            np.save(args.output_root / f"{sequence_id}_frames.npy", np.asarray(frame_ids, dtype=np.int64))
            split_manifest[sequence_id] = {"frames": len(frame_ids), "shape": list(cache.shape)}
        manifest["splits"]["test"] = split_manifest
        manifest["locked_test_append"] = {
            "explicit": True,
            "frames": len(test_records),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps({"locked_test_frames": len(test_records), "status": "appended"}, indent=2))
        return

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"cache output is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    records = {
        split: build_all_scene_index(args.data_root, protocol, split)
        for split in ("fit", "development")
    }
    fit_values = []
    for record in records["fit"]:
        fit_values.append(pooled_db(args.data_root / record.sequence_id / "range_angle_numpy" / f"{record.frame_id:06d}.npy"))
    fit_stack = np.stack(fit_values)
    mean_db = float(fit_stack.mean(dtype=np.float64))
    std_db = float(fit_stack.std(dtype=np.float64))
    del fit_values, fit_stack
    manifest: dict[str, object] = {
        "input": "CARRADA range_angle_numpy",
        "transform": "20log10 -> reverse range -> 2x2 mean pool -> Fit-only z-score",
        "mean_db": mean_db,
        "std_db": std_db,
        "protocol_sha256": sha256(args.protocol),
        "splits": {},
    }
    for split, split_records in records.items():
        sequence_ids = sorted({record.sequence_id for record in split_records})
        split_manifest: dict[str, object] = {}
        for sequence_id in sequence_ids:
            frame_ids = [record.frame_id for record in split_records if record.sequence_id == sequence_id]
            cache = np.stack(
                [
                    transform_ra(
                        np.load(args.data_root / sequence_id / "range_angle_numpy" / f"{frame_id:06d}.npy"),
                        mean_db=mean_db,
                        std_db=std_db,
                    )
                    for frame_id in frame_ids
                ]
            ).astype(np.float16)
            np.save(args.output_root / f"{sequence_id}.npy", cache)
            np.save(args.output_root / f"{sequence_id}_frames.npy", np.asarray(frame_ids, dtype=np.int64))
            split_manifest[sequence_id] = {"frames": len(frame_ids), "shape": list(cache.shape)}
        manifest["splits"][split] = split_manifest
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mean_db": mean_db,
                "std_db": std_db,
                "fit_frames": len(records["fit"]),
                "development_frames": len(records["development"]),
                "locked_test_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
