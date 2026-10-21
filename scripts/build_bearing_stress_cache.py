#!/usr/bin/env python3
"""Build deterministic, Development-only perturbations of frozen bearing caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from rvfusion.bearing_stress import (
    BearingStressError,
    apply_bearing_bias,
    apply_temporal_lag,
    apply_visual_dropout,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--output-cache-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("bearing_bias", "temporal_lag", "visual_dropout"),
        required=True,
    )
    parser.add_argument("--bias-deg", type=float)
    parser.add_argument("--lag-frames", type=int)
    parser.add_argument("--dropout-probability", type=float)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


def ensure_development_only(*paths: Path) -> None:
    forbidden_fragments = ("locked_test", "locked-test", "lockedtest")
    for path in paths:
        normalized = str(path).lower().replace("\\", "/")
        components = set(normalized.split("/"))
        if (
            any(token in normalized for token in forbidden_fragments)
            or "locked" in components
            or "test" in components
        ):
            raise BearingStressError(
                f"Development-only stress builder rejects path: {path}"
            )


def transform(payload: dict[str, np.ndarray], args: argparse.Namespace):
    if args.mode == "bearing_bias":
        if args.bias_deg is None:
            raise BearingStressError("--bias-deg is required for bearing_bias")
        return apply_bearing_bias(payload, bias_rad=float(np.deg2rad(args.bias_deg)))
    if args.mode == "temporal_lag":
        if args.lag_frames is None:
            raise BearingStressError("--lag-frames is required for temporal_lag")
        return apply_temporal_lag(payload, lag_frames=args.lag_frames)
    if args.dropout_probability is None:
        raise BearingStressError(
            "--dropout-probability is required for visual_dropout"
        )
    return apply_visual_dropout(
        payload,
        probability=args.dropout_probability,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    source = args.source_cache_root.resolve()
    output = args.output_cache_root.resolve()
    ensure_development_only(source, output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite stress cache: {output}")
    source_manifest = source / "cache_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"bearing cache manifest not found: {source_manifest}")
    source_files = sorted(source.glob("*.npz"))
    if not source_files:
        raise FileNotFoundError(f"no bearing cache files found under {source}")
    output.mkdir(parents=True)
    summaries = []
    for source_file in source_files:
        with np.load(source_file, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
        stressed, summary = transform(payload, args)
        np.savez_compressed(output / source_file.name, **stressed)
        summaries.append(
            {
                "sequence_cache": source_file.name,
                "affected_frames": summary.affected_frames,
                "total_frames": summary.total_frames,
            }
        )
    # Keep the frozen cache binding byte-for-byte identical so the existing
    # evaluator can verify the active data contract. Stress provenance lives
    # in a separate sidecar instead of masquerading as a new bearing cache.
    (output / "cache_manifest.json").write_bytes(source_manifest.read_bytes())
    stress_metadata = {
        "schema": "dream_bev_development_bearing_stress_v1",
        "scope": "Development-only frozen-cache perturbation",
        "source_cache_root": str(source),
        "source_cache_manifest_sha256": sha256(source_manifest),
        "mode": args.mode,
        "bias_deg": args.bias_deg,
        "lag_frames": args.lag_frames,
        "dropout_probability": args.dropout_probability,
        "seed": args.seed,
        "sequence_count": len(summaries),
        "affected_frames": sum(item["affected_frames"] for item in summaries),
        "total_frames": sum(item["total_frames"] for item in summaries),
        "files": summaries,
    }
    (output / "bearing_stress_metadata.json").write_text(
        json.dumps(stress_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stress_metadata, indent=2))


if __name__ == "__main__":
    main()
