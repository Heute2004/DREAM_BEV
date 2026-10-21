from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rv_bearing_net.training_teacher import (
    extract_radar_candidates,
    fit_geometry_calibration,
)


CACHE_VERSION = "RVBearingTrainingTeacherV1"
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
SAMPLE_RATE_HZ = 6.25e6
CHIRP_SLOPE_HZ_PER_S = 9.994e12
RANGE_FFT_SIZE = 512
RANGE_RESOLUTION_M = (
    SPEED_OF_LIGHT_M_PER_S
    * SAMPLE_RATE_HZ
    / (2.0 * CHIRP_SLOPE_HZ_PER_S * RANGE_FFT_SIZE)
)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _geometry_valid(row: dict[str, str] | None) -> bool:
    if row is None or not _as_bool(row.get("label_valid", False)):
        return False
    x_m = _as_float(row.get("x_m"))
    z_m = _as_float(row.get("z_m"))
    quality = _as_float(row.get("local_quality"), 0.0)
    status = str(row.get("raw_status", "")).strip().lower()
    return (
        math.isfinite(x_m)
        and math.isfinite(z_m)
        and z_m > 0.05
        and quality >= 0.20
        and status not in {"", "invalid", "missing", "no_depth"}
    )


def _teacher_sigma(
    row: dict[str, str],
    residual_scale_rad: float,
) -> float:
    quality = float(np.clip(_as_float(row.get("local_quality"), 0.0), 0.0, 1.0))
    mad_mm = max(0.0, _as_float(row.get("depth_mad_mm"), 500.0))
    quality_term = math.radians(1.5 * (1.0 - quality))
    depth_term = math.radians(0.5 * min(1.0, mad_mm / 200.0))
    sigma = math.sqrt(
        max(residual_scale_rad, math.radians(0.25)) ** 2
        + quality_term**2
        + depth_term**2
    )
    return float(np.clip(sigma, math.radians(0.25), math.radians(3.0)))


def resolve_geometry_sources(
    *,
    data_root: Path,
    geometry_root: Path | None,
    sequence: str,
) -> tuple[Path, Path]:
    root = geometry_root if geometry_root is not None else data_root
    sequence_root = root / sequence
    return (
        sequence_root
        / "CameraDepthCandidates"
        / "camera_depth_candidates.csv",
        sequence_root / "BEVFinalLabels" / "metadata.json",
    )


def _build_sequence(
    *,
    sequence: str,
    rows: list[dict[str, str]],
    data_root: Path,
    geometry_root: Path | None,
    output_root: Path,
    candidate_count: int,
    range_half_width_bins: int,
    overwrite: bool,
) -> dict[str, object]:
    output_path = output_root / f"{sequence}.npz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"teacher cache already exists: {output_path}")
    depth_csv, metadata_path = resolve_geometry_sources(
        data_root=data_root,
        geometry_root=geometry_root,
        sequence=sequence,
    )
    depth_by_frame = {
        int(row["frame_id"]): row for row in _read_rows(depth_csv)
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    x_offset_m = float(metadata.get("radar_camera_x_offset_m", 0.0))
    rows = sorted(rows, key=lambda item: int(item["frame_id"]))
    frame_ids = np.array([int(row["frame_id"]) for row in rows], dtype=np.int64)
    target_x = np.array([_as_float(row.get("final_bev_x_m")) for row in rows])
    target_y = np.array([_as_float(row.get("final_bev_y_m")) for row in rows])
    label_valid = np.array(
        [_as_bool(row.get("final_bev_label_valid", False)) for row in rows],
        dtype=bool,
    )
    target_theta = np.arctan2(target_x, np.maximum(target_y, 1e-3))
    camera_theta = np.full(frame_ids.shape, np.nan, dtype=np.float64)
    geometry_valid = np.zeros(frame_ids.shape, dtype=bool)
    for index, frame_id in enumerate(frame_ids.tolist()):
        depth_row = depth_by_frame.get(frame_id)
        if not _geometry_valid(depth_row):
            continue
        assert depth_row is not None
        camera_theta[index] = math.atan2(
            _as_float(depth_row["x_m"]) + x_offset_m,
            _as_float(depth_row["z_m"]),
        )
        geometry_valid[index] = label_valid[index]
    calibration = fit_geometry_calibration(
        camera_theta,
        target_theta,
        geometry_valid,
    )
    geometry_theta = np.full(frame_ids.shape, np.nan, dtype=np.float32)
    geometry_sigma = np.full(
        frame_ids.shape, math.radians(3.0), dtype=np.float32
    )
    for index, frame_id in enumerate(frame_ids.tolist()):
        if not geometry_valid[index]:
            continue
        depth_row = depth_by_frame[frame_id]
        geometry_theta[index] = (
            calibration.slope * camera_theta[index]
            + calibration.offset_rad
        )
        geometry_sigma[index] = _teacher_sigma(
            depth_row, calibration.residual_scale_rad
        )

    candidate_theta = np.zeros(
        (frame_ids.size, candidate_count), dtype=np.float32
    )
    candidate_support = np.zeros_like(candidate_theta)
    candidate_valid = np.zeros_like(candidate_theta, dtype=bool)
    positive_index = np.zeros(frame_ids.size, dtype=np.int64)
    radar_candidate_frames = 0
    for index, row in enumerate(rows):
        if not label_valid[index] or not (
            math.isfinite(target_x[index]) and math.isfinite(target_y[index])
        ):
            continue
        ra_path = data_root / row["ra_path"]
        ra_mti_path = data_root / row["ra_mti_path"]
        candidates = extract_radar_candidates(
            ra=np.load(ra_path, allow_pickle=False),
            ra_mti=np.load(ra_mti_path, allow_pickle=False),
            target_theta_rad=float(target_theta[index]),
            target_range_m=float(math.hypot(target_x[index], target_y[index])),
            range_resolution_m=RANGE_RESOLUTION_M,
            candidate_count=candidate_count,
            range_half_width_bins=range_half_width_bins,
        )
        candidate_theta[index] = candidates.theta_rad
        candidate_support[index] = candidates.support
        candidate_valid[index] = candidates.valid
        positive_index[index] = candidates.positive_index
        if int(candidates.valid.sum()) >= 2:
            radar_candidate_frames += 1
    _atomic_savez(
        output_path,
        frame_ids=frame_ids,
        geometry_theta_rad=geometry_theta,
        geometry_sigma_rad=geometry_sigma,
        geometry_valid=geometry_valid,
        candidate_theta_rad=candidate_theta,
        candidate_support=candidate_support,
        candidate_valid=candidate_valid,
        candidate_positive_index=positive_index,
    )
    return {
        "sequence": sequence,
        "frames": int(frame_ids.size),
        "geometry_valid_frames": int(geometry_valid.sum()),
        "ranking_valid_frames": radar_candidate_frames,
        "calibration": asdict(calibration),
        "radar_camera_x_offset_m": x_offset_m,
        "geometry_depth_csv_sha256": _sha256(depth_csv),
        "geometry_metadata_sha256": _sha256(metadata_path),
        "cache_file": output_path.name,
        "cache_sha256": _sha256(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build training-only geometry and radar-candidate teachers"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--range-half-width-bins", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-sequence", type=int, default=1)
    parser.add_argument("--end-sequence", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.candidate_count < 2 or args.workers < 1:
        raise ValueError("candidate-count must be >=2 and workers must be positive")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _read_rows(args.manifest):
        sequence = row["sequence"].strip().lower()
        sequence_number = int(sequence.split("_")[-1])
        if not args.start_sequence <= sequence_number <= args.end_sequence:
            continue
        if _as_bool(row.get("sequence_excluded", False)):
            continue
        grouped.setdefault(sequence, []).append(row)
    if not grouped:
        raise ValueError("no manifest sequences selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    futures = {}
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sequence, rows in grouped.items():
            future = executor.submit(
                _build_sequence,
                sequence=sequence,
                rows=rows,
                data_root=args.data_root,
                geometry_root=args.geometry_root,
                output_root=args.output_root,
                candidate_count=args.candidate_count,
                range_half_width_bins=args.range_half_width_bins,
                overwrite=args.overwrite,
            )
            futures[future] = sequence
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: str(item["sequence"]))
    report = {
        "cache_version": CACHE_VERSION,
        "training_only_teacher": True,
        "runtime_inputs": ["causal_left_rgb"],
        "prohibited_runtime_inputs": [
            "depth", "right_rgb", "radar", "ground_truth", "future_frames"
        ],
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "geometry_root": (
            str(args.geometry_root) if args.geometry_root is not None else None
        ),
        "candidate_count": args.candidate_count,
        "range_half_width_bins": args.range_half_width_bins,
        "sequences": results,
        "sequence_count": len(results),
        "frame_count": sum(int(item["frames"]) for item in results),
        "geometry_valid_frames": sum(
            int(item["geometry_valid_frames"]) for item in results
        ),
        "ranking_valid_frames": sum(
            int(item["ranking_valid_frames"]) for item in results
        ),
    }
    temporary = args.output_root / "cache_manifest.tmp.json"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, args.output_root / "cache_manifest.json")


if __name__ == "__main__":
    main()
