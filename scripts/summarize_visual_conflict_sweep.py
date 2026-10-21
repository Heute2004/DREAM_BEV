"""Summarize a frozen Development-only visual-conflict sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


GRID: dict[str, tuple[tuple[str, float, str], ...]] = {
    "bearing_bias": tuple((str(value), float(value), f"{value} deg") for value in range(1, 7)),
    "temporal_lag": tuple((str(value), float(value), f"{value} frames") for value in (2, 4, 8)),
    "visual_dropout": tuple(
        (str(percent), percent / 100.0, f"{percent}%") for percent in (25, 50, 75)
    ),
}
METHODS = ("guard_off", "protected")


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        result: list[str] = []
        for key, item in value.items():
            result.extend(_flatten_strings(key))
            result.extend(_flatten_strings(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_strings(item))
        return result
    return [str(value)]


def load_metrics(path: Path, *, expected_frames: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    provenance = " ".join(_flatten_strings(payload.get("configuration", {}))).lower()
    normalized = provenance.replace("-", "_").replace(" ", "_")
    if "locked_test" in normalized or "lockedblind" in normalized:
        raise ValueError(f"locked test provenance is forbidden: {path}")
    if int(payload.get("frames", -1)) != expected_frames:
        raise ValueError(f"unexpected frame count in {path}: {payload.get('frames')}")
    for key in (
        "radar",
        "guarded_fusion",
        "visual_acceptance_rate",
        "harmed_rate",
    ):
        if key not in payload:
            raise ValueError(f"missing {key} in {path}")
    return payload


def _row(
    *,
    perturbation: str,
    severity: float,
    severity_label: str,
    method: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    fusion = payload["guarded_fusion"]
    radar = payload["radar"]
    return {
        "perturbation": perturbation,
        "severity": severity,
        "severity_label": severity_label,
        "method": method,
        "frames": int(payload["frames"]),
        "radar_ap_0p30m": float(radar["ap_0p3m"]),
        "ap_0p25m": float(fusion["ap_0p25m"]),
        "ap_0p30m": float(fusion["ap_0p3m"]),
        "ap_0p50m": float(fusion["ap_0p5m"]),
        "p90_error_m": float(fusion["p90_location_error_m"]),
        "visual_admission_rate": float(payload["visual_acceptance_rate"]),
        "harmed_frame_rate": float(payload["harmed_rate"]),
    }


def build_summary(
    *,
    run_root: Path,
    clean_guard_off: Path,
    clean_protected: Path,
    expected_frames: int,
) -> list[dict[str, Any]]:
    clean = {
        "guard_off": load_metrics(clean_guard_off, expected_frames=expected_frames),
        "protected": load_metrics(clean_protected, expected_frames=expected_frames),
    }
    radar_reference = clean["guard_off"]["radar"]["ap_0p3m"]
    rows: list[dict[str, Any]] = []
    for perturbation, levels in GRID.items():
        for method in METHODS:
            rows.append(
                _row(
                    perturbation=perturbation,
                    severity=0.0,
                    severity_label="clean",
                    method=method,
                    payload=clean[method],
                )
            )
        for level_dir, severity, severity_label in levels:
            for method in METHODS:
                path = run_root / "replay" / perturbation / level_dir / method / "metrics.json"
                payload = load_metrics(path, expected_frames=expected_frames)
                if abs(float(payload["radar"]["ap_0p3m"]) - float(radar_reference)) > 1e-9:
                    raise ValueError(f"radar reference changed at {perturbation}/{level_dir}/{method}")
                rows.append(
                    _row(
                        perturbation=perturbation,
                        severity=severity,
                        severity_label=severity_label,
                        method=method,
                        payload=payload,
                    )
                )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clean-guard-off", type=Path, required=True)
    parser.add_argument("--clean-protected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=9969)
    args = parser.parse_args()

    forbidden = str(args.output_dir).lower().replace("-", "_")
    if "locked_test" in forbidden:
        raise SystemExit("Locked Test is forbidden for the visual-conflict sweep.")

    rows = build_summary(
        run_root=args.run_root,
        clean_guard_off=args.clean_guard_off,
        clean_protected=args.clean_protected,
        expected_frames=args.expected_frames,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "visual_conflict_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_paths = [args.clean_guard_off, args.clean_protected]
    for perturbation, levels in GRID.items():
        for level_dir, _, _ in levels:
            for method in METHODS:
                source_paths.append(
                    args.run_root / "replay" / perturbation / level_dir / method / "metrics.json"
                )
    manifest = {
        "contract": "Development-only frozen visual-conflict sweep",
        "expected_frames": args.expected_frames,
        "guard": {"ema_alpha": 0.10, "ema_threshold": 0.50},
        "severity_grid": {
            key: [severity for _, severity, _ in values] for key, values in GRID.items()
        },
        "rows": len(rows),
        "sources": {str(path): _sha256(path) for path in source_paths},
        "output_sha256": _sha256(csv_path),
    }
    (args.output_dir / "visual_conflict_sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
