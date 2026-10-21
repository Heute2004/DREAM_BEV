#!/usr/bin/env python3
"""Build a RGB condition policy dataset from canonical sequence manifests.

The condition-policy model observes only an actual current RGB frame. Route-ledger routing
records are used only as frozen supervision targets during construction; they
never appear in the prompt, model input, or runtime policy parser.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPERT_ROUTES = (
    "corner_nlos",
    "indoor_clutter",
    "open_indoor_los",
    "open_outdoor_weak_return",
    "outdoor_dry_complex",
    "outdoor_wet_return",
    "unknown",
)
VISUAL_CONDITIONS = ("normal_visibility", "low_light", "rain_or_wet", "unknown")
VISUAL_RELIABILITY = ("usable", "cautious", "unreliable", "unknown")
WEATHER_SURFACE = ("indoor", "dry", "wet_return", "unknown")
ACTUAL_RGB_PATTERN = re.compile(r"^Data_train/seq_\d{3}/RGB/frame_\d{4}\.jpg$")
SEQUENCE_PATTERN = re.compile(r"^seq_\d{3}$")
SPLITS = ("Train", "Development Val", "Locked Test")
L_CORRIDOR_SEQUENCE_MAX = 21

PROMPT_VARIANTS = (
    """<image>
Use only the visible evidence in this current RGB frame. Select the final executable radar expert route and estimate the current visual condition. Do not use filenames, sequence IDs, scene ledgers, radar, depth, history, future frames, or external metadata. Return exactly one JSON object with exactly these keys: expert_route, expert_route_basis, visual_condition, visual_reliability_prior, weather_surface_prior, rationale_zh. expert_route must be one of corner_nlos, indoor_clutter, open_indoor_los, open_outdoor_weak_return, outdoor_dry_complex, outdoor_wet_return, unknown.""",
    """<image>
From this single current RGB image alone, choose one final radar expert action and describe visual usability. Do not infer from image paths, sequence identity, annotations, radar, depth, prior frames, future frames, or any external condition record. Output only one JSON object with the keys expert_route, expert_route_basis, visual_condition, visual_reliability_prior, weather_surface_prior, rationale_zh.""",
    """<image>
Independently assess the present RGB observation. Return one direct expert route rather than a coarse route, together with visual condition, visual reliability, and indoor/dry/wet-return context. Use no information beyond this image. Reply only with a JSON object containing expert_route, expert_route_basis, visual_condition, visual_reliability_prior, weather_surface_prior, rationale_zh.""",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_records(split: str, *, raw_frames: int) -> tuple[int, int]:
    """Return (actual RGB images at 2 Hz, annotation records)."""

    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if raw_frames % 5:
        raise ValueError(f"raw frame count is not 2-Hz divisible: {raw_frames}")
    images = raw_frames // 5
    return images, images * (len(PROMPT_VARIANTS) if split == "Train" else 1)


def contains_l_corridor(sequences: Iterable[str]) -> bool:
    """Return whether an audited split package contains excluded sequences."""

    for sequence in sequences:
        if not SEQUENCE_PATTERN.fullmatch(sequence):
            raise ValueError(f"invalid sequence name: {sequence}")
        if int(sequence.rsplit("_", 1)[1]) <= L_CORRIDOR_SEQUENCE_MAX:
            return True
    return False


def derive_visual_fields(route_row: dict[str, Any]) -> tuple[str, str, str]:
    """Create auxiliary condition supervision from frozen route-ledger labels only."""

    camera = str(route_row.get("source_camera_environment_zh", ""))
    modifiers = {str(value) for value in route_row.get("environment_modifiers", [])}
    if "indoor" in modifiers:
        weather_surface = "indoor"
    elif "wet_ground" in modifiers:
        weather_surface = "wet_return"
    elif "dry" in modifiers or "dry_ground" in modifiers:
        weather_surface = "dry"
    else:
        weather_surface = "unknown"

    if "夜间" in camera or "night" in modifiers:
        return "low_light", "unreliable", weather_surface
    if "低照度" in camera or "low_light" in modifiers:
        return "low_light", "cautious", weather_surface
    if "降雨" in camera or "雨后" in camera or weather_surface == "wet_return":
        return "rain_or_wet", "cautious", weather_surface
    return "normal_visibility", "usable", weather_surface


def validate_actual_rgb_path(relative_path: str) -> None:
    normalized = str(relative_path).replace("\\", "/")
    if not ACTUAL_RGB_PATTERN.fullmatch(normalized):
        raise ValueError(f"non-actual RGB image path rejected: {relative_path}")


def load_routing_labels(paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sequence = str(row["sequence_name"])
            split = str(row["split"])
            route = str(row["adapter_route"])
            if not SEQUENCE_PATTERN.fullmatch(sequence):
                raise ValueError(f"invalid sequence name in route-ledger: {sequence}")
            if split not in SPLITS or route not in EXPERT_ROUTES:
                raise ValueError(f"invalid route-ledger route record for {sequence}: {row}")
            visual_condition, reliability, weather_surface = derive_visual_fields(row)
            label = {
                "split": split,
                "expert_route": route,
                "visual_condition": visual_condition,
                "visual_reliability_prior": reliability,
                "weather_surface_prior": weather_surface,
            }
            prior = labels.setdefault(sequence, label)
            if prior != label:
                raise ValueError(f"conflicting route-ledger route labels for {sequence}: {prior} vs {label}")
    return labels


def load_frame_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def make_target(label: dict[str, str], expression_index: int) -> dict[str, str]:
    rationale_variants = (
        "当前左目图像的可见证据支持该结构化策略。",
        "该输出仅概括当前 RGB 观测对应的保守策略。",
        "该策略依据来自当前帧的视觉条件估计。",
    )
    route = label["expert_route"]
    return {
        "expert_route": route,
        "expert_route_basis": "unresolved" if route == "unknown" else "current_left_rgb_estimate",
        "visual_condition": label["visual_condition"],
        "visual_reliability_prior": label["visual_reliability_prior"],
        "weather_surface_prior": label["weather_surface_prior"],
        "rationale_zh": rationale_variants[expression_index % len(rationale_variants)],
    }


def make_record(
    *,
    split: str,
    sequence: str,
    frame_id: int,
    image_path: str,
    label: dict[str, str],
    expression_index: int,
) -> dict[str, Any]:
    suffix = f"__expr{expression_index + 1}" if split == "Train" else ""
    return {
        "id": f"condition_policy_{split.lower().replace(' ', '_')}_{sequence}_frame_{frame_id:04d}{suffix}",
        "image": image_path,
        "conversations": [
            {"from": "human", "value": PROMPT_VARIANTS[expression_index]},
            {
                "from": "gpt",
                "value": json.dumps(make_target(label, expression_index), ensure_ascii=False, separators=(",", ":")),
            },
        ],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full Sequence-Locked RGB condition policy annotations.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frame-manifest", type=Path, action="append", required=True)
    parser.add_argument("--routing-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite condition-policy dataset: {args.output_root}")
    labels = load_routing_labels(args.routing_manifest)
    frame_rows = load_frame_rows(args.frame_manifest)
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    images: dict[str, set[str]] = defaultdict(set)
    sequences: dict[str, set[str]] = defaultdict(set)
    frame_coverage: dict[str, set[int]] = defaultdict(set)
    sequence_split: dict[str, str] = {}

    for row in frame_rows:
        split = str(row["split"]).strip()
        sequence = str(row["sequence"]).strip()
        if split not in SPLITS or not SEQUENCE_PATTERN.fullmatch(sequence):
            raise ValueError(f"invalid frame-contract frame record: {row}")
        prior_split = sequence_split.setdefault(sequence, split)
        if prior_split != split:
            raise ValueError(f"split conflict for {sequence}: {prior_split} vs {split}")
        if sequence not in labels:
            raise ValueError(f"missing frozen route-ledger supervision for {sequence}")
        if labels[sequence]["split"] != split:
            raise ValueError(f"frame-contract/route-ledger split mismatch for {sequence}")
        frame_id = int(row["frame_id"])
        frame_coverage[sequence].add(frame_id)
        if frame_id % 5:
            continue
        normalized_rgb_path = str(row["rgb_path"]).replace("\\", "/")
        image_path = f"Data_train/{normalized_rgb_path}"
        validate_actual_rgb_path(image_path)
        if not (args.data_root / image_path).is_file():
            raise FileNotFoundError(f"missing actual RGB frame: {args.data_root / image_path}")
        if image_path in images[split]:
            raise ValueError(f"duplicate condition-policy image in {split}: {image_path}")
        images[split].add(image_path)
        sequences[split].add(sequence)
        expression_count = len(PROMPT_VARIANTS) if split == "Train" else 1
        for expression_index in range(expression_count):
            records[split].append(
                make_record(
                    split=split,
                    sequence=sequence,
                    frame_id=frame_id,
                    image_path=image_path,
                    label=labels[sequence],
                    expression_index=expression_index,
                )
            )

    if set(sequence_split) != set(labels):
        missing_labels = sorted(set(sequence_split) - set(labels))
        unused_labels = sorted(set(labels) - set(sequence_split))
        raise ValueError(f"frame-contract/route-ledger sequence mismatch missing={missing_labels} unused={unused_labels}")
    for sequence, frames in frame_coverage.items():
        if frames != set(range(300)):
            raise ValueError(f"incomplete frame-contract frame coverage for {sequence}: {len(frames)}")
    if any(sequences[left] & sequences[right] for left in SPLITS for right in SPLITS if left != right):
        raise RuntimeError("sequence leakage across Sequence-Locked splits")
    if any(images[left] & images[right] for left in SPLITS for right in SPLITS if left != right):
        raise RuntimeError("image leakage across Sequence-Locked splits")

    args.output_root.mkdir(parents=True)
    annotation_dir = args.output_root / "annotations"
    annotation_dir.mkdir()
    filenames = {
        "Train": f"train_{len(records['Train'])}_3expr_condition_policy_actualrgb.jsonl",
        "Development Val": f"development_{len(records['Development Val'])}_condition_policy_actualrgb.jsonl",
        "Locked Test": f"locked_test_{len(records['Locked Test'])}_condition_policy_actualrgb_READ_ONLY.jsonl",
    }
    paths: dict[str, Path] = {}
    for split, filename in filenames.items():
        path = annotation_dir / filename
        write_jsonl(path, records[split])
        paths[split] = path
    # The training bootstrap never references this file. It is immutable before final evaluation.
    paths["Locked Test"].chmod(0o444)

    audit = {
        "dataset": "RGB condition policy actual-RGB sequence-isolated package",
        "model_input": "current_left_rgb_only",
        "routing_supervision": "frozen_route-ledger_adapter_route_only",
        "runtime_route_composition": "forbidden_direct_expert_route_only",
        "frame_sample_every": 5,
        "locked_test_training_access": False,
        "input_frame_manifests_sha256": {str(path): sha256_file(path) for path in args.frame_manifest},
        "routing_manifests_sha256": {str(path): sha256_file(path) for path in args.routing_manifest},
        "splits": {
            split: {
                "records": len(records[split]),
                "unique_actual_rgb_images": len(images[split]),
                "sequences": sorted(sequences[split]),
                "route_counts": dict(sorted(Counter(labels[sequence]["expert_route"] for sequence in sequences[split]).items())),
                "annotation": paths[split].name,
                "sha256": sha256_file(paths[split]),
            }
            for split in SPLITS
        },
        "cross_split_sequence_overlap": 0,
        "cross_split_image_overlap": 0,
        "representative_site_images": 0,
        "l_corridor_included": contains_l_corridor(sequence_split),
    }
    (args.output_root / "condition_policy_dataset_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



