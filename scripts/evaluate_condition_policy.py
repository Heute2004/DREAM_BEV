#!/usr/bin/env python3
"""Evaluate the RGB-only condition policy without opening Locked Test by default."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from condition_policy.contract import REQUIRED_FIELDS, TASK_FIELDS, validate_prediction


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse a single JSON object, allowing incidental text around it."""

    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def require_safe_annotation(annotation: Path, allow_locked_test: bool = False) -> None:
    """Prevent accidental Locked Test access unless the caller explicitly opts in."""

    if "locked_test" in annotation.name.lower() and not allow_locked_test:
        raise ValueError(
            "Locked Test annotation is sealed by default. "
            "Pass --allow-locked-test only after the protocol is frozen."
        )


def load_records(annotation: Path) -> list[dict[str, Any]]:
    with annotation.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target_json(record: dict[str, Any]) -> dict[str, Any]:
    value = extract_json(record["conversations"][1]["value"])
    if value is None:
        raise ValueError(f"invalid frozen target for {record['id']}")
    return value


def _field_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    labels: set[str] = set()
    for row in rows:
        target = str(row["target"][field])
        prediction = row.get("prediction")
        predicted_label = str(prediction.get(field, "<invalid>")) if prediction else "<invalid>"
        labels.add(target)
        labels.add(predicted_label)
        matrix[target][predicted_label] += 1

    labels.discard("<invalid>")
    ordered_labels = sorted(labels)
    correct = sum(matrix[label][label] for label in ordered_labels)
    per_label: dict[str, float] = {}
    for label in ordered_labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in matrix if other != label)
        fn = sum(count for predicted, count in matrix[label].items() if predicted != label)
        denominator = 2 * tp + fp + fn
        per_label[label] = 2 * tp / denominator if denominator else 0.0

    return {
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": sum(per_label.values()) / len(per_label) if per_label else 0.0,
        "per_label_f1": per_label,
        "confusion": {label: dict(matrix[label]) for label in sorted(matrix)},
    }


def render_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Render condition-policy schema and direct-route metrics from evaluation rows."""

    parsed = [row for row in rows if isinstance(row.get("prediction"), dict)]
    schema_valid = sum(validate_prediction(row["prediction"]) for row in parsed)
    metrics: dict[str, Any] = {
        "records": len(rows),
        "valid_json": len(parsed),
        "valid_json_rate": len(parsed) / len(rows) if rows else 0.0,
        "schema_valid": schema_valid,
        "schema_valid_rate": schema_valid / len(rows) if rows else 0.0,
        "required_fields_rate": (
            sum(set(row["prediction"]) == REQUIRED_FIELDS for row in parsed) / len(rows)
            if rows
            else 0.0
        ),
    }
    for field in TASK_FIELDS:
        metrics[field] = _field_metrics(rows, field)
    metrics["four_field_exact_match"] = (
        sum(
            row.get("prediction") is not None
            and all(row["prediction"].get(field) == row["target"][field] for field in TASK_FIELDS)
            for row in rows
        )
        / len(rows)
        if rows
        else 0.0
    )
    return metrics


def load_model(model_path: Path) -> tuple[Any, Any]:
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    from internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    config = InternVLChatConfig.from_pretrained(model_path)
    model = InternVLChatModel(config)
    state_dict = load_file(model_path / "model.safetensors", device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:8]} unexpected={incompatible.unexpected_keys[:8]}"
        )
    return model.to(dtype=torch.bfloat16, device="cuda").eval(), tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-locked-test", action="store_true")
    args = parser.parse_args()

    annotation = args.annotation.resolve()
    require_safe_annotation(annotation, allow_locked_test=args.allow_locked_test)
    records = load_records(annotation)
    if args.limit:
        records = records[: args.limit]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    completed: set[str] = set()
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            completed = {json.loads(line)["id"] for line in handle if line.strip()}

    model, tokenizer = load_model(args.model_path.resolve())
    from internvl.train.dataset import build_transform, dynamic_preprocess

    transform = build_transform(
        is_train=False,
        input_size=model.config.force_image_size,
        pad2square=model.config.pad2square,
        normalize_type="imagenet",
    )
    generation_config = {"num_beams": 1, "max_new_tokens": 256, "do_sample": False}
    with predictions_path.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            if record["id"] in completed:
                continue
            image = Image.open(args.data_root.resolve() / record["image"]).convert("RGB")
            tiles = dynamic_preprocess(
                image,
                image_size=model.config.force_image_size,
                use_thumbnail=model.config.use_thumbnail,
                max_num=model.config.max_dynamic_patch,
            )
            pixel_values = torch.stack([transform(tile) for tile in tiles]).to(torch.bfloat16).cuda()
            with torch.inference_mode():
                response = model.chat(
                    tokenizer,
                    pixel_values,
                    record["conversations"][0]["value"],
                    generation_config,
                )
            row = {
                "id": record["id"],
                "image": record["image"],
                "target": target_json(record),
                "prediction": extract_json(response),
                "raw_response": response,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(f"evaluated={index}/{len(records)}", flush=True)

    with predictions_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    metrics = render_metrics(rows)
    metrics.update(
        {
            "annotation": str(annotation),
            "model_path": str(args.model_path.resolve()),
            "output_records": len(rows),
            "locked_test_access": bool(args.allow_locked_test),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



