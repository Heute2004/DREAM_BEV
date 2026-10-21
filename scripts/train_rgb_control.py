#!/usr/bin/env python3
"""Train a current-RGB-only official CNN control under the RGB condition policy contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet152_Weights,
    mobilenet_v3_small,
    resnet152,
)
from torchvision.transforms import v2

try:
    from condition_policy_contract import EXPERT_ROUTES, TASK_FIELDS, validate_prediction
except ModuleNotFoundError:  # Supports both direct execution and unit-test imports.
    from server.condition_policy_contract import EXPERT_ROUTES, TASK_FIELDS, validate_prediction


TASK_VALUES: dict[str, tuple[str, ...]] = {
    "expert_route": tuple(sorted(EXPERT_ROUTES)),
    "visual_condition": ("normal_visibility", "low_light", "rain_or_wet", "unknown"),
    "visual_reliability_prior": ("usable", "cautious", "unreliable", "unknown"),
    "weather_surface_prior": ("indoor", "dry", "wet_return", "unknown"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_train_or_dev_annotation(annotation: Path) -> None:
    if "locked_test" in annotation.name.lower():
        raise ValueError("Locked Test annotations are forbidden for condition-policy non-VLM control training.")


def require_fresh_control_output(output_dir: Path) -> None:
    """Allow only launcher-created logs before a new control training run."""

    if not output_dir.exists():
        return
    unexpected = [entry.name for entry in output_dir.iterdir() if entry.name != "logs"]
    if unexpected:
        raise FileExistsError(f"refusing to overwrite condition-policy control run: {output_dir}")


def prepare_control_output_dir(output_dir: Path) -> None:
    """Validate an output location, then create it without rejecting launcher logs."""

    require_fresh_control_output(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    require_train_or_dev_annotation(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _target_object(record: dict[str, Any]) -> dict[str, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"invalid condition-policy conversations for {record.get('id')}")
    try:
        target = json.loads(conversations[1]["value"])
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid condition-policy target for {record.get('id')}") from error
    if not isinstance(target, dict) or not validate_prediction(target):
        raise ValueError(f"target does not satisfy condition-policy contract for {record.get('id')}")
    return target


def canonical_target(record: dict[str, Any]) -> dict[str, str]:
    target = _target_object(record)
    return {field: str(target[field]) for field in TASK_FIELDS}


def unique_records_by_image(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    signatures: dict[str, tuple[str, ...]] = {}
    for record in records:
        image = str(record.get("image", "")).replace("\\", "/")
        if not image:
            raise ValueError(f"record without image: {record.get('id')}")
        target = canonical_target(record)
        signature = tuple(target[field] for field in TASK_FIELDS)
        if image in signatures and signatures[image] != signature:
            raise ValueError(f"conflicting condition-policy targets for image {image}")
        signatures[image] = signature
        unique.setdefault(image, record)
    return [unique[image] for image in sorted(unique)]


@dataclass(frozen=True)
class EncodedRecord:
    identifier: str
    image: str
    labels: tuple[int, ...]


def encode_records(records: Iterable[dict[str, Any]]) -> list[EncodedRecord]:
    value_to_index = {
        field: {value: index for index, value in enumerate(values)}
        for field, values in TASK_VALUES.items()
    }
    result: list[EncodedRecord] = []
    for record in records:
        target = canonical_target(record)
        result.append(
            EncodedRecord(
                identifier=str(record["id"]),
                image=str(record["image"]).replace("\\", "/"),
                labels=tuple(value_to_index[field][target[field]] for field in TASK_FIELDS),
            )
        )
    return result


class RGBRouteDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, root: Path, records: list[EncodedRecord], transform: v2.Compose) -> None:
        self.root = root
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.records[index]
        with Image.open(self.root / item.image) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb), torch.tensor(item.labels, dtype=torch.long)


class RGBRouteControl(nn.Module):
    def __init__(self, backbone_name: str, *, pretrained: bool) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "resnet152":
            weights = ResNet152_Weights.IMAGENET1K_canonical if pretrained else None
            backbone = resnet152(weights=weights)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_canonical if pretrained else None
            backbone = mobilenet_v3_small(weights=weights)
            feature_dim = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"unsupported backbone: {backbone_name}")
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {field: nn.Linear(feature_dim, len(values)) for field, values in TASK_VALUES.items()}
        )

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.backbone(images)
        return {field: head(features) for field, head in self.heads.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1(prediction: Tensor, target: Tensor, class_count: int) -> float:
    scores: list[float] = []
    for class_index in range(class_count):
        if not bool((prediction == class_index).any() or (target == class_index).any()):
            continue
        tp = int(((prediction == class_index) & (target == class_index)).sum())
        fp = int(((prediction == class_index) & (target != class_index)).sum())
        fn = int(((prediction != class_index) & (target == class_index)).sum())
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(sum(scores) / len(scores))


@torch.inference_mode()
def evaluate_router(
    model: RGBRouteControl,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predicted: dict[str, list[Tensor]] = defaultdict(list)
    targets: dict[str, list[Tensor]] = defaultdict(list)
    for images, label_tensor in loader:
        output = model(images.to(device, non_blocking=True))
        for field_index, field in enumerate(TASK_FIELDS):
            predicted[field].append(output[field].argmax(dim=1).cpu())
            targets[field].append(label_tensor[:, field_index].cpu())
    metrics: dict[str, float] = {}
    macro_scores: list[float] = []
    for field in TASK_FIELDS:
        prediction = torch.cat(predicted[field])
        target = torch.cat(targets[field])
        metrics[f"{field}_accuracy"] = float((prediction == target).float().mean())
        score = macro_f1(prediction, target, len(TASK_VALUES[field]))
        metrics[f"{field}_macro_f1"] = score
        macro_scores.append(score)
    metrics["mean_four_task_macro_f1"] = float(sum(macro_scores) / len(macro_scores))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("resnet152", "mobilenet_v3_small"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-annotation", type=Path, required=True)
    parser.add_argument("--dev-annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_control_output_dir(args.output_dir)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    require_train_or_dev_annotation(args.train_annotation)
    require_train_or_dev_annotation(args.dev_annotation)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_rows = unique_records_by_image(read_jsonl(args.train_annotation))
    dev_rows = unique_records_by_image(read_jsonl(args.dev_annotation))
    train_encoded = encode_records(train_rows)
    dev_encoded = encode_records(dev_rows)
    if not train_encoded or not dev_encoded:
        raise ValueError("empty train or development data")

    if args.backbone == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_canonical
    else:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_canonical
    normalize = weights.transforms()
    train_transform = v2.Compose(
        [
            v2.Resize(256),
            v2.RandomCrop(224),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=normalize.mean, std=normalize.std),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=normalize.mean, std=normalize.std),
        ]
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        RGBRouteDataset(args.data_root, train_encoded, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_options,
    )
    dev_loader = DataLoader(
        RGBRouteDataset(args.data_root, dev_encoded, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    model = RGBRouteControl(args.backbone, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.heads.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_score = -1.0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, label_tensor in train_loader:
            output = model(images.to(device, non_blocking=True))
            labels = label_tensor.to(device, non_blocking=True)
            loss = sum(
                nn.functional.cross_entropy(output[field], labels[:, field_index])
                for field_index, field in enumerate(TASK_FIELDS)
            ) / len(TASK_FIELDS)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = evaluate_router(model, dev_loader, device)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "backbone_lr": float(optimizer.param_groups[0]["lr"]),
            **metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["mean_four_task_macro_f1"] > best_score:
            best_score = metrics["mean_four_task_macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "model": f"torchvision_{args.backbone}_multitask_condition_policy",
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "metrics": metrics,
                    "task_values": TASK_VALUES,
                    "input_contract": "current_left_rgb_only",
                    "train_annotation_sha256": sha256_file(args.train_annotation),
                    "dev_annotation_sha256": sha256_file(args.dev_annotation),
                },
                args.output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        if stale_epochs >= args.early_stop_patience:
            break

    payload = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    report = {
        "protocol": "condition_policy_current_rgb_nonvlm_control",
        "model": f"Torchvision {args.backbone} multi-task router",
        "input_contract": "current_left_rgb_only",
        "uses_vlm": False,
        "uses_radar_depth_metadata_or_future": False,
        "selection_split": "Development Val",
        "locked_test_access": False,
        "train_annotation": str(args.train_annotation),
        "dev_annotation": str(args.dev_annotation),
        "unique_train_images": len(train_rows),
        "unique_dev_images": len(dev_rows),
        "best_epoch": int(payload["epoch"]),
        "development_metrics": payload["metrics"],
        "checkpoint_sha256": sha256_file(args.output_dir / "best.pt"),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()



