#!/usr/bin/env python3
"""Evaluate frozen paper-contract CRUW RVFusion under a causal consistency guard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cruw_pedestrian_fusion.allscene import CRUWBEVGeometry, build_all_scene_index
from cruw_pedestrian_fusion.canonical_dream_bev import DREAMBEV
from cruw_pedestrian_fusion.causal_consistency import CausalLateralConsistencyGuard
from cruw_pedestrian_fusion.geometry import load_cruw_grids
from cruw_pedestrian_fusion.metrics import localization_metrics
from cruw_pedestrian_fusion.protocol import load_protocol
from cruw_pedestrian_fusion.rv_bearing_net.model import RVBearingNet
from cruw_pedestrian_fusion.rvfusion_decode import decode_admitted_posterior_mean

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train_cruw_rvfusion import (  # noqa: E402
    _AlignedFusionDataset,
    _build_expert,
    _build_fusion,
    _expert_output_xy,
    _freeze,
    _run_expert,
    _run_fusion,
    _run_visual,
    fixed_evidence,
    select_single_person_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cruw-devkit-root", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-kind", required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--ema-alpha", type=float, default=0.10)
    parser.add_argument("--ema-threshold", type=float, default=0.80)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-audit-csv", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_runtime_decoder_bias(fusion: torch.nn.Module) -> None:
    """Keep Development decoder calibration outside the frozen fusion body."""

    with torch.no_grad():
        fusion.nonidentity_bias.zero_()


def local_radar_sigma_x(
    logits: torch.Tensor,
    expert_xy_m: torch.Tensor,
    geometry: CRUWBEVGeometry,
    *,
    radius_m: float = 0.75,
) -> torch.Tensor:
    x_coordinates = (
        geometry.x_min_m
        + torch.arange(geometry.x_bins, device=logits.device, dtype=logits.dtype) * geometry.resolution_m
    )
    y_index = ((expert_xy_m[:, 1] - geometry.y_min_m) / geometry.resolution_m).round().long()
    y_index = y_index.clamp(0, geometry.y_bins - 1)
    row_logits = logits[:, 0].permute(0, 2, 1).gather(
        1,
        y_index[:, None, None].expand(-1, 1, geometry.x_bins),
    ).squeeze(1)
    local_weight = torch.sigmoid(row_logits) * (
        (x_coordinates[None] - expert_xy_m[:, :1]).abs() <= radius_m
    ).to(row_logits.dtype)
    local_weight = local_weight / local_weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mean_x = (local_weight * x_coordinates[None]).sum(dim=1, keepdim=True)
    variance = (local_weight * (x_coordinates[None] - mean_x).square()).sum(dim=1)
    return variance.sqrt().clamp(0.05, 0.50)


def guarded_lateral_replay(
    values: dict[str, np.ndarray],
    *,
    ema_alpha: float,
    ema_threshold: float,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    count = values["expert_x"].size
    guarded_x = values["expert_x"].copy()
    order = np.lexsort((values["frame_id"], values["sequence_id"]))
    guard = CausalLateralConsistencyGuard(ema_alpha=ema_alpha, ema_threshold=ema_threshold)
    audit: list[dict[str, float | int | bool]] = []
    for index in order:
        expert_x = float(values["expert_x"][index])
        visual_x = float(values["visual_x"][index])
        sigma = float(np.hypot(values["visual_sigma_x"][index], values["radar_sigma_x"][index]))
        innovation = abs(visual_x - expert_x) / max(sigma, 1e-6)
        signed_innovation = innovation if visual_x >= expert_x else -innovation
        eligible = bool(values["admitted"][index])
        accepted, ema = guard.step(
            sequence_id=int(values["sequence_id"][index]),
            frame_id=int(values["frame_id"][index]),
            signed_innovation=signed_innovation,
            eligible=eligible,
        )
        if accepted:
            guarded_x[index] = values["fused_x"][index]
        audit.append(
            {
                "sequence_id": int(values["sequence_id"][index]),
                "frame_id": int(values["frame_id"][index]),
                "expert_x_m": expert_x,
                "candidate_x_m": float(values["fused_x"][index]),
                "guarded_x_m": float(guarded_x[index]),
                "target_x_m": float(values["target_x"][index]),
                "target_y_m": float(values["target_y"][index]),
                "visual_x_m": visual_x,
                "visual_sigma_x_m": float(values["visual_sigma_x"][index]),
                "radar_sigma_x_m": float(values["radar_sigma_x"][index]),
                "cross_modal_innovation": innovation,
                "innovation_ema": ema,
                "posterior_admitted": eligible,
                "visual_accepted": accepted,
            }
        )
    if guarded_x.shape != (count,):
        raise RuntimeError("guarded output shape mismatch")
    return guarded_x, audit


@torch.no_grad()
def collect_frozen_outputs(args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = torch.device("cuda")
    geometry = CRUWBEVGeometry()
    protocol = load_protocol(args.protocol)
    partition = json.loads(args.protocol.read_text(encoding="utf-8"))
    development_sequences = {int(value) for value in partition["single_person_validation_sequences"]}
    range_grid, angle_grid = load_cruw_grids(args.data_root, args.cruw_devkit_root)
    records = select_single_person_records(
        build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride),
        development_sequences,
        geometry,
    )
    dataset = _AlignedFusionDataset(
        records,
        args.cache_root,
        range_grid,
        angle_grid,
        args.clip_length,
        args.image_size,
        geometry,
        False,
        "none",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    anchor_payload = torch.load(args.anchor_checkpoint, map_location="cpu", weights_only=False)
    anchor = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    anchor.load_state_dict(anchor_payload["model"], strict=True)
    expert = _build_expert(args.expert_kind, range_grid, angle_grid, geometry).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    visual = RVBearingNet().to(device)
    visual.load_state_dict(torch.load(args.visual_checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    fusion = _build_fusion("range-preserving", range_grid, angle_grid, geometry).to(device)
    fusion_payload = torch.load(args.fusion_checkpoint, map_location="cpu", weights_only=False)
    fusion.load_state_dict(fusion_payload["model"], strict=True)
    clear_runtime_decoder_bias(fusion)
    for model in (anchor, expert, visual, fusion):
        _freeze(model)

    keys = (
        "expert_x", "expert_y", "candidate_xy", "candidate_logits", "visual_x", "visual_sigma_x",
        "radar_sigma_x", "target_x", "target_y", "score", "sequence_id", "frame_id",
    )
    parts = {key: [] for key in keys}
    for batch in loader:
        radar, current, expert_output, visual_output, score = fixed_evidence(anchor, expert, visual, batch, device)
        output = _run_fusion(fusion, radar, current, expert, expert_output, visual_output, score, geometry)
        expert_xy = _expert_output_xy(expert, expert_output, geometry)
        visual_theta = visual_output.theta[:, 0]
        visual_sigma_x = (
            expert_xy[:, 1] * visual_output.sigma_theta[:, 0]
            / torch.cos(visual_theta).abs().clamp_min(0.2).square()
            + 0.05
        ).clamp(0.05, 0.75)
        values = {
            "expert_x": expert_xy[:, 0].cpu().numpy(),
            "expert_y": expert_xy[:, 1].cpu().numpy(),
            "candidate_xy": output.candidate_xy_m.cpu().numpy(),
            "candidate_logits": output.candidate_logits.cpu().numpy(),
            "visual_x": (expert_xy[:, 1] * torch.tan(visual_theta)).cpu().numpy(),
            "visual_sigma_x": visual_sigma_x.cpu().numpy(),
            "radar_sigma_x": local_radar_sigma_x(current.logits, expert_xy, geometry).cpu().numpy(),
            "target_x": batch["single_x_m"].numpy(),
            "target_y": batch["single_y_m"].numpy(),
            "score": score.cpu().numpy(),
            "sequence_id": batch["sequence_id"].numpy(),
            "frame_id": batch["frame_id"].numpy(),
        }
        for key in keys:
            parts[key].append(values[key])
    merged = {key: np.concatenate(value) for key, value in parts.items()}
    fused_xy, admitted = decode_admitted_posterior_mean(
        merged["candidate_xy"],
        merged["candidate_logits"],
        nonidentity_bias=float(fusion_payload["development_selected_nonidentity_bias"]),
        temperature=float(fusion_payload["development_selected_temperature"]),
        expectation_noop_logit_offset=float(fusion_payload["development_selected_expectation_noop_logit_offset"]),
    )
    merged["fused_x"] = fused_xy[:, 0]
    merged["fused_y"] = fused_xy[:, 1]
    merged["admitted"] = admitted
    return merged


def main() -> None:
    args = parse_args()
    values = collect_frozen_outputs(args)
    guarded_x, audit = guarded_lateral_replay(
        values,
        ema_alpha=args.ema_alpha,
        ema_threshold=args.ema_threshold,
    )
    expert = localization_metrics(values["expert_x"], values["expert_y"], values["target_x"], values["target_y"], values["score"])
    unguarded = localization_metrics(values["fused_x"], values["fused_y"], values["target_x"], values["target_y"], values["score"])
    guarded = localization_metrics(guarded_x, values["expert_y"], values["target_x"], values["target_y"], values["score"])
    expert_error = np.hypot(values["expert_x"] - values["target_x"], values["expert_y"] - values["target_y"])
    guarded_error = np.hypot(guarded_x - values["target_x"], values["expert_y"] - values["target_y"])
    result = {
        "contract": "frozen paper-contract RVFusion with causal cross-modal consistency guard; immutable expert range; no retraining; no official CRUW test",
        "protocol_split": "development",
        "official_test_read": False,
        "frame_count": int(values["expert_x"].size),
        "expert_radar": expert,
        "rvfusion_unguarded": unguarded,
        "rvfusion_guarded": guarded,
        "delta_guarded_vs_expert_AP@0.30_pp": 100.0 * (guarded["AP@0.30"] - expert["AP@0.30"]),
        "guard": {
            "ema_alpha": args.ema_alpha,
            "ema_threshold": args.ema_threshold,
            "posterior_admission_rate": float(values["admitted"].mean()),
            "visual_acceptance_rate": float(np.mean(guarded_x != values["expert_x"])),
            "identity_fallback_rate": float(np.mean(guarded_x == values["expert_x"])),
            "improved_fraction": float(np.mean(guarded_error + 1e-6 < expert_error)),
            "harmed_fraction": float(np.mean(guarded_error > expert_error + 1e-6)),
        },
        "input_sha256": {
            "anchor": sha256(args.anchor_checkpoint),
            "expert": sha256(args.expert_checkpoint),
            "visual": sha256(args.visual_checkpoint),
            "fusion": sha256(args.fusion_checkpoint),
            "protocol": sha256(args.protocol),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.output_audit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
