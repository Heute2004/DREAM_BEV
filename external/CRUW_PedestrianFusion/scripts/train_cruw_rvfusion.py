#!/usr/bin/env python3
"""Train range-preserving RVFusion on frozen CRUW DREAM-BEV components."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from cruw_pedestrian_fusion.allscene import CRUWBEVGeometry, RadarBEVWindowDataset, build_all_scene_index
from cruw_pedestrian_fusion.allscene_visual import AllSceneRGBDataset
from cruw_pedestrian_fusion.canonical_dream_bev import DREAMBEV
from cruw_pedestrian_fusion.cruw_expert import CRUWPBMHBRExpert
from cruw_pedestrian_fusion.global_association_expert import CRUWGlobalAssociationExpert
from cruw_pedestrian_fusion.paper_experts import (
    PAPER_EXPERT_KINDS,
    CRUWPaperExpert,
    decode_paper_expert_output_xy,
)
from cruw_pedestrian_fusion.cruw_rvfusion import (
    CRUWRangePreservingRVFusion,
    CRUWXYRadarCandidateRVFusion,
    ap30_recoverable_transition_mask,
    ap30_transition_soft_targets,
    ap30_transition_utility_targets,
    lateral_utility_targets,
    reachable_lateral_utility_targets,
    soft_lateral_utility_targets,
    xy_utility_targets,
)
from cruw_pedestrian_fusion.geometry import load_cruw_grids
from cruw_pedestrian_fusion.metrics import localization_metrics
from cruw_pedestrian_fusion.protocol import load_protocol
from cruw_pedestrian_fusion.rv_bearing_net.model import RVBearingNet
from cruw_pedestrian_fusion.rvfusion_decode import decode_admitted_posterior_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cruw-devkit-root", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expert-kind",
        choices=("pbmhbr", "global-association", *PAPER_EXPERT_KINDS),
        default="pbmhbr",
    )
    parser.add_argument(
        "--fusion-kind",
        choices=("range-preserving", "xy-radar-candidate"),
        default="range-preserving",
    )
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-kind", choices=("rv-bearing", "causal-vast"), default="rv-bearing")
    parser.add_argument("--visual-manifest", type=Path)
    parser.add_argument("--proposal-cache", type=Path)
    parser.add_argument("--roi-feature-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument(
        "--fusion-augmentation-mode",
        choices=("independent", "clip_consistent", "none"),
        default="independent",
        help="RGB augmentation used only while training the frozen visual-evidence fusion head.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-improvement-m", type=float, default=0.05)
    parser.add_argument(
        "--utility-target-mode",
        choices=("lateral", "lateral_soft", "ap30_reachable", "ap30_transition", "ap30_transition_soft"),
        default="lateral",
        help="Frozen-candidate supervision: hard/soft lateral, reachable, or AP@0.30 transition utility.",
    )
    parser.add_argument(
        "--utility-target-radius-m",
        type=float,
        default=0.30,
        help="Target-radius requirement for --utility-target-mode ap30_reachable.",
    )
    parser.add_argument(
        "--soft-target-sigma-m",
        type=float,
        default=0.10,
        help="Lateral Gaussian width for --utility-target-mode lateral_soft.",
    )
    parser.add_argument(
        "--ap30-transition-loss-weight",
        type=float,
        default=1.0,
        help="Extra supervised-loss weight for AP@0.30 miss-to-hit candidate frames.",
    )
    parser.add_argument(
        "--identity-target-weight",
        type=float,
        default=0.35,
        help="Relative cross-entropy weight of exact radar-fallback labels during fusion-head training.",
    )
    parser.add_argument(
        "--posterior-regression-weight",
        type=float,
        default=0.20,
        help="Auxiliary Smooth-L1 weight on the posterior lateral expectation.",
    )
    parser.add_argument(
        "--posterior-regression-beta",
        type=float,
        default=0.10,
        help="Smooth-L1 transition width in metres for the posterior auxiliary loss.",
    )
    parser.add_argument(
        "--selection-decoder",
        choices=("map", "admitted_posterior_mean"),
        default="map",
        help="Development-only checkpoint selection readout; does not alter the RVFusion network.",
    )
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shifted_previous(radar: torch.Tensor) -> torch.Tensor:
    return torch.cat((radar[:, :1], radar[:, :-1]), dim=1)


def select_single_person_records(records, allowed: set[int], geometry: CRUWBEVGeometry):
    return [
        record
        for record in records
        if record.sequence_id in allowed
        and sum(geometry.contains(person.x_m, person.y_m) for person in record.pedestrians) == 1
    ]


class _AlignedFusionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records,
        cache_root: Path,
        range_grid: np.ndarray,
        angle_grid: np.ndarray,
        clip_length: int,
        image_size: int,
        geometry: CRUWBEVGeometry,
        training: bool,
        augmentation_mode: str = "independent",
    ) -> None:
        self.radar = RadarBEVWindowDataset(records, cache_root, range_grid, angle_grid, clip_length, geometry)
        self.visual = AllSceneRGBDataset(
            records,
            clip_length,
            image_size,
            training=training,
            augmentation_mode=augmentation_mode,
            geometry=geometry,
        )

    def __len__(self) -> int:
        return len(self.radar)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        radar = self.radar[index]
        visual = self.visual[index]
        if not torch.equal(radar["sequence_id"], visual["sequence_id"]) or not torch.equal(
            radar["frame_id"], visual["frame_id"]
        ):
            raise RuntimeError("radar/RGB record alignment failure")
        radar["images"] = visual["images"]
        radar["frame_valid"] = visual["frame_valid"]
        return radar


class _AlignedVASTFusionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records, cache_root, range_grid, angle_grid, clip_length, geometry, vast_dataset) -> None:
        self.radar = RadarBEVWindowDataset(records, cache_root, range_grid, angle_grid, clip_length, geometry)
        self.vast = vast_dataset
        if len(self.radar) != len(self.vast):
            raise RuntimeError("radar/VAST dataset length mismatch")

    def __len__(self) -> int:
        return len(self.radar)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        radar = self.radar[index]
        vast = self.vast[index]
        sequence_id = int(str(vast["sequence"]).rsplit("_", maxsplit=1)[-1])
        if int(radar["sequence_id"]) != sequence_id or int(radar["frame_id"]) != int(vast["frame_id"]):
            raise RuntimeError("radar/VAST record alignment failure")
        radar["proposals"] = vast["proposals"]
        radar["roi_features"] = vast["roi_features"]
        return radar


def _loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def _build_expert(kind, range_grid, angle_grid, geometry):
    if kind == "pbmhbr":
        return CRUWPBMHBRExpert(range_grid, angle_grid, geometry)
    if kind == "global-association":
        return CRUWGlobalAssociationExpert(range_grid, angle_grid, geometry)
    if kind in PAPER_EXPERT_KINDS:
        return CRUWPaperExpert(kind, geometry)
    raise ValueError(f"unknown expert kind: {kind}")


def _run_expert(expert, radar, current, previous, visual_output):
    arguments = {
        "polar_feature": current.polar_feature,
        "shared_feature": current.feature,
        "radar": radar,
        "base_logits": current.logits,
        "previous_logits": previous.logits,
    }
    if isinstance(expert, CRUWGlobalAssociationExpert):
        arguments.update(
            {
                "visual_theta": visual_output.theta[:, 0],
                "visual_sigma": visual_output.sigma_theta[:, 0],
                "visual_reliability": visual_output.reliability[:, 0],
                "visual_feature": visual_output.temporal_feature,
            }
        )
    return expert(**arguments)


def _expert_output_xy(expert, expert_output, geometry):
    """Return the exact expert-radar output consumed by the fusion contract."""

    if isinstance(expert, CRUWPaperExpert):
        return decode_paper_expert_output_xy(expert_output, geometry)
    return expert_output["proposal_xy_m"]


def _run_visual(visual, batch, device):
    if "proposals" not in batch:
        return visual(
            batch["images"].to(device, non_blocking=True),
            frame_valid=batch["frame_valid"].to(device, non_blocking=True),
        )
    output = visual(
        batch["proposals"].to(device, non_blocking=True),
        batch["roi_features"].to(device, non_blocking=True),
    )
    posterior = output["selection_logits"].softmax(dim=1)
    presence = 1.0 - posterior[:, -1:]
    return SimpleNamespace(
        theta=output["camera_theta"],
        sigma_theta=output["bearing_sigma"],
        reliability=output["temporal_quality"] * presence,
        temporal_feature=output["temporal_feature"],
    )


def _build_fusion(
    kind,
    range_grid,
    angle_grid,
    geometry,
):
    if kind == "range-preserving":
        return CRUWRangePreservingRVFusion(range_grid, angle_grid, geometry)
    if kind == "xy-radar-candidate":
        return CRUWXYRadarCandidateRVFusion(range_grid, angle_grid, geometry)
    raise ValueError(f"unknown fusion kind: {kind}")


def _run_fusion(fusion, radar, current, expert, expert_output, visual_output, radar_score, geometry):
    arguments = {
        "expert_xy_m": _expert_output_xy(expert, expert_output, geometry),
        "base_logits": current.logits,
        "shared_feature": current.feature,
        "polar_feature": current.polar_feature,
        "radar": radar,
        "radar_score": radar_score,
        "visual_theta": visual_output.theta[:, 0],
        "visual_sigma": visual_output.sigma_theta[:, 0],
        "visual_reliability": visual_output.reliability[:, 0],
        "visual_feature": visual_output.temporal_feature,
    }
    if isinstance(fusion, CRUWXYRadarCandidateRVFusion):
        if "candidate_xy_m" not in expert_output:
            raise ValueError("XY radar-candidate fusion requires expert radar candidates")
        arguments["radar_candidate_xy_m"] = expert_output["candidate_xy_m"]
    return fusion(**arguments)


@torch.no_grad()
def fixed_evidence(anchor, expert, visual, batch, device):
    radar = batch["radar"].to(device, non_blocking=True)
    current = anchor(radar)
    previous = anchor(shifted_previous(radar))
    visual_output = _run_visual(visual, batch, device)
    expert_output = _run_expert(expert, radar, current, previous, visual_output)
    radar_score = torch.sigmoid(current.logits).flatten(1).amax(dim=1)
    return radar, current, expert_output, visual_output, radar_score


def _metric_bundle(values: dict[str, np.ndarray]) -> dict[str, object]:
    expert = localization_metrics(
        values["expert_x"], values["expert_y"], values["target_x"], values["target_y"], values["score"]
    )
    fused = localization_metrics(
        values["fused_x"], values["fused_y"], values["target_x"], values["target_y"], values["score"]
    )
    expert_error = np.hypot(values["expert_x"] - values["target_x"], values["expert_y"] - values["target_y"])
    fused_error = np.hypot(values["fused_x"] - values["target_x"], values["fused_y"] - values["target_y"])
    absolute_y_shift = np.abs(values["fused_y"] - values["expert_y"])
    return {
        "expert": expert,
        "fusion": fused,
        "delta_AP@0.30_pp": 100.0 * (fused["AP@0.30"] - expert["AP@0.30"]),
        "accepted_rate": float(values["accepted"].mean()),
        "improved_rate": float(np.mean(fused_error + 1e-7 < expert_error)),
        "harmed_rate": float(np.mean(fused_error > expert_error + 1e-7)),
        "unchanged_rate": float(np.mean(np.abs(fused_error - expert_error) <= 1e-7)),
        "y_identity_max_abs_m": float(np.max(np.abs(values["fused_y"] - values["expert_y"]))),
        "absolute_y_shift_median_m": float(np.median(absolute_y_shift)),
        "absolute_y_shift_P95_m": float(np.quantile(absolute_y_shift, 0.95)),
    }


def _compose_candidate_outputs(
    collected: dict[str, np.ndarray],
    nonidentity_bias: float,
    *,
    include_per_sequence: bool,
) -> dict[str, object]:
    logits = collected["candidate_logits"].copy()
    logits[:, 1:] += nonidentity_bias
    selected_index = logits.argmax(axis=1)
    selected = collected["candidate_xy"][np.arange(selected_index.shape[0]), selected_index]
    values = {
        "expert_x": collected["expert_x"],
        "expert_y": collected["expert_y"],
        "fused_x": selected[:, 0],
        "fused_y": selected[:, 1],
        "target_x": collected["target_x"],
        "target_y": collected["target_y"],
        "score": collected["score"],
        "accepted": selected_index != 0,
    }
    result = _metric_bundle(values)
    if include_per_sequence:
        result["per_sequence"] = {}
        for sequence_id in sorted(np.unique(collected["sequence_id"])):
            mask = collected["sequence_id"] == sequence_id
            result["per_sequence"][str(int(sequence_id))] = _metric_bundle(
                {key: value[mask] for key, value in values.items()}
            )
    return result


def _compose_admitted_posterior_mean_outputs(
    collected: dict[str, np.ndarray],
    nonidentity_bias: float,
    temperature: float,
    expectation_noop_logit_offset: float = 0.0,
    *,
    include_per_sequence: bool,
) -> dict[str, object]:
    prediction, accepted = decode_admitted_posterior_mean(
        collected["candidate_xy"],
        collected["candidate_logits"],
        nonidentity_bias=nonidentity_bias,
        temperature=temperature,
        expectation_noop_logit_offset=expectation_noop_logit_offset,
    )
    values = {
        "expert_x": collected["expert_x"],
        "expert_y": collected["expert_y"],
        "fused_x": prediction[:, 0],
        "fused_y": prediction[:, 1],
        "target_x": collected["target_x"],
        "target_y": collected["target_y"],
        "score": collected["score"],
        "accepted": accepted,
    }
    result = _metric_bundle(values)
    if include_per_sequence:
        result["per_sequence"] = {}
        for sequence_id in sorted(np.unique(collected["sequence_id"])):
            mask = collected["sequence_id"] == sequence_id
            result["per_sequence"][str(int(sequence_id))] = _metric_bundle(
                {key: value[mask] for key, value in values.items()}
            )
    return result


def _compose_score_binned_admitted_posterior_mean_outputs(
    collected: dict[str, np.ndarray],
    score_threshold: float,
    low_score_nonidentity_bias: float,
    high_score_nonidentity_bias: float,
    temperature: float,
    expectation_noop_logit_offset: float = 0.0,
    *,
    include_per_sequence: bool,
) -> dict[str, object]:
    """Apply a two-bin, radar-score-only admission calibration.

    The threshold and both biases are decoder scalars selected on Development.
    They do not alter the RVFusion model, candidate lattice, or immutable range.
    """

    return _compose_feature_binned_admitted_posterior_mean_outputs(
        collected,
        feature=collected["score"],
        feature_threshold=score_threshold,
        low_feature_nonidentity_bias=low_score_nonidentity_bias,
        high_feature_nonidentity_bias=high_score_nonidentity_bias,
        temperature=temperature,
        expectation_noop_logit_offset=expectation_noop_logit_offset,
        include_per_sequence=include_per_sequence,
    )


def _compose_feature_binned_admitted_posterior_mean_outputs(
    collected: dict[str, np.ndarray],
    *,
    feature: np.ndarray,
    feature_threshold: float,
    low_feature_nonidentity_bias: float,
    high_feature_nonidentity_bias: float,
    temperature: float,
    expectation_noop_logit_offset: float = 0.0,
    include_per_sequence: bool,
) -> dict[str, object]:
    """Apply a two-bin scalar feature calibration to posterior admission."""

    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != collected["score"].shape:
        raise ValueError("calibration feature must contain one value per frame")
    frame_bias = np.where(
        feature <= feature_threshold,
        low_feature_nonidentity_bias,
        high_feature_nonidentity_bias,
    ).astype(np.float32)
    prediction, accepted = decode_admitted_posterior_mean(
        collected["candidate_xy"],
        collected["candidate_logits"],
        nonidentity_bias=frame_bias,
        temperature=temperature,
        expectation_noop_logit_offset=expectation_noop_logit_offset,
    )
    values = {
        "expert_x": collected["expert_x"],
        "expert_y": collected["expert_y"],
        "fused_x": prediction[:, 0],
        "fused_y": prediction[:, 1],
        "target_x": collected["target_x"],
        "target_y": collected["target_y"],
        "score": collected["score"],
        "accepted": accepted,
    }
    result = _metric_bundle(values)
    if include_per_sequence:
        result["per_sequence"] = {}
        for sequence_id in sorted(np.unique(collected["sequence_id"])):
            mask = collected["sequence_id"] == sequence_id
            result["per_sequence"][str(int(sequence_id))] = _metric_bundle(
                {key: value[mask] for key, value in values.items()}
            )
    return result


def _visual_bearing_logit_adjustment(
    collected: dict[str, np.ndarray],
    *,
    strength: float,
    sigma_floor_rad: float,
) -> np.ndarray:
    """Compute a frozen visual-bearing likelihood relative to candidate zero."""

    if sigma_floor_rad <= 0.0:
        raise ValueError("sigma_floor_rad must be positive")
    required = ("visual_theta", "visual_sigma_theta", "visual_reliability")
    if any(key not in collected for key in required):
        raise ValueError("collected values omit frozen visual bearing outputs")
    candidate_xy = collected["candidate_xy"]
    candidate_theta = np.arctan2(candidate_xy[..., 0], candidate_xy[..., 1])
    delta = np.arctan2(
        np.sin(candidate_theta - collected["visual_theta"][:, None]),
        np.cos(candidate_theta - collected["visual_theta"][:, None]),
    )
    sigma = np.maximum(np.asarray(collected["visual_sigma_theta"], dtype=np.float32), sigma_floor_rad)
    reliability = np.clip(np.asarray(collected["visual_reliability"], dtype=np.float32), 0.0, 1.0)
    adjustment = -0.5 * float(strength) * reliability[:, None] * np.square(delta / sigma[:, None])
    return (adjustment - adjustment[:, :1]).astype(np.float32)


def _compose_visual_bearing_admitted_posterior_mean_outputs(
    collected: dict[str, np.ndarray],
    *,
    nonidentity_bias: float | np.ndarray,
    temperature: float,
    expectation_noop_logit_offset: float,
    visual_bearing_strength: float,
    visual_sigma_floor_rad: float,
    include_per_sequence: bool,
) -> dict[str, object]:
    """Decode the fixed candidate lattice with a visual-bearing likelihood prior."""

    adjustment = _visual_bearing_logit_adjustment(
        collected,
        strength=visual_bearing_strength,
        sigma_floor_rad=visual_sigma_floor_rad,
    )
    prediction, accepted = decode_admitted_posterior_mean(
        collected["candidate_xy"],
        collected["candidate_logits"],
        nonidentity_bias=nonidentity_bias,
        temperature=temperature,
        expectation_noop_logit_offset=expectation_noop_logit_offset,
        candidate_logit_adjustment=adjustment,
    )
    values = {
        "expert_x": collected["expert_x"],
        "expert_y": collected["expert_y"],
        "fused_x": prediction[:, 0],
        "fused_y": prediction[:, 1],
        "target_x": collected["target_x"],
        "target_y": collected["target_y"],
        "score": collected["score"],
        "accepted": accepted,
    }
    result = _metric_bundle(values)
    if include_per_sequence:
        result["per_sequence"] = {}
        for sequence_id in sorted(np.unique(collected["sequence_id"])):
            mask = collected["sequence_id"] == sequence_id
            result["per_sequence"][str(int(sequence_id))] = _metric_bundle(
                {key: value[mask] for key, value in values.items()}
            )
    return result


def _compose_candidate_oracle_outputs(
    collected: dict[str, np.ndarray],
    *,
    include_per_sequence: bool,
) -> dict[str, object]:
    """Report only the geometric capacity of the fixed candidate lattice."""

    target = np.stack((collected["target_x"], collected["target_y"]), axis=1)
    candidate_error = np.linalg.norm(collected["candidate_xy"] - target[:, None, :], axis=-1)
    selected_index = candidate_error.argmin(axis=1)
    selected = collected["candidate_xy"][np.arange(selected_index.shape[0]), selected_index]
    values = {
        "expert_x": collected["expert_x"],
        "expert_y": collected["expert_y"],
        "fused_x": selected[:, 0],
        "fused_y": selected[:, 1],
        "target_x": collected["target_x"],
        "target_y": collected["target_y"],
        "score": collected["score"],
        "accepted": selected_index != 0,
    }
    result = _metric_bundle(values)
    result["oracle_only"] = True
    if include_per_sequence:
        result["per_sequence"] = {}
        for sequence_id in sorted(np.unique(collected["sequence_id"])):
            mask = collected["sequence_id"] == sequence_id
            result["per_sequence"][str(int(sequence_id))] = _metric_bundle(
                {key: value[mask] for key, value in values.items()}
            )
    return result


def _calibrate_nonidentity_bias(collected: dict[str, np.ndarray]) -> tuple[float, dict[str, object]]:
    # The first operating point is an exact identity fallback. The remaining
    # grid is selected only on Development and is frozen into the checkpoint.
    candidates = [-1e6, *np.linspace(-8.0, 2.0, 201).tolist()]
    scored: list[tuple[tuple[float, float, float], float, dict[str, object]]] = []
    for bias in candidates:
        metrics = _compose_candidate_outputs(collected, bias, include_per_sequence=False)
        rank = (
            float(metrics["fusion"]["AP@0.30"]),
            -float(metrics["fusion"]["P90_error_m"]),
            -abs(float(bias)),
        )
        scored.append((rank, float(bias), metrics))
    _, selected_bias, _ = max(scored, key=lambda item: item[0])
    selected_metrics = _compose_candidate_outputs(collected, selected_bias, include_per_sequence=True)
    selected_metrics["development_calibration"] = {
        "selection_metric": "AP@0.30, then lower P90, then smaller absolute bias",
        "selected_nonidentity_bias": selected_bias,
        "candidate_count": len(candidates),
        "includes_exact_all_fallback": True,
    }
    return selected_bias, selected_metrics


def _calibrate_admitted_posterior_mean(
    collected: dict[str, np.ndarray],
) -> tuple[float, float, dict[str, object]]:
    """Select a causal posterior-mean decoder only from Development records."""

    biases = [-1e6, *np.linspace(-8.0, 2.0, 201).tolist()]
    temperatures = (0.25, 0.50, 0.75, 1.0, 1.50, 2.0, 3.0)
    expectation_noop_offsets = (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0)
    scored: list[tuple[tuple[float, float, float, float, float, float], float, float, float]] = []
    for bias in biases:
        for temperature in temperatures:
            for noop_offset in expectation_noop_offsets:
                metrics = _compose_admitted_posterior_mean_outputs(
                    collected,
                    bias,
                    temperature,
                    noop_offset,
                    include_per_sequence=False,
                )
                rank = (
                    float(metrics["fusion"]["AP@0.30"]),
                    float(metrics["fusion"]["AP@0.25"]),
                    -float(metrics["fusion"]["P90_error_m"]),
                    -abs(float(bias)),
                    -abs(float(np.log(temperature))),
                    -abs(float(noop_offset)),
                )
                scored.append((rank, float(bias), float(temperature), float(noop_offset)))
    _, selected_bias, selected_temperature, selected_noop_offset = max(scored, key=lambda item: item[0])
    selected_metrics = _compose_admitted_posterior_mean_outputs(
        collected,
        selected_bias,
        selected_temperature,
        selected_noop_offset,
        include_per_sequence=True,
    )
    selected_metrics["development_calibration"] = {
        "selection_metric": "AP@0.30, then AP@0.25, then lower P90, then smaller calibration shift",
        "selected_nonidentity_bias": selected_bias,
        "selected_temperature": selected_temperature,
        "selected_expectation_noop_logit_offset": selected_noop_offset,
        "candidate_count": len(biases) * len(temperatures) * len(expectation_noop_offsets),
        "includes_exact_all_fallback": True,
    }
    return selected_bias, selected_temperature, selected_metrics


def calibrate_score_binned_admitted_posterior_mean(
    collected: dict[str, np.ndarray],
) -> dict[str, object]:
    """Audit a compact radar-score-only fallback calibration on Development."""

    return calibrate_feature_binned_admitted_posterior_mean(
        collected,
        feature=collected["score"],
        feature_name="radar_score",
        constrain_low_bias_at_least_high=True,
    )


def calibrate_visual_reliability_binned_admitted_posterior_mean(
    collected: dict[str, np.ndarray],
) -> dict[str, object]:
    """Audit a two-bin calibration using the frozen visual reliability output."""

    if "visual_reliability" not in collected:
        raise ValueError("collected evaluation values omit visual_reliability")
    return calibrate_feature_binned_admitted_posterior_mean(
        collected,
        feature=collected["visual_reliability"],
        feature_name="visual_reliability",
        constrain_low_bias_at_least_high=False,
    )


def calibrate_visual_bearing_admitted_posterior_mean(
    collected: dict[str, np.ndarray],
) -> dict[str, object]:
    """Development-only likelihood calibration using frozen visual bearing outputs."""

    base = calibrate_decoder_modes(collected)["admitted_posterior_mean"]
    base_bias = float(base["nonidentity_bias"])
    base_temperature = float(base["temperature"])
    base_offset = float(base["metrics"]["development_calibration"]["selected_expectation_noop_logit_offset"])
    biases = tuple(float(base_bias + shift) for shift in (-1.0, -0.5, 0.0, 0.5, 1.0))
    temperatures = tuple(sorted({0.50, 1.0, 1.50, 3.0, base_temperature}))
    expectation_noop_offsets = tuple(sorted({-1.0, 0.0, 1.0, base_offset}))
    strengths = (0.0, 0.125, 0.25, 0.50, 1.0, 2.0, 4.0)
    sigma_floors = tuple(float(np.deg2rad(value)) for value in (2.0, 5.0, 10.0, 15.0, 20.0))
    scored: list[tuple[tuple[float, float, float, float, float, float, float], float, float, float, float, float]] = []
    for bias in biases:
        for temperature in temperatures:
            for noop_offset in expectation_noop_offsets:
                for strength in strengths:
                    for sigma_floor in sigma_floors:
                        metrics = _compose_visual_bearing_admitted_posterior_mean_outputs(
                            collected,
                            nonidentity_bias=bias,
                            temperature=temperature,
                            expectation_noop_logit_offset=noop_offset,
                            visual_bearing_strength=strength,
                            visual_sigma_floor_rad=sigma_floor,
                            include_per_sequence=False,
                        )
                        rank = (
                            float(metrics["fusion"]["AP@0.30"]),
                            float(metrics["fusion"]["AP@0.25"]),
                            -float(metrics["fusion"]["P90_error_m"]),
                            -abs(bias - base_bias),
                            -abs(float(np.log(temperature / base_temperature))),
                            -abs(noop_offset - base_offset),
                            -strength,
                        )
                        scored.append((rank, bias, temperature, noop_offset, strength, sigma_floor))
    _, bias, temperature, noop_offset, strength, sigma_floor = max(scored, key=lambda item: item[0])
    metrics = _compose_visual_bearing_admitted_posterior_mean_outputs(
        collected,
        nonidentity_bias=bias,
        temperature=temperature,
        expectation_noop_logit_offset=noop_offset,
        visual_bearing_strength=strength,
        visual_sigma_floor_rad=sigma_floor,
        include_per_sequence=True,
    )
    metrics["development_calibration"] = {
        "selection_metric": "AP@0.30, then AP@0.25, then lower P90, then smaller calibration shift",
        "selected_nonidentity_bias": bias,
        "selected_temperature": temperature,
        "selected_expectation_noop_logit_offset": noop_offset,
        "selected_visual_bearing_strength": strength,
        "selected_visual_sigma_floor_deg": float(np.rad2deg(sigma_floor)),
        "candidate_count": len(scored),
        "decoder": "visual-bearing likelihood calibration",
    }
    return metrics


def calibrate_score_binned_visual_bearing_admitted_posterior_mean(
    collected: dict[str, np.ndarray],
) -> dict[str, object]:
    """Calibrate score-dependent admission with the frozen bearing likelihood.

    This is a compact readout-only search.  The two admission biases use the
    frozen radar score, while the visual term only reorders the existing local
    candidates.  Neither input changes the candidate lattice or radar range.
    """

    base = calibrate_visual_bearing_admitted_posterior_mean(collected)
    calibration = base["development_calibration"]
    base_bias = float(calibration["selected_nonidentity_bias"])
    base_temperature = float(calibration["selected_temperature"])
    base_noop_offset = float(calibration["selected_expectation_noop_logit_offset"])
    base_strength = float(calibration["selected_visual_bearing_strength"])
    base_sigma_floor = float(np.deg2rad(calibration["selected_visual_sigma_floor_deg"]))
    thresholds = tuple(float(np.quantile(collected["score"], q)) for q in (0.25, 0.50, 0.75))
    bias_shifts = (-1.0, -0.5, 0.0, 0.5, 1.0)
    temperatures = tuple(sorted({1.5, 3.0, base_temperature}))
    noop_offsets = tuple(sorted({0.0, 1.0, base_noop_offset}))
    strengths = tuple(sorted({0.5, 1.0, 2.0, base_strength}))
    sigma_floors = tuple(sorted({float(np.deg2rad(value)) for value in (2.0, 5.0, 10.0)} | {base_sigma_floor}))
    scored: list[
        tuple[
            tuple[float, float, float, float, float, float, float, float],
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ] = []
    for threshold in thresholds:
        for low_shift in bias_shifts:
            for high_shift in bias_shifts:
                low_bias = base_bias + low_shift
                high_bias = base_bias + high_shift
                # Weak radar evidence may admit more bounded visual correction,
                # never less, than the high-score bin.
                if low_bias < high_bias:
                    continue
                frame_bias = np.where(collected["score"] <= threshold, low_bias, high_bias).astype(np.float32)
                for temperature in temperatures:
                    for noop_offset in noop_offsets:
                        for strength in strengths:
                            for sigma_floor in sigma_floors:
                                metrics = _compose_visual_bearing_admitted_posterior_mean_outputs(
                                    collected,
                                    nonidentity_bias=frame_bias,
                                    temperature=temperature,
                                    expectation_noop_logit_offset=noop_offset,
                                    visual_bearing_strength=strength,
                                    visual_sigma_floor_rad=sigma_floor,
                                    include_per_sequence=False,
                                )
                                rank = (
                                    float(metrics["fusion"]["AP@0.30"]),
                                    float(metrics["fusion"]["AP@0.25"]),
                                    -float(metrics["fusion"]["P90_error_m"]),
                                    -abs(low_shift) - abs(high_shift),
                                    -abs(float(np.log(temperature / base_temperature))),
                                    -abs(noop_offset - base_noop_offset),
                                    -abs(strength - base_strength),
                                    -abs(sigma_floor - base_sigma_floor),
                                )
                                scored.append(
                                    (rank, threshold, low_bias, high_bias, temperature, noop_offset, strength, sigma_floor)
                                )
    _, threshold, low_bias, high_bias, temperature, noop_offset, strength, sigma_floor = max(
        scored, key=lambda item: item[0]
    )
    frame_bias = np.where(collected["score"] <= threshold, low_bias, high_bias).astype(np.float32)
    metrics = _compose_visual_bearing_admitted_posterior_mean_outputs(
        collected,
        nonidentity_bias=frame_bias,
        temperature=temperature,
        expectation_noop_logit_offset=noop_offset,
        visual_bearing_strength=strength,
        visual_sigma_floor_rad=sigma_floor,
        include_per_sequence=True,
    )
    metrics["development_calibration"] = {
        "selection_metric": "AP@0.30, then AP@0.25, then lower P90, then smaller calibration shift",
        "selected_score_threshold": threshold,
        "selected_low_score_nonidentity_bias": low_bias,
        "selected_high_score_nonidentity_bias": high_bias,
        "selected_temperature": temperature,
        "selected_expectation_noop_logit_offset": noop_offset,
        "selected_visual_bearing_strength": strength,
        "selected_visual_sigma_floor_deg": float(np.rad2deg(sigma_floor)),
        "candidate_count": len(scored),
        "decoder": "score-binned visual-bearing likelihood calibration",
    }
    return metrics


def calibrate_fine_score_binned_visual_bearing_admitted_posterior_mean(
    collected: dict[str, np.ndarray],
) -> dict[str, object]:
    """Refine only the selected score/bearing decoder operating point on Development."""

    base = calibrate_score_binned_visual_bearing_admitted_posterior_mean(collected)
    calibration = base["development_calibration"]
    base_threshold = float(calibration["selected_score_threshold"])
    base_low_bias = float(calibration["selected_low_score_nonidentity_bias"])
    base_high_bias = float(calibration["selected_high_score_nonidentity_bias"])
    base_temperature = float(calibration["selected_temperature"])
    base_noop_offset = float(calibration["selected_expectation_noop_logit_offset"])
    base_strength = float(calibration["selected_visual_bearing_strength"])
    base_sigma_floor = float(np.deg2rad(calibration["selected_visual_sigma_floor_deg"]))
    thresholds = tuple(float(np.quantile(collected["score"], q)) for q in np.linspace(0.35, 0.65, 7))
    low_biases = tuple(float(value) for value in np.arange(base_low_bias - 0.3, base_low_bias + 0.31, 0.1))
    high_biases = tuple(float(value) for value in np.arange(base_high_bias - 0.3, base_high_bias + 0.31, 0.1))
    temperatures = tuple(float(value) for value in np.arange(max(0.5, base_temperature - 1.0), base_temperature + 1.01, 0.5))
    noop_offsets = tuple(float(value) for value in np.arange(base_noop_offset - 0.5, base_noop_offset + 0.51, 0.5))
    strengths = tuple(float(value) for value in np.arange(max(0.0, base_strength - 0.25), base_strength + 0.26, 0.25))
    sigma_floors = tuple(float(np.deg2rad(value)) for value in (1.0, 2.0, 3.0))
    scored: list[
        tuple[
            tuple[float, float, float, float, float, float, float, float],
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ] = []
    for threshold in thresholds:
        for low_bias in low_biases:
            for high_bias in high_biases:
                if low_bias < high_bias:
                    continue
                frame_bias = np.where(collected["score"] <= threshold, low_bias, high_bias).astype(np.float32)
                for temperature in temperatures:
                    for noop_offset in noop_offsets:
                        for strength in strengths:
                            for sigma_floor in sigma_floors:
                                metrics = _compose_visual_bearing_admitted_posterior_mean_outputs(
                                    collected,
                                    nonidentity_bias=frame_bias,
                                    temperature=temperature,
                                    expectation_noop_logit_offset=noop_offset,
                                    visual_bearing_strength=strength,
                                    visual_sigma_floor_rad=sigma_floor,
                                    include_per_sequence=False,
                                )
                                rank = (
                                    float(metrics["fusion"]["AP@0.30"]),
                                    float(metrics["fusion"]["AP@0.25"]),
                                    -float(metrics["fusion"]["P90_error_m"]),
                                    -abs(low_bias - base_low_bias) - abs(high_bias - base_high_bias),
                                    -abs(float(np.log(temperature / base_temperature))),
                                    -abs(noop_offset - base_noop_offset),
                                    -abs(strength - base_strength),
                                    -abs(sigma_floor - base_sigma_floor),
                                )
                                scored.append(
                                    (rank, threshold, low_bias, high_bias, temperature, noop_offset, strength, sigma_floor)
                                )
    _, threshold, low_bias, high_bias, temperature, noop_offset, strength, sigma_floor = max(
        scored, key=lambda item: item[0]
    )
    frame_bias = np.where(collected["score"] <= threshold, low_bias, high_bias).astype(np.float32)
    metrics = _compose_visual_bearing_admitted_posterior_mean_outputs(
        collected,
        nonidentity_bias=frame_bias,
        temperature=temperature,
        expectation_noop_logit_offset=noop_offset,
        visual_bearing_strength=strength,
        visual_sigma_floor_rad=sigma_floor,
        include_per_sequence=True,
    )
    metrics["development_calibration"] = {
        "selection_metric": "AP@0.30, then AP@0.25, then lower P90, then smaller calibration shift",
        "selected_score_threshold": threshold,
        "selected_low_score_nonidentity_bias": low_bias,
        "selected_high_score_nonidentity_bias": high_bias,
        "selected_temperature": temperature,
        "selected_expectation_noop_logit_offset": noop_offset,
        "selected_visual_bearing_strength": strength,
        "selected_visual_sigma_floor_deg": float(np.rad2deg(sigma_floor)),
        "candidate_count": len(scored),
        "decoder": "fine score-binned visual-bearing likelihood calibration",
    }
    return metrics


def calibrate_feature_binned_admitted_posterior_mean(
    collected: dict[str, np.ndarray],
    *,
    feature: np.ndarray,
    feature_name: str,
    constrain_low_bias_at_least_high: bool,
) -> dict[str, object]:
    """Development-only two-bin calibration for an existing scalar evidence feature."""

    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != collected["score"].shape:
        raise ValueError("calibration feature must contain one value per frame")
    score_thresholds = tuple(float(np.quantile(feature, q)) for q in (0.2, 0.4, 0.6, 0.8))
    biases = tuple(float(value) for value in np.linspace(-7.0, -1.0, 13))
    temperatures = (0.25, 0.50, 1.0, 1.50, 3.0)
    expectation_noop_offsets = (-1.0, 0.0, 1.0)
    scored: list[tuple[tuple[float, float, float, float, float, float, float], float, float, float, float, float]] = []
    for threshold in score_thresholds:
        for low_bias in biases:
            for high_bias in biases:
                if constrain_low_bias_at_least_high and low_bias < high_bias:
                    continue
                for temperature in temperatures:
                    for noop_offset in expectation_noop_offsets:
                        metrics = _compose_feature_binned_admitted_posterior_mean_outputs(
                            collected,
                            feature=feature,
                            feature_threshold=threshold,
                            low_feature_nonidentity_bias=low_bias,
                            high_feature_nonidentity_bias=high_bias,
                            temperature=temperature,
                            expectation_noop_logit_offset=noop_offset,
                            include_per_sequence=False,
                        )
                        rank = (
                            float(metrics["fusion"]["AP@0.30"]),
                            float(metrics["fusion"]["AP@0.25"]),
                            -float(metrics["fusion"]["P90_error_m"]),
                            -abs(low_bias - high_bias),
                            -abs(float(np.log(temperature))),
                            -abs(noop_offset),
                            -abs(threshold - float(np.median(collected["score"]))),
                        )
                        scored.append((rank, threshold, low_bias, high_bias, temperature, noop_offset))
    _, threshold, low_bias, high_bias, temperature, noop_offset = max(scored, key=lambda item: item[0])
    metrics = _compose_feature_binned_admitted_posterior_mean_outputs(
        collected,
        feature=feature,
        feature_threshold=threshold,
        low_feature_nonidentity_bias=low_bias,
        high_feature_nonidentity_bias=high_bias,
        temperature=temperature,
        expectation_noop_logit_offset=noop_offset,
        include_per_sequence=True,
    )
    metrics["development_calibration"] = {
        "selection_metric": "AP@0.30, then AP@0.25, then lower P90, then smaller calibration shift",
        "feature_name": feature_name,
        "selected_feature_threshold": threshold,
        "selected_low_feature_nonidentity_bias": low_bias,
        "selected_high_feature_nonidentity_bias": high_bias,
        "selected_temperature": temperature,
        "selected_expectation_noop_logit_offset": noop_offset,
        "candidate_count": len(scored),
        "decoder": f"two-bin {feature_name} admission calibration",
    }
    return metrics


def calibrate_decoder_modes(collected: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    """Calibrate MAP and bounded posterior-mean readouts on Development only."""

    map_bias, map_metrics = _calibrate_nonidentity_bias(collected)
    posterior_bias, temperature, posterior_metrics = _calibrate_admitted_posterior_mean(collected)
    return {
        "map": {
            "nonidentity_bias": map_bias,
            "metrics": map_metrics,
        },
        "admitted_posterior_mean": {
            "nonidentity_bias": posterior_bias,
            "temperature": temperature,
            "metrics": posterior_metrics,
        },
    }


def calibrate_training_decoder(
    collected: dict[str, np.ndarray],
    selection_decoder: str,
) -> dict[str, object]:
    """Choose the checkpoint readout from Development without changing the head."""

    modes = calibrate_decoder_modes(collected)
    if selection_decoder not in modes:
        raise ValueError(f"unsupported selection decoder: {selection_decoder}")
    return modes[selection_decoder]


@torch.no_grad()
def evaluate(
    anchor,
    expert,
    visual,
    fusion,
    loader,
    device,
    geometry,
    *,
    calibrate_nonidentity_bias: bool = False,
    return_collected: bool = False,
):
    _freeze(anchor)
    _freeze(expert)
    _freeze(visual)
    fusion.eval()
    keys = (
        "expert_x",
        "expert_y",
        "candidate_xy",
        "candidate_logits",
        "visual_theta",
        "visual_reliability",
        "visual_sigma_theta",
        "target_x",
        "target_y",
        "score",
        "sequence_id",
    )
    parts = {key: [] for key in keys}
    for batch in loader:
        radar, current, expert_output, visual_output, radar_score = fixed_evidence(anchor, expert, visual, batch, device)
        output = _run_fusion(
            fusion, radar, current, expert, expert_output, visual_output, radar_score, geometry
        )
        target = torch.stack((batch["single_x_m"], batch["single_y_m"]), dim=1).to(device)
        values = {
            "expert_x": _expert_output_xy(expert, expert_output, geometry)[:, 0].cpu().numpy(),
            "expert_y": _expert_output_xy(expert, expert_output, geometry)[:, 1].cpu().numpy(),
            "candidate_xy": output.candidate_xy_m.cpu().numpy(),
            "candidate_logits": output.candidate_logits.cpu().numpy(),
            "visual_theta": visual_output.theta[:, 0].cpu().numpy(),
            "visual_reliability": visual_output.reliability[:, 0].cpu().numpy(),
            "visual_sigma_theta": visual_output.sigma_theta[:, 0].cpu().numpy(),
            "target_x": target[:, 0].cpu().numpy(),
            "target_y": target[:, 1].cpu().numpy(),
            "score": radar_score.cpu().numpy(),
            "sequence_id": batch["sequence_id"].numpy(),
        }
        for key in keys:
            parts[key].append(values[key])
    collected = {key: np.concatenate(value) for key, value in parts.items()}
    if return_collected:
        return collected
    if calibrate_nonidentity_bias:
        return _calibrate_nonidentity_bias(collected)
    return _compose_candidate_outputs(collected, 0.0, include_per_sequence=True)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.identity_target_weight <= 1.0:
        raise ValueError("--identity-target-weight must be in (0, 1]")
    if args.posterior_regression_weight < 0.0:
        raise ValueError("--posterior-regression-weight must be non-negative")
    if args.posterior_regression_beta <= 0.0:
        raise ValueError("--posterior-regression-beta must be positive")
    if args.utility_target_radius_m <= 0.0:
        raise ValueError("--utility-target-radius-m must be positive")
    if args.soft_target_sigma_m <= 0.0:
        raise ValueError("--soft-target-sigma-m must be positive")
    if args.ap30_transition_loss_weight <= 0.0:
        raise ValueError("--ap30-transition-loss-weight must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = torch.device("cuda")
    geometry = CRUWBEVGeometry()
    protocol = load_protocol(args.protocol)
    payload = json.loads(args.protocol.read_text(encoding="utf-8"))
    fit_allowed = {int(value) for value in payload["single_person_fit_sequences"]}
    development_allowed = {int(value) for value in payload["single_person_validation_sequences"]}
    if fit_allowed & development_allowed:
        raise RuntimeError("single-person fit/development leakage")
    range_grid, angle_grid = load_cruw_grids(args.data_root, args.cruw_devkit_root)
    fit_records = select_single_person_records(
        build_all_scene_index(args.data_root, protocol, "fit", args.sample_stride), fit_allowed, geometry
    )
    development_records = select_single_person_records(
        build_all_scene_index(args.data_root, protocol, "validation", args.sample_stride), development_allowed, geometry
    )
    if args.visual_kind == "causal-vast":
        if not all((args.visual_manifest, args.proposal_cache, args.roi_feature_cache)):
            raise ValueError("causal VAST requires manifest, proposal cache and RoI feature cache")
        from brace_v4.vast_mamba import CausalVastMambaDataset

        fit_vast = CausalVastMambaDataset(
            args.visual_manifest, args.proposal_cache, split="train", roi_feature_path=args.roi_feature_cache
        )
        development_vast = CausalVastMambaDataset(
            args.visual_manifest, args.proposal_cache, split="development", roi_feature_path=args.roi_feature_cache
        )
        fit_dataset = _AlignedVASTFusionDataset(
            fit_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry, fit_vast
        )
        development_dataset = _AlignedVASTFusionDataset(
            development_records, args.cache_root, range_grid, angle_grid, args.clip_length, geometry, development_vast
        )
    else:
        fit_dataset = _AlignedFusionDataset(
            fit_records,
            args.cache_root,
            range_grid,
            angle_grid,
            args.clip_length,
            args.image_size,
            geometry,
            True,
            args.fusion_augmentation_mode,
        )
        development_dataset = _AlignedFusionDataset(
            development_records,
            args.cache_root,
            range_grid,
            angle_grid,
            args.clip_length,
            args.image_size,
            geometry,
            False,
            "none",
        )
    fit_loader = _loader(fit_dataset, args.batch_size, args.workers, True)
    development_loader = _loader(development_dataset, args.batch_size, args.workers, False)

    anchor = DREAMBEV(range_grid, angle_grid, geometry).to(device)
    anchor.load_state_dict(torch.load(args.anchor_checkpoint, map_location=device, weights_only=False)["model"], strict=True)
    expert = _build_expert(args.expert_kind, range_grid, angle_grid, geometry).to(device)
    expert.load_state_dict(torch.load(args.expert_checkpoint, map_location=device, weights_only=False)["model"], strict=True)
    if args.visual_kind == "causal-vast":
        from brace_v4.vast_mamba import VastMambaObserver

        visual = VastMambaObserver().to(device)
        visual.load_state_dict(
            torch.load(args.visual_checkpoint, map_location=device, weights_only=False)["model_state"], strict=True
        )
    else:
        visual = RVBearingNet().to(device)
        visual.load_state_dict(
            torch.load(args.visual_checkpoint, map_location=device, weights_only=False)["model"], strict=True
        )
    for module in (anchor, expert, visual):
        _freeze(module)
    if args.fusion_kind == "xy-radar-candidate" and args.expert_kind != "global-association":
        raise ValueError("XY radar-candidate fusion requires --expert-kind global-association")
    fusion = _build_fusion(
        args.fusion_kind,
        range_grid,
        angle_grid,
        geometry,
    )
    if args.visual_kind == "causal-vast" and isinstance(fusion, CRUWRangePreservingRVFusion):
        fusion = CRUWRangePreservingRVFusion(
            range_grid,
            angle_grid,
            geometry,
            visual_dim=128,
        )
    fusion = fusion.to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = {
        "model": "CRUW candidate-local RVFusion",
        "fusion_kind": args.fusion_kind,
        "expert_kind": args.expert_kind,
        "expert_anchor_coordinate": (
            "final_bounded_paper_expert_bev_logit_peak"
            if args.expert_kind in PAPER_EXPERT_KINDS
            else "expert_proposal_coordinate"
        ),
        "visual_kind": args.visual_kind,
        "utility_target_mode": args.utility_target_mode,
        "utility_target_radius_m": args.utility_target_radius_m,
        "soft_target_sigma_m": args.soft_target_sigma_m,
        "ap30_transition_loss_weight": args.ap30_transition_loss_weight,
        "selection_decoder": args.selection_decoder,
        "contract": (
            "frozen expert-radar anchor; causal left RGB bearing/temporal evidence; local lateral candidates plus "
            "global radar-supported XY candidates; exact candidate-zero fallback; no direct visual range regression; "
            "no VLM, depth, RIS, route label, future frame, or official CRUW test"
            if args.fusion_kind == "xy-radar-candidate"
            else "frozen expert-radar anchor; causal left RGB bearing/temporal evidence; bounded lateral candidates; "
            "exact candidate-zero fallback; immutable expert y; no VLM, depth, RIS, route label, future frame, "
            "or official CRUW test"
        ),
        "fit_sequences": sorted(fit_allowed),
        "development_sequences": sorted(development_allowed),
        "fit_frames": len(fit_records),
        "development_frames": len(development_records),
        "trainable_parameters": sum(parameter.numel() for parameter in fusion.parameters() if parameter.requires_grad),
        "anchor_sha256": sha256(args.anchor_checkpoint),
        "expert_sha256": sha256(args.expert_checkpoint),
        "visual_sha256": sha256(args.visual_checkpoint),
        "protocol_sha256": sha256(args.protocol),
        "official_test_read": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    best_ap30 = -float("inf")
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        fusion.nonidentity_bias.zero_()
        fusion.train()
        losses: list[float] = []
        nonidentity_targets: list[float] = []
        for batch in fit_loader:
            radar, current, expert_output, visual_output, radar_score = fixed_evidence(
                anchor, expert, visual, batch, device
            )
            output = _run_fusion(
                fusion, radar, current, expert, expert_output, visual_output, radar_score, geometry
            )
            target = torch.stack((batch["single_x_m"], batch["single_y_m"]), dim=1).to(device)
            transition_mask = ap30_recoverable_transition_mask(
                output.candidate_xy_m.detach(),
                target,
                success_radius_m=args.utility_target_radius_m,
            )
            soft_targets = None
            if isinstance(fusion, CRUWXYRadarCandidateRVFusion):
                labels = xy_utility_targets(
                    output.candidate_xy_m.detach(),
                    target,
                    minimum_improvement_m=args.minimum_improvement_m,
                    candidate_valid_mask=output.candidate_valid_mask,
                )
            elif args.utility_target_mode == "ap30_reachable":
                labels = reachable_lateral_utility_targets(
                    output.candidate_xy_m.detach(),
                    target,
                    minimum_improvement_m=args.minimum_improvement_m,
                    success_radius_m=args.utility_target_radius_m,
                )
            elif args.utility_target_mode == "lateral_soft":
                soft_targets = soft_lateral_utility_targets(
                    output.candidate_xy_m.detach(),
                    target,
                    minimum_improvement_m=args.minimum_improvement_m,
                    sigma_m=args.soft_target_sigma_m,
                )
                labels = soft_targets.argmax(dim=1)
            elif args.utility_target_mode == "ap30_transition":
                labels = ap30_transition_utility_targets(
                    output.candidate_xy_m.detach(),
                    target,
                    minimum_improvement_m=args.minimum_improvement_m,
                    success_radius_m=args.utility_target_radius_m,
                )
            elif args.utility_target_mode == "ap30_transition_soft":
                soft_targets = ap30_transition_soft_targets(
                    output.candidate_xy_m.detach(),
                    target,
                    minimum_improvement_m=args.minimum_improvement_m,
                    success_radius_m=args.utility_target_radius_m,
                    sigma_m=args.soft_target_sigma_m,
                )
                labels = soft_targets.argmax(dim=1)
            else:
                labels = lateral_utility_targets(
                    output.candidate_xy_m.detach(), target, minimum_improvement_m=args.minimum_improvement_m
                )
            if soft_targets is None:
                per_sample = F.cross_entropy(output.candidate_logits, labels, reduction="none")
                fallback_target = labels.eq(0)
            else:
                per_sample = -(soft_targets * F.log_softmax(output.candidate_logits, dim=1)).sum(dim=1)
                fallback_target = soft_targets[:, 0].eq(1.0)
            weights = torch.where(
                fallback_target,
                per_sample.new_tensor(args.identity_target_weight),
                per_sample.new_tensor(1.0),
            )
            weights = weights * torch.where(
                transition_mask,
                per_sample.new_tensor(args.ap30_transition_loss_weight),
                per_sample.new_tensor(1.0),
            )
            posterior_xy = (output.posterior[..., None] * output.candidate_xy_m).sum(dim=1)
            regression_prediction = posterior_xy if isinstance(fusion, CRUWXYRadarCandidateRVFusion) else posterior_xy[:, :1]
            regression_target = target if isinstance(fusion, CRUWXYRadarCandidateRVFusion) else target[:, :1]
            loss = (per_sample * weights).mean() + args.posterior_regression_weight * F.smooth_l1_loss(
                regression_prediction, regression_target, beta=args.posterior_regression_beta
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            nonidentity_targets.append(float((~fallback_target).float().mean()))
        scheduler.step()
        collected = evaluate(
            anchor,
            expert,
            visual,
            fusion,
            development_loader,
            device,
            geometry,
            return_collected=True,
        )
        selected_decoder = calibrate_training_decoder(collected, args.selection_decoder)
        selected_bias = float(selected_decoder["nonidentity_bias"])
        development = selected_decoder["metrics"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "nonidentity_target_rate": float(np.mean(nonidentity_targets)),
            "elapsed_seconds": time.time() - started,
            **development,
        }
        with (args.output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        ap30 = development["fusion"]["AP@0.30"]
        if ap30 > best_ap30 + 1e-6:
            best_ap30 = ap30
            stale = 0
            state = {key: value.detach().cpu().clone() for key, value in fusion.state_dict().items()}
            state["nonidentity_bias"] = torch.tensor(selected_bias, dtype=torch.float32)
            torch.save(
                {
                    "model": state,
                    "epoch": epoch,
                    "metrics": development,
                    "development_selected_nonidentity_bias": selected_bias,
                    "development_selected_decoder": args.selection_decoder,
                    "development_selected_temperature": selected_decoder.get("temperature"),
                    "development_selected_expectation_noop_logit_offset": development[
                        "development_calibration"
                    ].get("selected_expectation_noop_logit_offset", 0.0),
                },
                args.output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    reloaded = _build_fusion(
        args.fusion_kind,
        range_grid,
        angle_grid,
        geometry,
    )
    if args.visual_kind == "causal-vast" and isinstance(reloaded, CRUWRangePreservingRVFusion):
        reloaded = CRUWRangePreservingRVFusion(
            range_grid,
            angle_grid,
            geometry,
            visual_dim=128,
        )
    reloaded = reloaded.to(device)
    reloaded.load_state_dict(checkpoint["model"], strict=True)
    reloaded.nonidentity_bias.zero_()
    reloaded_collected = evaluate(
        anchor,
        expert,
        visual,
        reloaded,
        development_loader,
        device,
        geometry,
        return_collected=True,
    )
    if checkpoint["development_selected_decoder"] == "admitted_posterior_mean":
        independent_reload = _compose_admitted_posterior_mean_outputs(
            reloaded_collected,
            checkpoint["development_selected_nonidentity_bias"],
            float(checkpoint["development_selected_temperature"]),
            float(checkpoint["development_selected_expectation_noop_logit_offset"]),
            include_per_sequence=True,
        )
    else:
        independent_reload = _compose_candidate_outputs(
            reloaded_collected,
            checkpoint["development_selected_nonidentity_bias"],
            include_per_sequence=True,
        )
    report = {
        "best_epoch": checkpoint["epoch"],
        "development_selected_nonidentity_bias": checkpoint["development_selected_nonidentity_bias"],
        "development_selected_decoder": checkpoint["development_selected_decoder"],
        "development_selected_temperature": checkpoint["development_selected_temperature"],
        "development_selected_expectation_noop_logit_offset": checkpoint[
            "development_selected_expectation_noop_logit_offset"
        ],
        "development_calibration": checkpoint["metrics"]["development_calibration"],
        "independent_reload": independent_reload,
        "checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "elapsed_seconds": time.time() - started,
        "official_test_read": False,
    }
    (args.output_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
