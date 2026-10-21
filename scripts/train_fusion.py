from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
POLICY_SCHEDULE = json.loads(
    (PROJECT_ROOT / "contracts" / "policy_schedule.json").read_text(encoding="utf-8")
)
POLICY_UPDATE_HZ = (
    float(POLICY_SCHEDULE["radar_frame_rate_hz"])
    / int(POLICY_SCHEDULE["policy_capture_interval_frames"])
)
POLICY_AVAILABILITY_DELAY_FRAMES = int(
    POLICY_SCHEDULE["policy_availability_delay_frames"]
)
POLICY_HOLD_FRAMES = int(POLICY_SCHEDULE["policy_hold_frames"])
for project_path in (PROJECT_ROOT, SOURCE_ROOT):
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))

from radar_bev.evaluation import (
    evaluate_grouped_single_person_predictions,
    evaluate_single_person_predictions,
)
from radar_bev.models.backbones import DREAMBEV, extract_dream_bev_state
from radar_bev.protocol import (
    DEV_SPLIT,
    TEST_SPLIT,
    validate_locked_test_manifest,
    validate_training_manifest,
)
from rvfusion.acceptance_sweep import apply_acceptance_threshold
from rvfusion.adapters import FrozenRadarAdapter
from rvfusion.bearing import (
    CausalVisualBearingEncoder,
    aggregate_top_edge_anchor_geometry,
)
from rvfusion.contracts import (
    ConditionPrior,
    TemporalBearingEvidence,
    VisualEvidence,
)
from rvfusion.data import (
    ConditionTimeline,
    RVFusionCachedDataset,
    SequenceGroupedSampler,
)
from rvfusion.evaluation_oracles import (
    apply_beneficial_candidate_oracle,
    apply_beneficial_candidate_set_oracle,
)
from rvfusion.radar_health import persistent_uncertainty_state
from rvfusion.rv_bearing_cache import bearing_from_collated_cache
from rvfusion.radar_route_experts import (
    RadarRouteExpertPaths,
    build_frozen_route_experts,
)
from rvfusion.routed_radar_evidence import RoutedRadarEvidenceAdapter
from rvfusion.lattice_fusion import DenseLatticeRVFusion
from rvfusion.lattice_losses import (
    LatticeLossWeights,
    dense_lattice_loss,
)
from rvfusion.quantized_lateral_loss import quantized_lateral_residual_loss
from rvfusion.quantized_lateral_residual import QuantizedLateralResidual
from rvfusion.metric_contract import (
    frozen_radar_confidence,
    resolved_radar_position,
)
from rvfusion.training import (
    condition_from_collated,
    safe_checkpoint_decision,
    safe_checkpoint_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate the Radar-H-forward-coordinate-preserving "
            "radar-centric fusion model."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("candidate", "continuous", "gain", "prior_gain", "joint"),
        required=True,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-cache-root", type=Path, required=True)
    parser.add_argument("--radar-health-cache-root", type=Path, required=True)
    parser.add_argument(
        "--condition-zip",
        type=Path,
        help=(
            "Condition annotation package. Required only when Train or "
            "Development policy predictions are not supplied."
        ),
    )
    parser.add_argument(
        "--train-condition-members",
        nargs="+",
        default=["annotations/train_relative.jsonl"],
    )
    parser.add_argument(
        "--train-condition-predictions",
        type=Path,
        help=(
            "Optional frame-level causal policy predictions for Train. When set, "
            "they replace annotation targets so the fusion control uses the same "
            "policy realization as the deployed hierarchy."
        ),
    )
    parser.add_argument(
        "--eval-condition-member",
        default="annotations/development_relative.jsonl",
    )
    parser.add_argument("--eval-condition-predictions", type=Path)
    parser.add_argument("--allow-oracle-eval-condition", action="store_true")
    parser.add_argument("--disable-vlm-cap", action="store_true")
    parser.add_argument(
        "--vlm-trust-policy",
        choices=(
            "soft",
            "strict_usable",
            "normal_nonunreliable",
            "normal_or_rain_nonunreliable",
        ),
        default="soft",
        help=(
            "soft retains bounded cautious admission; strict_usable "
            "requires usable normal RGB; normal_nonunreliable also admits "
            "cautious normal RGB while rejecting every degraded condition; "
            "normal_or_rain_nonunreliable additionally admits cautious "
            "day-rain/wet RGB while unreliable low-light remains rejected."
        ),
    )
    parser.add_argument(
        "--policy-update-hz",
        type=float,
        default=POLICY_UPDATE_HZ,
        help="Causal policy update rate represented by eval-condition-predictions.",
    )
    parser.add_argument(
        "--policy-hold-frames",
        type=int,
        default=POLICY_HOLD_FRAMES,
        help="Number of radar frames holding one causal policy update.",
    )
    parser.add_argument(
        "--policy-availability-delay-frames",
        type=int,
        default=POLICY_AVAILABILITY_DELAY_FRAMES,
        help="Measured policy completion delay before a held interval can begin.",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--bearing-source",
        choices=("detector_anchor", "rv_bearing_cache"),
        default="detector_anchor",
    )
    parser.add_argument("--bearing-checkpoint", type=Path)
    parser.add_argument("--rv-bearing-cache-root", type=Path)
    parser.add_argument("--expert-open-indoor-los", type=Path)
    parser.add_argument("--expert-indoor-clutter", type=Path)
    parser.add_argument("--expert-open-outdoor-weak-return", type=Path)
    parser.add_argument("--expert-outdoor-dry-complex", type=Path)
    parser.add_argument("--expert-outdoor-wet-return", type=Path)
    parser.add_argument("--gg-recovery-scale", type=float, default=1.15)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--radar-clip-length", type=int, default=16)
    parser.add_argument("--visual-clip-length", type=int, default=8)
    parser.add_argument("--correction-radius-m", type=float, default=0.5)
    parser.add_argument("--lattice-step-m", type=float, default=0.025)
    parser.add_argument("--acceptance-threshold", type=float, default=0.5)
    parser.add_argument(
        "--cross-modal-guard-z-threshold",
        type=float,
        help=(
            "Optional fast standardized radar--vision innovation limit. "
            "When exceeded, the deployed output exactly falls back to "
            "expert-radar identity. Calibrate this value on Train only."
        ),
    )
    parser.add_argument(
        "--acceptance-threshold-sweep",
        nargs="*",
        type=float,
        default=[],
    )
    parser.add_argument(
        "--acceptance-mode",
        choices=("factorized", "direct_utility", "candidate_utility"),
        default="factorized",
    )
    parser.add_argument(
        "--correction-source",
        choices=("lattice", "continuous"),
        default="lattice",
    )
    parser.add_argument(
        "--fusion-core",
        choices=(
            "local_posterior",
            "quantized_lateral_residual",
        ),
        default="local_posterior",
        help=(
            "local_posterior uses candidate-local evidence; "
            "quantized_lateral_residual uses fixed local bins without local crops."
        ),
    )
    parser.add_argument(
        "--disable-posterior-no-op",
        action="store_true",
        help=(
            "Development-only frozen local-posterior operator study: force the "
            "non-no-op posterior branch during evaluation. This option is "
            "forbidden for training and Locked Test execution."
        ),
    )
    parser.add_argument(
        "--candidate-geometry",
        choices=(
            "dense_local",
            "measured_peaks",
            "visual_blend",
            "visual_blend_route",
            "visual_blend_route_dense",
        ),
        default="dense_local",
    )
    parser.add_argument(
        "--candidate-prior-mode",
        choices=("none", "route_blend", "bearing_likelihood"),
        default="none",
    )
    parser.add_argument(
        "--candidate-utility-source",
        choices=("token", "scalar"),
        default="token",
    )
    parser.add_argument(
        "--candidate-prior-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--posterior-emission-mode",
        choices=("shared", "route_residual", "route_finetune"),
        default="shared",
    )
    parser.add_argument(
        "--posterior-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--continuous-mode",
        choices=(
            "free_offset",
            "posterior_mean",
            "route_blend_residual",
            "route_blend_offset_residual",
        ),
        default="free_offset",
    )
    parser.add_argument(
        "--continuous-context-mode",
        choices=("scalar", "rich"),
        default="scalar",
    )
    parser.add_argument(
        "--posterior-offset-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--assist-policy",
        choices=("radar_uncertainty", "vlm_segment"),
        default="vlm_segment",
    )
    parser.add_argument("--regret-weight", type=float, default=8.0)
    parser.add_argument("--candidate-positive-weight", type=float, default=10.0)
    parser.add_argument("--candidate-presence-weight", type=float, default=1.0)
    parser.add_argument(
        "--nearest-x-background-weight",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--candidate-target-mode",
        choices=("utility", "nearest_x"),
        default="utility",
    )
    parser.add_argument("--candidate-target-sigma-m", type=float, default=0.075)
    parser.add_argument(
        "--disable-visual-anchor-geometry",
        action="store_true",
        help=(
            "Keep the frozen causal bearing evidence but zero the duplicate "
            "current-box anchor geometry supplied to the fusion model."
        ),
    )
    parser.add_argument(
        "--continuous-location-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the bounded continuous-x regression head during "
            "candidate or joint training."
        ),
    )
    parser.add_argument(
        "--continuous-positive-weight",
        type=float,
        default=1.0,
        help=(
            "Extra continuous-x supervision weight for frames where the "
            "frozen radar/expert prediction misses AP@0.30."
        ),
    )
    parser.add_argument(
        "--continuous-ap30-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the differentiable AP@0.30 crossing and "
            "protection margin."
        ),
    )
    parser.add_argument(
        "--expert-bypass-probability",
        type=float,
        default=0.0,
        help=(
            "Training-only probability of using the frozen base radar "
            "evidence instead of a route expert."
        ),
    )
    parser.add_argument(
        "--expert-bypass-routes",
        nargs="*",
        default=[],
        help=(
            "Optional route allowlist for training-only expert bypass; "
            "empty means every route that has an expert."
        ),
    )
    parser.add_argument("--gain-positive-weight", type=float, default=6.0)
    parser.add_argument("--harm-positive-weight", type=float, default=10.0)
    parser.add_argument(
        "--utility-target-mode",
        choices=("broad", "ap30_crossing", "decomposed_correctness"),
        default="broad",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sequence-cache-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--locked-test-release-record", type=Path)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Reject Locked Test and evaluation-only paths for mechanism studies.",
    )
    parser.add_argument(
        "--reference-split",
        choices=(DEV_SPLIT, "Train", TEST_SPLIT),
        default=DEV_SPLIT,
    )
    parser.add_argument("--save-acceptance-audit", action="store_true")
    parser.add_argument(
        "--eval-exclude-sequences",
        nargs="*",
        default=[],
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_train_condition_timeline(
    *,
    condition_zip: Path | None,
    train_condition_members: list[str],
    train_condition_predictions: Path | None,
) -> tuple[ConditionTimeline, str]:
    """Load Train conditions from deployed policy predictions when available."""
    if train_condition_predictions is not None:
        return (
            ConditionTimeline.from_predictions_jsonl(
                [train_condition_predictions]
            ),
            "condition_policy_predictions",
        )
    if condition_zip is None:
        raise ValueError(
            "--condition-zip is required when --train-condition-predictions "
            "is not supplied"
        )
    return (
        ConditionTimeline.from_zip(condition_zip, train_condition_members),
        "annotation_contract",
    )


def describe_vlm_temporal_policy(
    *,
    update_hz: float,
    availability_delay_frames: int,
    hold_frames: int,
    trust_policy: str,
) -> str:
    if update_hz <= 0.0:
        raise ValueError("policy update rate must be positive")
    if hold_frames < 1:
        raise ValueError("policy hold frames must be positive")
    if availability_delay_frames < 1:
        raise ValueError("policy availability delay must be positive")
    return (
        f"{update_hz:.3f}Hz condition available after {availability_delay_frames} radar frames "
        f"and held causally for {hold_frames} radar frames; "
        f"visual correction follows {trust_policy}"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")


def load_radar_adapter(
    path: Path,
    device: torch.device,
    *,
    expert_paths: RadarRouteExpertPaths | None = None,
    gg_recovery_scale: float = 1.15,
) -> RoutedRadarEvidenceAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    backbone = DREAMBEV()
    backbone.load_state_dict(extract_dream_bev_state(payload), strict=True)
    adapter = FrozenRadarAdapter(backbone).to(device)
    if any(parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("DREAM-BEV must remain frozen")
    experts = (
        build_frozen_route_experts(
            base_checkpoint=path,
            paths=expert_paths,
            device=device,
            gg_recovery_scale=gg_recovery_scale,
        )
        if expert_paths is not None
        else {}
    )
    return RoutedRadarEvidenceAdapter(adapter, experts).to(device)


def route_expert_paths_from_args(
    args: argparse.Namespace,
) -> RadarRouteExpertPaths | None:
    values = (
        args.expert_open_indoor_los,
        args.expert_indoor_clutter,
        args.expert_open_outdoor_weak_return,
        args.expert_outdoor_dry_complex,
        args.expert_outdoor_wet_return,
    )
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError("all five route expert checkpoints must be supplied together")
    return RadarRouteExpertPaths(
        open_indoor_los=args.expert_open_indoor_los,
        indoor_clutter=args.expert_indoor_clutter,
        open_outdoor_weak_return=args.expert_open_outdoor_weak_return,
        outdoor_dry_complex=args.expert_outdoor_dry_complex,
        outdoor_wet_return=args.expert_outdoor_wet_return,
    )


def load_bearing_model(
    path: Path,
    device: torch.device,
) -> CausalVisualBearingEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalVisualBearingEncoder()
    model.load_state_dict(payload["bearing_state"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def visual_from_batch(
    batch: dict[str, object],
    device: torch.device,
) -> VisualEvidence:
    return VisualEvidence(
        boxes=batch["visual_boxes"].to(device, non_blocking=True),  # type: ignore[union-attr]
        raw_scores=batch["visual_raw_scores"].to(device, non_blocking=True),  # type: ignore[union-attr]
        raw_tokens=batch["visual_raw_tokens"].to(device, non_blocking=True),  # type: ignore[union-attr]
        temporal_tokens=batch["visual_temporal_tokens"].to(device, non_blocking=True),  # type: ignore[union-attr]
        temporal_support=batch["visual_temporal_support"].to(device, non_blocking=True),  # type: ignore[union-attr]
        valid_mask=batch["visual_valid_mask"].to(device, non_blocking=True),  # type: ignore[union-attr]
    )


def bearing_from_source(
    batch: dict[str, object],
    *,
    device: torch.device,
    bearing_model: CausalVisualBearingEncoder | None,
    visual: VisualEvidence | None,
) -> TemporalBearingEvidence:
    if "rv_bearing_theta" in batch:
        return bearing_from_collated_cache(batch, device)
    if bearing_model is None or visual is None:
        raise RuntimeError(
            "uncached bearing evidence requires a bearing model and visual cache"
        )
    return bearing_model(visual)


def visual_anchor_from_source(
    batch: dict[str, object],
    *,
    device: torch.device,
    visual: VisualEvidence | None,
) -> Tensor:
    if visual is None:
        visual = visual_from_batch(batch, device)
    return aggregate_top_edge_anchor_geometry(visual)


def sample_expert_bypass_mask(
    routes: tuple[str, ...] | list[str],
    *,
    probability: float,
    device: torch.device,
    allowed_routes: tuple[str, ...] = (),
) -> Tensor | None:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    if probability == 0.0:
        return None
    allowed = {
        str(route).strip().lower() for route in allowed_routes
    }
    eligible = torch.tensor(
        [
            not allowed
            or str(route).strip().lower() in allowed
            for route in routes
        ],
        device=device,
        dtype=torch.bool,
    )
    return (torch.rand(len(routes), device=device) < probability) & eligible


def phase_loss_weights(
    phase: str,
    *,
    regret_weight: float,
    acceptance_mode: str,
    continuous_location_weight: float = 0.0,
    continuous_ap30_margin_weight: float = 0.0,
) -> LatticeLossWeights:
    if continuous_location_weight < 0.0:
        raise ValueError("continuous_location_weight must be non-negative")
    if continuous_ap30_margin_weight < 0.0:
        raise ValueError(
            "continuous_ap30_margin_weight must be non-negative"
        )
    if phase == "candidate":
        return LatticeLossWeights(
            candidate=1.0,
            residual=0.25,
            continuous_location=continuous_location_weight,
            continuous_ap30_margin=continuous_ap30_margin_weight,
            mixture_nll=0.40,
            gain=0.0,
            harm=0.0,
            utility=0.0,
            regret=0.0,
            identity=10.0,
        )
    if phase == "continuous":
        if continuous_location_weight <= 0.0:
            raise ValueError(
                "continuous phase requires a positive "
                "continuous_location_weight"
            )
        return LatticeLossWeights(
            candidate=0.0,
            residual=0.0,
            continuous_location=continuous_location_weight,
            continuous_ap30_margin=continuous_ap30_margin_weight,
            mixture_nll=0.0,
            gain=0.0,
            harm=0.0,
            utility=0.0,
            regret=0.0,
            identity=0.0,
        )
    if phase in {"gain", "prior_gain"}:
        direct = acceptance_mode == "direct_utility"
        candidate_direct = acceptance_mode == "candidate_utility"
        return LatticeLossWeights(
            candidate=0.0,
            residual=0.0,
            mixture_nll=0.0,
            gain=0.0 if direct or candidate_direct else 1.0,
            harm=0.0 if direct or candidate_direct else 2.0,
            utility=1.0 if direct else 0.0,
            candidate_utility=1.0 if candidate_direct else 0.0,
            candidate_utility_rank=(
                1.0
                if candidate_direct
                else 0.0
            ),
            regret=regret_weight,
            identity=10.0,
        )
    direct = acceptance_mode == "direct_utility"
    candidate_direct = acceptance_mode == "candidate_utility"
    return LatticeLossWeights(
        candidate=0.5,
        residual=0.15,
        continuous_location=continuous_location_weight,
        continuous_ap30_margin=continuous_ap30_margin_weight,
        mixture_nll=0.30,
        gain=0.0 if direct or candidate_direct else 1.0,
        harm=0.0 if direct or candidate_direct else 2.0,
        utility=1.0 if direct else 0.0,
        candidate_utility=1.0 if candidate_direct else 0.0,
        regret=regret_weight,
        identity=10.0,
    )


def configure_phase(
    model: DenseLatticeRVFusion | SimpleLateralResidual,
    phase: str,
    acceptance_mode: str,
) -> None:
    def set_trainable(module: torch.nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    set_trainable(model, phase not in {"continuous", "gain", "prior_gain"})
    if phase == "candidate":
        set_trainable(model.gain_head, False)
        set_trainable(model.harm_head, False)
        set_trainable(model.utility_head, False)
        set_trainable(model.selected_utility_context, False)
        set_trainable(model.selected_utility_head, False)
        set_trainable(model.route_utility_heads, False)
        set_trainable(model.candidate_utility_head, False)
        set_trainable(model.candidate_scalar_utility_head, False)
        set_trainable(model.route_blend_residual_head, False)
        set_trainable(model.acceptance_context, False)
    elif phase == "continuous":
        if model.continuous_mode == "posterior_mean":
            if model.posterior_emission_mode == "route_finetune":
                for module in (
                    model.condition_encoder,
                    model.bev_projection,
                    model.polar_projection,
                    model.visual_projection,
                    model.scalar_projection,
                    model.token_projection,
                    model.candidate_attention,
                    model.attention_norm,
                    model.feed_forward,
                    model.feed_forward_norm,
                    model.emission_head,
                    model.posterior_route_emission_heads,
                    model.candidate_presence_head,
                ):
                    set_trainable(module, True)
            elif model.posterior_emission_mode == "route_residual":
                set_trainable(
                    model.posterior_route_emission_heads,
                    True,
                )
            else:
                set_trainable(model.emission_head, True)
            set_trainable(model.candidate_presence_head, True)
        elif model.continuous_mode in {
            "route_blend_residual",
            "route_blend_offset_residual",
        }:
            set_trainable(model.route_blend_residual_head, True)
            if model.continuous_context_mode == "rich":
                set_trainable(model.continuous_location_context, True)
        else:
            set_trainable(model.continuous_location_context, True)
            set_trainable(model.continuous_offset_head, True)
    elif phase == "gain":
        if acceptance_mode == "direct_utility":
            set_trainable(model.acceptance_context, True)
            set_trainable(model.utility_head, True)
            set_trainable(model.selected_utility_context, True)
            set_trainable(model.selected_utility_head, True)
            set_trainable(model.route_utility_heads, True)
        elif acceptance_mode == "candidate_utility":
            if (
                model.candidate_utility_source == "scalar"
                or model.candidate_prior_mode == "route_blend"
            ):
                set_trainable(model.candidate_scalar_utility_head, True)
            else:
                set_trainable(model.candidate_utility_head, True)
        else:
            set_trainable(model.acceptance_context, True)
            set_trainable(model.gain_head, True)
            set_trainable(model.harm_head, True)
    elif phase == "prior_gain":
        set_trainable(model.candidate_scalar_utility_head, True)
    if model.candidate_geometry in {
        "visual_blend",
        "visual_blend_route",
        "visual_blend_route_dense",
    }:
        set_trainable(model.visual_calibrator, False)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable:
        raise RuntimeError(f"phase {phase} has no trainable parameters")


def configure_quantized_lateral_residual(model: QuantizedLateralResidual) -> None:
    """Train the shared-context quantized residual end-to-end."""

    for parameter in model.parameters():
        parameter.requires_grad = True


def _metric_bundle(
    scores: list[Tensor],
    predictions: list[Tensor],
    targets: list[Tensor],
    routes: list[str],
) -> dict[str, object]:
    score = torch.cat(scores)
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    result: dict[str, object] = evaluate_single_person_predictions(
        score,
        prediction,
        target,
        distance_thresholds=(0.25, 0.30, 0.40, 0.50),
    )
    result["per_route"] = evaluate_grouped_single_person_predictions(
        score,
        prediction,
        target,
        routes,
        distance_thresholds=(0.25, 0.30, 0.40, 0.50),
    )
    error = prediction - target
    result.update(
        {
            "x_mae_m": float(error[:, 0].abs().mean()),
            "x_p90_m": float(torch.quantile(error[:, 0].abs(), 0.9)),
            "y_mae_m": float(error[:, 1].abs().mean()),
            "y_p90_m": float(torch.quantile(error[:, 1].abs(), 0.9)),
        }
    )
    return result


def run_epoch(
    *,
    model: DenseLatticeRVFusion | QuantizedLateralResidual,
    radar_adapter: RoutedRadarEvidenceAdapter,
    bearing_model: CausalVisualBearingEncoder | None,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_weights: LatticeLossWeights,
    disable_vlm_cap: bool,
    amp: bool,
    max_batches: int,
    candidate_positive_weight: float = 10.0,
    candidate_presence_weight: float = 1.0,
    nearest_x_background_weight: float = 0.1,
    gain_positive_weight: float = 6.0,
    harm_positive_weight: float = 10.0,
    candidate_target_mode: str = "utility",
    candidate_target_sigma_m: float = 0.075,
    continuous_positive_weight: float = 1.0,
    disable_visual_anchor_geometry: bool = False,
    utility_target_mode: str = "broad",
    acceptance_thresholds: tuple[float, ...] = (),
    cross_modal_guard_z_threshold: float | None = None,
    disable_posterior_no_op: bool = False,
    collect_acceptance_audit: bool = False,
    expert_bypass_probability: float = 0.0,
    expert_bypass_routes: tuple[str, ...] = (),
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    radar_adapter.eval()
    radar_adapter.reset_causal_state()
    if bearing_model is not None:
        bearing_model.eval()
    totals: dict[str, float] = {}
    samples = 0
    fused_scores: list[Tensor] = []
    fused_predictions: list[Tensor] = []
    radar_scores: list[Tensor] = []
    radar_predictions: list[Tensor] = []
    presence_oracle_predictions: list[Tensor] = []
    location_oracle_predictions: list[Tensor] = []
    soft_location_oracle_predictions: list[Tensor] = []
    continuous_location_oracle_predictions: list[Tensor] = []
    learned_location_union_oracle_predictions: list[Tensor] = []
    immutable_y_contract_ceiling_predictions: list[Tensor] = []
    bounded_x_contract_oracle_predictions: dict[float, list[Tensor]] = {
        radius_m: [] for radius_m in (0.5, 0.75, 1.0)
    }
    selected_candidate_predictions: list[Tensor] = []
    effective_acceptances: list[Tensor] = []
    identity_masks: list[Tensor] = []
    selected_non_noop_masks: list[Tensor] = []
    acceptance_audit: list[dict[str, object]] = []
    targets: list[Tensor] = []
    routes: list[str] = []
    fallback_frames = 0
    active_gate_sum = 0.0
    regret_frames = 0
    y_mismatch_frames = 0
    presence_oracle_accepted = 0
    location_oracle_accepted = 0
    soft_location_oracle_accepted = 0
    continuous_location_oracle_accepted = 0
    learned_location_union_oracle_accepted = 0
    if not 0.0 <= expert_bypass_probability <= 1.0:
        raise ValueError(
            "expert_bypass_probability must be in [0,1]"
        )

    for batch_index, raw_batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = dict(raw_batch)
        target_xy_m = batch["target_xy_m"].to(device, non_blocking=True)  # type: ignore[union-attr]
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)  # type: ignore[union-attr]
        radar_window = batch["radar"].to(device, non_blocking=True)  # type: ignore[union-attr]
        health_history = batch["radar_health_history"].to(device, non_blocking=True)  # type: ignore[union-attr]
        history_frame_ids = batch["history_frame_ids"].to(device, non_blocking=True)  # type: ignore[union-attr]
        visual = (
            None
            if "rv_bearing_theta" in batch
            else visual_from_batch(batch, device)
        )
        condition = condition_from_collated(batch["condition"])  # type: ignore[arg-type]
        sequences = tuple(str(value) for value in batch["sequence"])
        current_frame_ids = tuple(
            int(value)
            for value in history_frame_ids[:, -1].detach().cpu().tolist()
        )
        state = persistent_uncertainty_state(
            health_history.float(),
            frame_ids=history_frame_ids,
        )
        visual_anchor_geometry = visual_anchor_from_source(
            batch,
            device=device,
            visual=visual,
        )
        if disable_visual_anchor_geometry:
            visual_anchor_geometry = torch.zeros_like(
                visual_anchor_geometry
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
        expert_bypass_mask = sample_expert_bypass_mask(
            list(condition.global_route),
            probability=(
                expert_bypass_probability if training else 0.0
            ),
            device=device,
            allowed_routes=expert_bypass_routes,
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            radar = radar_adapter(
                radar_window,
                condition.global_route,
                expert_bypass_mask=expert_bypass_mask,
                sequences=sequences,
                frame_ids=current_frame_ids,
            )
            bearing = bearing_from_source(
                batch,
                device=device,
                bearing_model=bearing_model,
                visual=visual,
            )
        frozen_radar_logits = radar.logits.detach().clone()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            forward_kwargs: dict[str, object] = {
                "visual_anchor_geometry": visual_anchor_geometry,
                "visual_trust_cap_override": (
                    torch.ones(
                        target_xy_m.shape[0],
                        1,
                        device=device,
                    )
                    if disable_vlm_cap
                    else None
                ),
                "cross_modal_guard_z_threshold": (
                    cross_modal_guard_z_threshold
                ),
            }
            if not getattr(model, "is_quantized_lateral_residual", False):
                forward_kwargs["disable_no_op"] = disable_posterior_no_op
            output = model(
                radar,
                bearing,
                condition,
                state,
                **forward_kwargs,
            )
            if getattr(model, "is_quantized_lateral_residual", False):
                loss, terms = quantized_lateral_residual_loss(
                    output,
                    target_xy_m,
                    sample_weight,
                    radius_m=model.radius_m,
                    step_m=model.step_m,
                )
            else:
                loss, terms = dense_lattice_loss(
                    output,
                    target_xy_m,
                    sample_weight,
                    weights=loss_weights,
                    candidate_positive_weight=candidate_positive_weight,
                    candidate_presence_weight=candidate_presence_weight,
                    nearest_x_background_weight=nearest_x_background_weight,
                    gain_positive_weight=gain_positive_weight,
                    harm_positive_weight=harm_positive_weight,
                    candidate_target_mode=candidate_target_mode,
                    candidate_target_sigma_m=candidate_target_sigma_m,
                    utility_target_mode=utility_target_mode,
                    continuous_target_mode=(
                        "radar_radius"
                        if model.continuous_mode
                        == "route_blend_offset_residual"
                        else "candidate_span"
                    ),
                    continuous_radius_m=model.radius_m,
                    continuous_positive_weight=continuous_positive_weight,
                )
        if not torch.equal(radar.logits, frozen_radar_logits):
            raise RuntimeError("fusion mutated the frozen DREAM-BEV logits in place")
        canonical_radar_xy_m = resolved_radar_position(radar)
        if not torch.equal(output.radar_xy_m.float(), canonical_radar_xy_m):
            maximum_delta = float(
                (
                    output.radar_xy_m.float() - canonical_radar_xy_m
                ).abs().max()
            )
            raise RuntimeError(
                "fusion changed the frozen DREAM-BEV peak contract: "
                f"max_delta_m={maximum_delta}"
            )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()

        count = int(target_xy_m.shape[0])
        samples += count
        for name, value in {"loss": loss, **terms}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * count
        radar_confidence = frozen_radar_confidence(frozen_radar_logits)
        fused_scores.append(radar_confidence.detach().cpu())
        fused_predictions.append(output.decoded_xy_m.detach().cpu())
        radar_scores.append(radar_confidence.detach().cpu())
        radar_predictions.append(canonical_radar_xy_m.detach().cpu())
        row_index = torch.arange(count, device=device)
        presence_candidate_xy = torch.stack(
            (
                output.correction_x_m[:, 0],
                output.radar_xy_m[:, 1],
            ),
            dim=1,
        )
        selected_candidate_predictions.append(
            presence_candidate_xy.detach().cpu()
        )
        effective_acceptances.append(
            output.effective_acceptance[:, 0].detach().cpu()
        )
        identity_masks.append(output.identity_mask.detach().cpu())
        selected_non_noop_masks.append(
            output.correction_non_noop.detach().cpu()
        )
        if collect_acceptance_audit:
            radar_score_batch = radar_confidence
            candidate_score_batch = radar_confidence
            radar_error_batch = torch.linalg.vector_norm(
                output.radar_xy_m - target_xy_m,
                dim=1,
            )
            candidate_error_batch = torch.linalg.vector_norm(
                presence_candidate_xy - target_xy_m,
                dim=1,
            )
            fused_error_batch = torch.linalg.vector_norm(
                output.decoded_xy_m - target_xy_m,
                dim=1,
            )
            row_index = torch.arange(
                count,
                device=output.selected_index.device,
            )
            selected_posterior = output.posterior[
                row_index,
                output.correction_index,
            ]
            selected_candidate_utility = (
                output.candidate_utility_probability[
                    row_index,
                    output.correction_index,
                ]
            )
            selected_sigma_m = output.candidate_log_sigma_m[
                row_index,
                output.correction_index,
            ].exp()
            posterior_top2 = output.posterior.topk(
                k=min(2, output.posterior.shape[1]),
                dim=1,
            ).values
            posterior_margin = (
                posterior_top2[:, 0] - posterior_top2[:, 1]
                if posterior_top2.shape[1] == 2
                else posterior_top2[:, 0]
            )
            posterior_entropy = -(
                output.posterior
                * output.posterior.clamp_min(1e-12).log()
            ).sum(dim=1)
            audit_tensors = {
                "target": target_xy_m.detach().cpu(),
                "radar_xy": output.radar_xy_m.detach().cpu(),
                "candidate_xy": presence_candidate_xy.detach().cpu(),
                "fused_xy": output.decoded_xy_m.detach().cpu(),
                "radar_score": radar_score_batch.detach().cpu(),
                "candidate_score": candidate_score_batch.detach().cpu(),
                "acceptance": output.effective_acceptance[:, 0]
                .detach()
                .cpu(),
                "utility": output.predicted_utility_probability[:, 0]
                .detach()
                .cpu(),
                "gain": output.predicted_gain[:, 0].detach().cpu(),
                "harm": output.predicted_harm[:, 0].detach().cpu(),
                "presence": output.candidate_presence_probability[:, 0]
                .detach()
                .cpu(),
                "selected_candidate_utility": (
                    selected_candidate_utility.detach().cpu()
                ),
                "selected_posterior": selected_posterior.detach().cpu(),
                "selected_sigma_m": selected_sigma_m.detach().cpu(),
                "posterior_margin": posterior_margin.detach().cpu(),
                "posterior_entropy": posterior_entropy.detach().cpu(),
                "selected_index": output.selected_index.detach().cpu(),
                "correction_index": (
                    output.correction_index.detach().cpu()
                ),
                "identity": output.identity_mask.detach().cpu(),
                "accepted": output.accepted.detach().cpu(),
                "visual_x": output.calibrated_visual_x_m[:, 0]
                .detach()
                .cpu(),
                "continuous_x": output.continuous_x_m[:, 0]
                .detach()
                .cpu(),
                "bearing_theta": bearing.theta[:, 0].detach().cpu(),
                "bearing_sigma": bearing.sigma_theta[:, 0]
                .detach()
                .cpu(),
                "bearing_reliability": bearing.reliability[:, 0]
                .detach()
                .cpu(),
                "bearing_valid": bearing.valid.detach().cpu(),
                "radar_sigma_x_m": output.radar_sigma_x_m[:, 0]
                .detach()
                .cpu(),
                "cross_modal_innovation": output.cross_modal_innovation[
                    :, 0
                ]
                .detach()
                .cpu(),
                "cross_modal_guard_passed": (
                    output.cross_modal_guard_passed.detach().cpu()
                ),
                "assist_enabled": state.assist_enabled.detach().cpu(),
                "uncertainty_strength": (
                    state.uncertainty_strength.detach().cpu()
                ),
                "radar_error": radar_error_batch.detach().cpu(),
                "candidate_error": candidate_error_batch.detach().cpu(),
                "fused_error": fused_error_batch.detach().cpu(),
            }
            sequences = [str(value) for value in batch["sequence"]]
            frame_ids = batch["frame_id"].tolist()  # type: ignore[union-attr]
            for index in range(count):
                target_cpu = audit_tensors["target"]
                radar_xy_cpu = audit_tensors["radar_xy"]
                candidate_xy_cpu = audit_tensors["candidate_xy"]
                fused_xy_cpu = audit_tensors["fused_xy"]
                radar_error_cpu = audit_tensors["radar_error"]
                candidate_error_cpu = audit_tensors["candidate_error"]
                fused_error_cpu = audit_tensors["fused_error"]
                acceptance_audit.append(
                    {
                        "sequence": sequences[index],
                        "frame_id": int(frame_ids[index]),
                        "route": condition.global_route[index],
                        "target_x_m": float(target_cpu[index, 0]),
                        "target_y_m": float(target_cpu[index, 1]),
                        "radar_x_m": float(radar_xy_cpu[index, 0]),
                        "radar_y_m": float(radar_xy_cpu[index, 1]),
                        "candidate_x_m": float(
                            candidate_xy_cpu[index, 0]
                        ),
                        "candidate_y_m": float(
                            candidate_xy_cpu[index, 1]
                        ),
                        "fused_x_m": float(fused_xy_cpu[index, 0]),
                        "fused_y_m": float(fused_xy_cpu[index, 1]),
                        "radar_score": float(
                            audit_tensors["radar_score"][index]
                        ),
                        "candidate_score": float(
                            audit_tensors["candidate_score"][index]
                        ),
                        "effective_acceptance": float(
                            audit_tensors["acceptance"][index]
                        ),
                        "predicted_utility_probability": float(
                            audit_tensors["utility"][index]
                        ),
                        "predicted_gain_probability": float(
                            audit_tensors["gain"][index]
                        ),
                        "predicted_harm_probability": float(
                            audit_tensors["harm"][index]
                        ),
                        "candidate_presence_probability": float(
                            audit_tensors["presence"][index]
                        ),
                        "selected_candidate_utility_probability": float(
                            audit_tensors["selected_candidate_utility"][
                                index
                            ]
                        ),
                        "selected_posterior_probability": float(
                            audit_tensors["selected_posterior"][index]
                        ),
                        "selected_sigma_m": float(
                            audit_tensors["selected_sigma_m"][index]
                        ),
                        "posterior_margin": float(
                            audit_tensors["posterior_margin"][index]
                        ),
                        "posterior_entropy": float(
                            audit_tensors["posterior_entropy"][index]
                        ),
                        "selected_index": int(
                            audit_tensors["selected_index"][index]
                        ),
                        "correction_index": int(
                            audit_tensors["correction_index"][index]
                        ),
                        "identity": bool(audit_tensors["identity"][index]),
                        "accepted": bool(audit_tensors["accepted"][index]),
                        "correction_x_m": float(
                            candidate_xy_cpu[index, 0]
                            - radar_xy_cpu[index, 0]
                        ),
                        "calibrated_visual_x_m": float(
                            audit_tensors["visual_x"][index]
                        ),
                        "continuous_x_m": float(
                            audit_tensors["continuous_x"][index]
                        ),
                        "bearing_theta_rad": float(
                            audit_tensors["bearing_theta"][index]
                        ),
                        "bearing_sigma_rad": float(
                            audit_tensors["bearing_sigma"][index]
                        ),
                        "bearing_reliability": float(
                            audit_tensors["bearing_reliability"][index]
                        ),
                        "bearing_valid": bool(
                            audit_tensors["bearing_valid"][index]
                        ),
                        "radar_sigma_x_m": float(
                            audit_tensors["radar_sigma_x_m"][index]
                        ),
                        "cross_modal_innovation": float(
                            audit_tensors["cross_modal_innovation"][index]
                        ),
                        "cross_modal_guard_passed": bool(
                            audit_tensors["cross_modal_guard_passed"][index]
                        ),
                        "assist_enabled": bool(
                            audit_tensors["assist_enabled"][index]
                        ),
                        "uncertainty_strength": float(
                            audit_tensors["uncertainty_strength"][index]
                        ),
                        "radar_error_m": float(radar_error_cpu[index]),
                        "candidate_error_m": float(
                            candidate_error_cpu[index]
                        ),
                        "fused_error_m": float(fused_error_cpu[index]),
                        "ap30_crossing_gain": bool(
                            radar_error_cpu[index] > 0.30
                            and candidate_error_cpu[index] <= 0.30
                        ),
                        "candidate_beneficial": bool(
                            candidate_error_cpu[index] + 0.02
                            < radar_error_cpu[index]
                        ),
                    }
                )
        location_index = output.emission_logits[:, 1:].argmax(dim=1) + 1
        location_candidate_xy = torch.stack(
            (
                output.candidate_x_m[row_index, location_index],
                output.radar_xy_m[:, 1],
            ),
            dim=1,
        )
        conditional_location_probability = output.emission_logits[
            :, 1:
        ].softmax(dim=1)
        soft_location_x_m = (
            conditional_location_probability
            * output.candidate_x_m[:, 1:]
        ).sum(dim=1)
        soft_location_candidate_xy = torch.stack(
            (soft_location_x_m, output.radar_xy_m[:, 1]),
            dim=1,
        )
        continuous_location_candidate_xy = torch.cat(
            (
                output.continuous_x_m,
                output.radar_xy_m[:, 1:2],
            ),
            dim=1,
        )
        correction_allowed = ~output.identity_mask.detach()
        presence_oracle = apply_beneficial_candidate_oracle(
            output.radar_xy_m.detach(),
            presence_candidate_xy.detach(),
            target_xy_m,
            correction_allowed,
        )
        location_oracle = apply_beneficial_candidate_oracle(
            output.radar_xy_m.detach(),
            location_candidate_xy.detach(),
            target_xy_m,
            correction_allowed,
        )
        soft_location_oracle = apply_beneficial_candidate_oracle(
            output.radar_xy_m.detach(),
            soft_location_candidate_xy.detach(),
            target_xy_m,
            correction_allowed,
        )
        continuous_location_oracle = apply_beneficial_candidate_oracle(
            output.radar_xy_m.detach(),
            continuous_location_candidate_xy.detach(),
            target_xy_m,
            correction_allowed,
        )
        learned_location_union_oracle = (
            apply_beneficial_candidate_set_oracle(
                output.radar_xy_m.detach(),
                torch.stack(
                    (
                        location_candidate_xy,
                        soft_location_candidate_xy,
                        continuous_location_candidate_xy,
                    ),
                    dim=1,
                ).detach(),
                target_xy_m,
                correction_allowed,
            )
        )
        immutable_y_contract_ceiling_xy = output.radar_xy_m.detach().clone()
        immutable_y_contract_ceiling_xy[correction_allowed, 0] = (
            target_xy_m[correction_allowed, 0]
        )
        for radius_m, predictions in (
            bounded_x_contract_oracle_predictions.items()
        ):
            bounded_xy = output.radar_xy_m.detach().clone()
            bounded_target_x = target_xy_m[:, 0].clamp(
                min=output.radar_xy_m[:, 0] - radius_m,
                max=output.radar_xy_m[:, 0] + radius_m,
            )
            bounded_xy[correction_allowed, 0] = bounded_target_x[
                correction_allowed
            ]
            predictions.append(bounded_xy.cpu())
        presence_oracle_predictions.append(
            presence_oracle.decoded_xy_m.cpu()
        )
        location_oracle_predictions.append(
            location_oracle.decoded_xy_m.cpu()
        )
        soft_location_oracle_predictions.append(
            soft_location_oracle.decoded_xy_m.cpu()
        )
        continuous_location_oracle_predictions.append(
            continuous_location_oracle.decoded_xy_m.cpu()
        )
        learned_location_union_oracle_predictions.append(
            learned_location_union_oracle.decoded_xy_m.cpu()
        )
        immutable_y_contract_ceiling_predictions.append(
            immutable_y_contract_ceiling_xy.cpu()
        )
        presence_oracle_accepted += int(presence_oracle.accepted.sum())
        location_oracle_accepted += int(location_oracle.accepted.sum())
        soft_location_oracle_accepted += int(
            soft_location_oracle.accepted.sum()
        )
        continuous_location_oracle_accepted += int(
            continuous_location_oracle.accepted.sum()
        )
        learned_location_union_oracle_accepted += int(
            learned_location_union_oracle.accepted.sum()
        )
        targets.append(target_xy_m.detach().cpu())
        routes.extend(condition.global_route)
        fallback_frames += int((~output.accepted).sum().detach())
        active_gate_sum += float(output.effective_acceptance.sum().detach())
        radar_error = torch.linalg.vector_norm(
            output.radar_xy_m.detach() - target_xy_m,
            dim=1,
        )
        fused_error = torch.linalg.vector_norm(
            output.decoded_xy_m.detach() - target_xy_m,
            dim=1,
        )
        regret_frames += int((fused_error > radar_error + 1e-6).sum())
        y_mismatch_frames += int(
            (output.decoded_xy_m[:, 1] != output.radar_xy_m[:, 1]).sum().detach()
        )
    if samples == 0:
        raise ValueError("epoch produced no samples")
    result: dict[str, object] = {
        "loss": {name: value / samples for name, value in totals.items()},
        "fused": _metric_bundle(
            fused_scores,
            fused_predictions,
            targets,
            routes,
        ),
        "radar": _metric_bundle(
            radar_scores,
            radar_predictions,
            targets,
            routes,
        ),
        "samples": samples,
        "fallback_rate": fallback_frames / samples,
        "mean_effective_acceptance": active_gate_sum / samples,
        "accepted_rate": 1.0 - fallback_frames / samples,
        "regret_frame_rate": regret_frames / samples,
        "decoded_y_mismatch_frames": y_mismatch_frames,
        "oracle_diagnostics": {
            "presence_plus_location_oracle_gate": {
                "metrics": _metric_bundle(
                    radar_scores,
                    presence_oracle_predictions,
                    targets,
                    routes,
                ),
                "accepted_rate": presence_oracle_accepted / samples,
            },
            "location_head_oracle_gate": {
                "metrics": _metric_bundle(
                    radar_scores,
                    location_oracle_predictions,
                    targets,
                    routes,
                ),
                "accepted_rate": location_oracle_accepted / samples,
            },
            "soft_location_head_oracle_gate": {
                "metrics": _metric_bundle(
                    radar_scores,
                    soft_location_oracle_predictions,
                    targets,
                    routes,
                ),
                "accepted_rate": soft_location_oracle_accepted / samples,
            },
            "continuous_location_head_oracle_gate": {
                "metrics": _metric_bundle(
                    radar_scores,
                    continuous_location_oracle_predictions,
                    targets,
                    routes,
                ),
                "accepted_rate": (
                    continuous_location_oracle_accepted / samples
                ),
            },
            "learned_location_union_oracle_gate": {
                "metrics": _metric_bundle(
                    radar_scores,
                    learned_location_union_oracle_predictions,
                    targets,
                    routes,
                ),
                "accepted_rate": (
                    learned_location_union_oracle_accepted / samples
                ),
            },
            "immutable_y_contract_oracle_ceiling": {
                "metrics": _metric_bundle(
                    radar_scores,
                    immutable_y_contract_ceiling_predictions,
                    targets,
                    routes,
                ),
                "diagnostic_only": True,
                "description": (
                    "GT x on assist-eligible routes with frozen radar y; "
                    "corner_nlos remains exact identity."
                ),
            },
            "bounded_x_contract_oracle_sweep": {
                f"{radius_m:.2f}m": {
                    "metrics": _metric_bundle(
                        radar_scores,
                        predictions,
                        targets,
                        routes,
                    ),
                    "diagnostic_only": True,
                }
                for radius_m, predictions in (
                    bounded_x_contract_oracle_predictions.items()
                )
            },
        },
    }
    if acceptance_thresholds:
        radar_score = torch.cat(radar_scores)
        candidate_score = torch.cat(fused_scores)
        radar_prediction = torch.cat(radar_predictions)
        candidate_prediction = torch.cat(selected_candidate_predictions)
        target = torch.cat(targets)
        effective_acceptance = torch.cat(effective_acceptances)
        identity_mask = torch.cat(identity_masks)
        selected_non_noop = torch.cat(selected_non_noop_masks)
        sweep: dict[str, object] = {}
        for threshold in acceptance_thresholds:
            threshold_output = apply_acceptance_threshold(
                radar_xy_m=radar_prediction,
                candidate_xy_m=candidate_prediction,
                radar_score=radar_score,
                candidate_score=candidate_score,
                effective_acceptance=effective_acceptance,
                identity_mask=identity_mask,
                selected_non_noop=selected_non_noop,
                threshold=threshold,
            )
            metrics = _metric_bundle(
                [threshold_output.score],
                [threshold_output.prediction_xy_m],
                [target],
                routes,
            )
            metrics["accepted_rate"] = float(
                threshold_output.accepted.float().mean()
            )
            sweep[f"{float(threshold):.6f}"] = metrics
        result["acceptance_threshold_sweep"] = sweep
    if collect_acceptance_audit:
        result["acceptance_audit"] = acceptance_audit
    return result


def _initialize_lazy_model(
    *,
    model: DenseLatticeRVFusion | QuantizedLateralResidual,
    radar_adapter: RoutedRadarEvidenceAdapter,
    bearing_model: CausalVisualBearingEncoder | None,
    loader: DataLoader,
    device: torch.device,
    disable_vlm_cap: bool,
    amp: bool,
) -> None:
    batch = dict(next(iter(loader)))
    radar_window = batch["radar"].to(device)  # type: ignore[union-attr]
    health = batch["radar_health_history"].to(device)  # type: ignore[union-attr]
    frame_ids = batch["history_frame_ids"].to(device)  # type: ignore[union-attr]
    visual = (
        None
        if "rv_bearing_theta" in batch
        else visual_from_batch(batch, device)
    )
    condition = condition_from_collated(batch["condition"])  # type: ignore[arg-type]
    state = persistent_uncertainty_state(health.float(), frame_ids=frame_ids)
    visual_anchor_geometry = visual_anchor_from_source(
        batch,
        device=device,
        visual=visual,
    )
    # LazyLinear must materialize ordinary parameters that can later train.
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=amp and device.type == "cuda",
    ):
        radar_adapter.reset_causal_state()
        radar = radar_adapter(
            radar_window,
            condition.global_route,
            sequences=tuple(str(value) for value in batch["sequence"]),
            frame_ids=tuple(
                int(value)
                for value in frame_ids[:, -1].detach().cpu().tolist()
            ),
        )
        bearing = bearing_from_source(
            batch,
            device=device,
            bearing_model=bearing_model,
            visual=visual,
        )
        model(
            radar,
            bearing,
            condition,
            state,
            visual_anchor_geometry=visual_anchor_geometry,
            visual_trust_cap_override=(
                torch.ones(
                    radar_window.shape[0],
                    1,
                    device=device,
                )
                if disable_vlm_cap
                else None
            ),
        )


def _checkpoint_payload(
    *,
    model: DenseLatticeRVFusion | QuantizedLateralResidual,
    epoch: int,
    phase: str,
    metrics: dict[str, object],
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "model": model.__class__.__name__,
        "phase": phase,
        "epoch": epoch,
        "dense_lattice_state": model.state_dict(),
        "metrics": metrics,
        "metadata": metadata,
    }


def checkpoint_allowed_missing_keys(
    model: DenseLatticeRVFusion | QuantizedLateralResidual,
    *,
    skipped_shape_mismatches: set[str],
) -> set[str]:
    """Return optional checkpoint keys supported by the active fusion core."""
    allowed_missing = {
        "utility_head.weight",
        "utility_head.bias",
        "selected_utility_context.0.weight",
        "selected_utility_context.0.bias",
        "selected_utility_context.2.weight",
        "selected_utility_context.2.bias",
        "selected_utility_head.weight",
        "selected_utility_head.bias",
        "candidate_utility_head.0.weight",
        "candidate_utility_head.0.bias",
        "candidate_utility_head.2.weight",
        "candidate_utility_head.2.bias",
        "candidate_scalar_utility_head.0.weight",
        "candidate_scalar_utility_head.0.bias",
        "candidate_scalar_utility_head.2.weight",
        "candidate_scalar_utility_head.2.bias",
        "continuous_location_context.0.weight",
        "continuous_location_context.0.bias",
        "continuous_location_context.2.weight",
        "continuous_location_context.2.bias",
        "continuous_offset_head.0.weight",
        "continuous_offset_head.0.bias",
        "continuous_offset_head.2.weight",
        "continuous_offset_head.2.bias",
        "route_blend_residual_head.0.weight",
        "route_blend_residual_head.0.bias",
        "route_blend_residual_head.2.weight",
        "route_blend_residual_head.2.bias",
        "route_blend_residual_head.3.weight",
        "route_blend_residual_head.3.bias",
    }
    for name in (
        "route_utility_heads",
        "posterior_route_emission_heads",
    ):
        heads = getattr(model, name, ())
        allowed_missing.update(
            f"{name}.{index}.{field}"
            for index in range(len(heads))
            for field in ("weight", "bias")
        )
    allowed_missing.update(skipped_shape_mismatches)
    return allowed_missing


def validate_phase_contract(
    *,
    phase: str,
    initial_checkpoint: Path | None,
    acceptance_mode: str,
    candidate_geometry: str,
    candidate_prior_mode: str,
    continuous_mode: str = "free_offset",
) -> None:
    fresh_route_blend_continuous = (
        phase == "continuous"
        and continuous_mode
        in {"route_blend_residual", "route_blend_offset_residual"}
    )
    if (
        phase in {"continuous", "gain"}
        and initial_checkpoint is None
        and not fresh_route_blend_continuous
    ):
        raise ValueError(
            f"{phase} phase requires --initial-checkpoint"
        )
    if phase != "prior_gain":
        return
    if initial_checkpoint is not None:
        raise ValueError(
            "prior_gain starts from the deterministic route prior and "
            "does not accept --initial-checkpoint"
        )
    if acceptance_mode != "candidate_utility":
        raise ValueError(
            "prior_gain requires --acceptance-mode candidate_utility"
        )
    if candidate_prior_mode != "route_blend":
        raise ValueError(
            "prior_gain requires --candidate-prior-mode route_blend"
        )
    if candidate_geometry not in {
        "visual_blend_route",
        "visual_blend_route_dense",
    }:
        raise ValueError(
            "prior_gain requires visual_blend_route or "
            "visual_blend_route_dense candidate geometry"
        )


def validate_reference_split(
    reference_split: str,
    *,
    evaluation_only: bool,
    locked_test_release_record: Path | None = None,
) -> None:
    if reference_split not in {DEV_SPLIT, "Train", TEST_SPLIT}:
        raise ValueError(
            f"reference split must be {DEV_SPLIT}, Train, or {TEST_SPLIT}"
        )
    if reference_split == "Train" and not evaluation_only:
        raise ValueError(
            "Train reference split is restricted to evaluation-only audits"
        )
    if reference_split == TEST_SPLIT:
        if not evaluation_only:
            raise ValueError(
                "Locked Test is restricted to evaluation-only execution"
            )
        if locked_test_release_record is None:
            raise ValueError(
                "Locked Test requires an explicit release record"
            )


def validate_development_only_execution(
    *,
    reference_split: str,
    evaluation_only: bool,
    locked_test_release_record: Path | str | None,
) -> None:
    if reference_split != DEV_SPLIT or locked_test_release_record is not None:
        raise ValueError(
            "Development-only execution requires Development Val and forbids "
            "a Locked Test release record"
        )


def validate_posterior_operator_study(
    *,
    fusion_core: str,
    initial_checkpoint: Path | str | None,
    evaluation_only: bool,
    development_only: bool,
    disable_posterior_no_op: bool,
) -> None:
    if not disable_posterior_no_op:
        return
    if (
        fusion_core != "local_posterior"
        or initial_checkpoint is None
        or not evaluation_only
        or not development_only
    ):
        raise ValueError(
            "no-op operator study requires a frozen local_posterior checkpoint "
            "and Development-only evaluation"
        )


def validate_locked_test_release_record(
    path: Path,
    *,
    manifest: Path,
    initial_checkpoint: Path,
) -> None:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != "sequence_locked":
        raise ValueError("locked-test release record has wrong protocol")
    if payload.get("scope") != "base":
        raise ValueError("locked-test release record has wrong scope")
    if payload.get("authorization") != "approved_final_evaluation":
        raise ValueError("locked-test release record is not authorized")
    if payload.get("manifest_sha256") != sha256(manifest):
        raise ValueError("locked-test release manifest hash mismatch")
    if payload.get("initial_checkpoint_sha256") != sha256(
        initial_checkpoint
    ):
        raise ValueError("locked-test release checkpoint hash mismatch")
    with Path(manifest).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        sequences = sorted(
            {row["sequence"].strip() for row in csv.DictReader(handle)}
        )
    if payload.get("allowed_sequences") != sequences:
        raise ValueError(
            "locked-test release sequence list does not match manifest"
        )


def validate_cache_manifest_binding(
    *,
    manifest: Path,
    visual_cache_root: Path,
    bearing_cache_root: Path | None,
    radar_health_cache_root: Path,
) -> None:
    expected = sha256(manifest)
    visual_metadata = sorted(
        visual_cache_root.glob("cache_metadata_shard_*.json")
    )
    if not visual_metadata:
        raise FileNotFoundError(
            f"visual cache metadata not found: {visual_cache_root}"
        )
    visual_shas = {
        str(json.loads(path.read_text(encoding="utf-8")).get("manifest_sha256", ""))
        for path in visual_metadata
    }
    if visual_shas != {expected}:
        raise ValueError(
            "visual cache manifest does not match active canonical manifest"
        )
    if bearing_cache_root is not None:
        bearing_manifest = bearing_cache_root / "cache_manifest.json"
        if not bearing_manifest.is_file():
            raise FileNotFoundError(
                f"RV-Bearing cache manifest not found: {bearing_manifest}"
            )
        bearing_payload = json.loads(
            bearing_manifest.read_text(encoding="utf-8")
        )
        if str(bearing_payload.get("manifest_sha256", "")) != expected:
            raise ValueError(
                "RV-Bearing cache manifest does not match active canonical manifest"
            )
    health_manifest = radar_health_cache_root / "cache_manifest.json"
    if not health_manifest.is_file():
        raise FileNotFoundError(
            f"Radar Health cache manifest not found: {health_manifest}"
        )
    health_payload = json.loads(
        health_manifest.read_text(encoding="utf-8")
    )
    if str(health_payload.get("source_manifest_sha256", "")) != expected:
        raise ValueError(
            "Radar Health cache manifest does not match active canonical manifest"
        )


def validate_initial_checkpoint_manifest_binding(
    payload: object,
    manifest: Path,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError("initial checkpoint must be a mapping")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("sequence-locked initial checkpoint lacks metadata")
    if metadata.get("manifest_sha256") != sha256(manifest):
        raise ValueError(
            "initial checkpoint protocol manifest does not match active manifest"
        )


def filter_compatible_checkpoint_state(
    checkpoint_state: dict[str, Tensor],
    current_state: dict[str, Tensor],
    *,
    allowed_shape_mismatches: set[str],
) -> tuple[dict[str, Tensor], set[str]]:
    """Drop explicitly rebuildable heads while rejecting other shape drift."""

    filtered: dict[str, Tensor] = {}
    skipped: set[str] = set()
    for name, value in checkpoint_state.items():
        current = current_state.get(name)
        if current is not None and tuple(value.shape) != tuple(current.shape):
            if name not in allowed_shape_mismatches:
                raise RuntimeError(
                    "checkpoint shape mismatch for "
                    f"{name}: checkpoint={tuple(value.shape)}, "
                    f"current={tuple(current.shape)}"
                )
            skipped.add(name)
            continue
        filtered[name] = value
    return filtered, skipped


def main() -> None:
    args = parse_args()
    locked_test_evaluation = args.reference_split == TEST_SPLIT
    if locked_test_evaluation:
        validate_locked_test_manifest(args.manifest, scope="base")
    else:
        validate_training_manifest(args.manifest, scope="base")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    if (
        args.bearing_source == "detector_anchor"
        and args.bearing_checkpoint is None
    ):
        raise ValueError(
            "detector_anchor requires --bearing-checkpoint"
        )
    if (
        args.bearing_source == "rv_bearing_cache"
        and args.rv_bearing_cache_root is None
    ):
        raise ValueError(
            "rv_bearing_cache requires --rv-bearing-cache-root"
        )
    validate_cache_manifest_binding(
        manifest=args.manifest,
        visual_cache_root=args.visual_cache_root,
        bearing_cache_root=(
            args.rv_bearing_cache_root
            if args.bearing_source == "rv_bearing_cache"
            else None
        ),
        radar_health_cache_root=args.radar_health_cache_root,
    )
    validate_phase_contract(
        phase=args.phase,
        initial_checkpoint=args.initial_checkpoint,
        acceptance_mode=args.acceptance_mode,
        candidate_geometry=args.candidate_geometry,
        candidate_prior_mode=args.candidate_prior_mode,
        continuous_mode=args.continuous_mode,
    )
    validate_reference_split(
        args.reference_split,
        evaluation_only=args.evaluation_only,
        locked_test_release_record=args.locked_test_release_record,
    )
    if args.development_only:
        validate_development_only_execution(
            reference_split=args.reference_split,
            evaluation_only=args.evaluation_only,
            locked_test_release_record=args.locked_test_release_record,
        )
    validate_posterior_operator_study(
        fusion_core=args.fusion_core,
        initial_checkpoint=args.initial_checkpoint,
        evaluation_only=args.evaluation_only,
        development_only=args.development_only,
        disable_posterior_no_op=args.disable_posterior_no_op,
    )
    if locked_test_evaluation:
        if args.initial_checkpoint is None:
            raise ValueError(
                "Locked Test requires a frozen --initial-checkpoint"
            )
        validate_locked_test_release_record(
            args.locked_test_release_record,
            manifest=args.manifest,
            initial_checkpoint=args.initial_checkpoint,
        )
    if (
        args.eval_condition_predictions is None
        and not args.allow_oracle_eval_condition
    ):
        raise ValueError(
            "formal evaluation requires --eval-condition-predictions; "
            "use --allow-oracle-eval-condition only for smoke tests"
        )
    if args.regret_weight < 0.0:
        raise ValueError("regret weight must be non-negative")
    if args.candidate_prior_scale <= 0.0:
        raise ValueError("candidate prior scale must be positive")
    if not 0.0 < args.posterior_offset_scale <= 1.0:
        raise ValueError(
            "posterior offset scale must be in (0,1]"
        )
    if args.continuous_positive_weight <= 0.0:
        raise ValueError("continuous positive weight must be positive")
    if args.continuous_ap30_margin_weight < 0.0:
        raise ValueError(
            "continuous AP@0.30 margin weight must be non-negative"
        )
    if not 0.0 <= args.expert_bypass_probability <= 1.0:
        raise ValueError(
            "expert bypass probability must be in [0,1]"
        )
    if any(
        threshold < args.acceptance_threshold or threshold > 1.0
        for threshold in args.acceptance_threshold_sweep
    ):
        raise ValueError(
            "acceptance threshold sweep values must be between the "
            "model threshold and 1"
        )
    args.output_dir.mkdir(parents=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_timeline, train_condition_source = resolve_train_condition_timeline(
        condition_zip=args.condition_zip,
        train_condition_members=args.train_condition_members,
        train_condition_predictions=args.train_condition_predictions,
    )
    if args.eval_condition_predictions is None:
        if args.condition_zip is None:
            raise ValueError(
                "--condition-zip is required when --eval-condition-predictions "
                "is not supplied"
            )
        eval_timeline = ConditionTimeline.from_zip(
            args.condition_zip,
            [args.eval_condition_member],
        )
        eval_condition_source = "oracle_smoke_only"
    else:
        eval_timeline = ConditionTimeline.from_predictions_jsonl(
            [args.eval_condition_predictions]
        )
        eval_condition_source = "condition_policy_predictions"

    common_dataset = {
        "data_root": args.data_root,
        "manifest_path": args.manifest,
        "visual_cache_root": args.visual_cache_root,
        "rv_bearing_cache_root": (
            args.rv_bearing_cache_root
            if args.bearing_source == "rv_bearing_cache"
            else None
        ),
        "radar_health_cache_root": args.radar_health_cache_root,
        "radar_clip_length": args.radar_clip_length,
        "visual_clip_length": args.visual_clip_length,
        "sequence_cache_size": args.sequence_cache_size,
        "return_radar": True,
    }
    eval_set = RVFusionCachedDataset(
        condition_timeline=eval_timeline,
        split=(TEST_SPLIT if locked_test_evaluation else DEV_SPLIT),
        excluded_sequences=args.eval_exclude_sequences,
        **common_dataset,
    )
    loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    eval_sampler = SequenceGroupedSampler(
        eval_set,
        seed=args.seed,
        shuffle_sequences=False,
    )
    eval_loader = DataLoader(eval_set, sampler=eval_sampler, **loader_arguments)
    if locked_test_evaluation:
        train_sampler = None
        train_loader = None
        initialization_loader = eval_loader
        reference_loader = eval_loader
    else:
        train_set = RVFusionCachedDataset(
            condition_timeline=train_timeline,
            split="Train",
            **common_dataset,
        )
        train_sampler = SequenceGroupedSampler(
            train_set,
            seed=args.seed,
            shuffle_sequences=True,
        )
        train_loader = DataLoader(
            train_set,
            sampler=train_sampler,
            **loader_arguments,
        )
        initialization_loader = train_loader
        reference_loader = (
            train_loader
            if args.reference_split == "Train"
            else eval_loader
        )

    expert_paths = route_expert_paths_from_args(args)
    radar_adapter = load_radar_adapter(
        args.base_checkpoint,
        device,
        expert_paths=expert_paths,
        gg_recovery_scale=args.gg_recovery_scale,
    )
    bearing_model = (
        load_bearing_model(args.bearing_checkpoint, device)
        if args.bearing_source == "detector_anchor"
        and args.bearing_checkpoint is not None
        else None
    )
    if args.fusion_core == "quantized_lateral_residual":
        model: DenseLatticeRVFusion | QuantizedLateralResidual = QuantizedLateralResidual(
            radius_m=args.correction_radius_m,
            step_m=args.lattice_step_m,
            trust_policy=args.vlm_trust_policy,
        ).to(device)
    else:
        model = DenseLatticeRVFusion(
            radius_m=args.correction_radius_m,
            lattice_step_m=args.lattice_step_m,
            acceptance_threshold=args.acceptance_threshold,
            acceptance_mode=args.acceptance_mode,
            correction_source=args.correction_source,
            candidate_geometry=args.candidate_geometry,
            candidate_prior_mode=args.candidate_prior_mode,
            candidate_utility_source=args.candidate_utility_source,
            candidate_prior_scale=args.candidate_prior_scale,
            posterior_emission_mode=args.posterior_emission_mode,
            posterior_temperature=args.posterior_temperature,
            continuous_mode=args.continuous_mode,
            continuous_context_mode=args.continuous_context_mode,
            posterior_offset_scale=args.posterior_offset_scale,
            visual_trust_policy=args.vlm_trust_policy,
            assist_policy=args.assist_policy,
        ).to(device)
    _initialize_lazy_model(
        model=model,
        radar_adapter=radar_adapter,
        bearing_model=bearing_model,
        loader=initialization_loader,
        device=device,
        disable_vlm_cap=args.disable_vlm_cap,
        amp=not args.no_amp,
    )
    if args.initial_checkpoint is not None:
        payload = torch.load(
            args.initial_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if not locked_test_evaluation:
            validate_initial_checkpoint_manifest_binding(
                payload,
                args.manifest,
            )
        route_blend_keys = {
            "route_blend_residual_head.0.weight",
            "route_blend_residual_head.0.bias",
            "route_blend_residual_head.2.weight",
            "route_blend_residual_head.2.bias",
            "route_blend_residual_head.3.weight",
            "route_blend_residual_head.3.bias",
        }
        checkpoint_state, skipped_shape_mismatches = (
            filter_compatible_checkpoint_state(
                payload["dense_lattice_state"],
                model.state_dict(),
                allowed_shape_mismatches=(
                    route_blend_keys
                    if args.phase == "continuous"
                    and args.continuous_mode
                    in {
                        "route_blend_residual",
                        "route_blend_offset_residual",
                    }
                    else set()
                ),
            )
        )
        incompatible = model.load_state_dict(
            checkpoint_state,
            strict=False,
        )
        allowed_missing = checkpoint_allowed_missing_keys(
            model,
            skipped_shape_mismatches=skipped_shape_mismatches,
        )
        unexpected_missing = set(incompatible.missing_keys) - allowed_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible fusion checkpoint: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )
    if getattr(model, "is_quantized_lateral_residual", False):
        configure_quantized_lateral_residual(model)
    else:
        configure_phase(model, args.phase, args.acceptance_mode)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    loss_weights = phase_loss_weights(
        args.phase,
        regret_weight=args.regret_weight,
        acceptance_mode=args.acceptance_mode,
        continuous_location_weight=args.continuous_location_weight,
        continuous_ap30_margin_weight=(
            args.continuous_ap30_margin_weight
        ),
    )
    metadata: dict[str, object] = {
        "stage": "radar_centric_fusion",
        "phase": args.phase,
        "fusion_core": args.fusion_core,
        "architecture": (
            "shared-context fixed-bin lateral residual at the expert-radar anchor; "
            "no candidate-local crops, attention, or mixture loss"
            if args.fusion_core == "quantized_lateral_residual"
            else "radar-supported dense local x lattice with causal visual bearing, "
            "immutable radar y, and exact radar-only fallback"
        ),
        "runtime_inputs": ["RA", "RA-MTI", "left_RGB_evidence", "slow_RGB_condition"],
        "bearing_source": args.bearing_source,
        "visual_anchor_contract": (
            {
                "coordinate": "bbox_top_edge_midpoint",
                "proposal_policy": (
                    "highest_score_valid_person_box_per_frame"
                ),
                "missing_policy": "causal_hold_latest_valid_anchor",
                "radar_y_usage": "immutable",
            }
            if args.bearing_source == "detector_anchor"
            else {
                "coordinate": "rv_bearing_distribution",
                "runtime_input": "causal_left_rgb_only",
                "detector_anchor_geometry_used": False,
                "radar_y_usage": "immutable",
            }
        ),
        "depth_used_at_runtime": False,
        "right_rgb_used": False,
        "vlm_condition_source_eval": eval_condition_source,
        "vlm_condition_source_train": train_condition_source,
        "vlm_cap_disabled": args.disable_vlm_cap,
        "vlm_trust_policy": args.vlm_trust_policy,
        "vlm_temporal_policy": describe_vlm_temporal_policy(
            update_hz=args.policy_update_hz,
            availability_delay_frames=args.policy_availability_delay_frames,
            hold_frames=args.policy_hold_frames,
            trust_policy=args.vlm_trust_policy,
        ),
        "correction_radius_m": args.correction_radius_m,
        "lattice_step_m": args.lattice_step_m,
        "acceptance_threshold": args.acceptance_threshold,
        "cross_modal_guard_z_threshold": (
            args.cross_modal_guard_z_threshold
        ),
        "acceptance_threshold_sweep": args.acceptance_threshold_sweep,
        "posterior_no_op_disabled": args.disable_posterior_no_op,
        "acceptance_audit_saved": args.save_acceptance_audit,
        "acceptance_mode": args.acceptance_mode,
        "correction_source": args.correction_source,
        "candidate_geometry": args.candidate_geometry,
        "candidate_prior_mode": args.candidate_prior_mode,
        "candidate_utility_source": args.candidate_utility_source,
        "candidate_prior_scale": args.candidate_prior_scale,
        "posterior_emission_mode": args.posterior_emission_mode,
        "posterior_temperature": args.posterior_temperature,
        "continuous_mode": args.continuous_mode,
        "continuous_context_mode": args.continuous_context_mode,
        "posterior_offset_scale": args.posterior_offset_scale,
        "continuous_location_weight": args.continuous_location_weight,
        "continuous_positive_weight": args.continuous_positive_weight,
        "continuous_ap30_margin_weight": (
            args.continuous_ap30_margin_weight
        ),
        "assist_policy": args.assist_policy,
        "expert_bypass_probability": (
            args.expert_bypass_probability
        ),
        "expert_bypass_routes": list(args.expert_bypass_routes),
        "candidate_positive_weight": args.candidate_positive_weight,
        "candidate_presence_weight": args.candidate_presence_weight,
        "nearest_x_background_weight": args.nearest_x_background_weight,
        "candidate_target_mode": args.candidate_target_mode,
        "candidate_target_sigma_m": args.candidate_target_sigma_m,
        "disable_visual_anchor_geometry": (
            args.disable_visual_anchor_geometry
        ),
        "gain_positive_weight": args.gain_positive_weight,
        "harm_positive_weight": args.harm_positive_weight,
        "utility_target_mode": args.utility_target_mode,
        "loss_weights": vars(loss_weights),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "route_expert_checkpoints": (
            {
                "open_indoor_los": {
                    "path": str(expert_paths.open_indoor_los),
                    "sha256": sha256(expert_paths.open_indoor_los),
                },
                "indoor_clutter": {
                    "path": str(expert_paths.indoor_clutter),
                    "sha256": sha256(expert_paths.indoor_clutter),
                },
                "open_outdoor_weak_return": {
                    "path": str(expert_paths.open_outdoor_weak_return),
                    "sha256": sha256(expert_paths.open_outdoor_weak_return),
                },
                "outdoor_dry_complex": {
                    "path": str(expert_paths.outdoor_dry_complex),
                    "sha256": sha256(expert_paths.outdoor_dry_complex),
                },
                "outdoor_wet_return": {
                    "path": str(expert_paths.outdoor_wet_return),
                    "sha256": sha256(expert_paths.outdoor_wet_return),
                },
            }
            if expert_paths is not None
            else None
        ),
        "gg_recovery_scale": args.gg_recovery_scale,
        "bearing_checkpoint_sha256": (
            sha256(args.bearing_checkpoint)
            if args.bearing_checkpoint is not None
            else None
        ),
        "rv_bearing_cache_manifest_sha256": (
            sha256(args.rv_bearing_cache_root / "cache_manifest.json")
            if args.rv_bearing_cache_root is not None
            and (
                args.rv_bearing_cache_root / "cache_manifest.json"
            ).is_file()
            else None
        ),
        "initial_checkpoint_sha256": (
            sha256(args.initial_checkpoint)
            if args.initial_checkpoint is not None
            else None
        ),
        "manifest_sha256": sha256(args.manifest),
        "condition_package_sha256": (
            sha256(args.condition_zip)
            if args.condition_zip is not None
            else None
        ),
        "train_condition_predictions_sha256": (
            sha256(args.train_condition_predictions)
            if args.train_condition_predictions is not None
            else None
        ),
        "eval_condition_predictions_sha256": (
            sha256(args.eval_condition_predictions)
            if args.eval_condition_predictions is not None
            else None
        ),
        "radar_health_manifest_sha256": sha256(
            args.radar_health_cache_root / "cache_manifest.json"
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "radar_clip_length": args.radar_clip_length,
        "visual_clip_length": args.visual_clip_length,
        "eval_excluded_sequences": list(args.eval_exclude_sequences),
        "reference_split": args.reference_split,
        "development_only": args.development_only,
        "locked_test_used": locked_test_evaluation,
        "checkpoint_guard": {
            "global_ap_0p3_drop": 0.0,
            "global_ap_0p5_drop": 0.001,
            "route_ap_0p3_drop": 0.002,
            "p90_increase_m": 0.01,
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    with torch.no_grad():
        reference_run = run_epoch(
            model=model,
            radar_adapter=radar_adapter,
            bearing_model=bearing_model,
            loader=reference_loader,
            device=device,
            optimizer=None,
            loss_weights=loss_weights,
            disable_vlm_cap=args.disable_vlm_cap,
            amp=not args.no_amp,
            max_batches=args.max_eval_batches,
            candidate_positive_weight=args.candidate_positive_weight,
            candidate_presence_weight=args.candidate_presence_weight,
            nearest_x_background_weight=args.nearest_x_background_weight,
            gain_positive_weight=args.gain_positive_weight,
            harm_positive_weight=args.harm_positive_weight,
            candidate_target_mode=args.candidate_target_mode,
            candidate_target_sigma_m=args.candidate_target_sigma_m,
            continuous_positive_weight=args.continuous_positive_weight,
            disable_visual_anchor_geometry=(
                args.disable_visual_anchor_geometry
            ),
            utility_target_mode=args.utility_target_mode,
            acceptance_thresholds=tuple(args.acceptance_threshold_sweep),
            cross_modal_guard_z_threshold=(
                args.cross_modal_guard_z_threshold
            ),
            disable_posterior_no_op=args.disable_posterior_no_op,
            collect_acceptance_audit=args.save_acceptance_audit,
            expert_bypass_probability=0.0,
            expert_bypass_routes=(),
        )
    acceptance_audit = reference_run.pop("acceptance_audit", None)
    if acceptance_audit:
        audit_path = args.output_dir / "acceptance_audit.csv"
        with audit_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(acceptance_audit[0]),
            )
            writer.writeheader()
            writer.writerows(acceptance_audit)
    radar_reference = dict(reference_run["radar"])  # type: ignore[arg-type]
    (args.output_dir / "radar_reference_metrics.json").write_text(
        json.dumps(radar_reference, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.evaluation_only:
        audit = {
            "phase": args.phase,
            "checkpoint": (
                str(args.initial_checkpoint)
                if args.initial_checkpoint is not None
                else None
            ),
            "eval": reference_run,
            "metadata": metadata,
        }
        (args.output_dir / "evaluation_only.json").write_text(
            json.dumps(audit, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(audit), flush=True)
        return

    if train_loader is None or train_sampler is None:
        raise RuntimeError("Locked Test cannot enter the training loop")

    history: list[dict[str, object]] = []
    best_unconstrained_score = -float("inf")
    best_safe_score = -float("inf")
    best_phase_score = -float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_result = run_epoch(
            model=model,
            radar_adapter=radar_adapter,
            bearing_model=bearing_model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_weights=loss_weights,
            disable_vlm_cap=args.disable_vlm_cap,
            amp=not args.no_amp,
            max_batches=args.max_train_batches,
            candidate_positive_weight=args.candidate_positive_weight,
            candidate_presence_weight=args.candidate_presence_weight,
            nearest_x_background_weight=args.nearest_x_background_weight,
            gain_positive_weight=args.gain_positive_weight,
            harm_positive_weight=args.harm_positive_weight,
            candidate_target_mode=args.candidate_target_mode,
            candidate_target_sigma_m=args.candidate_target_sigma_m,
            continuous_positive_weight=args.continuous_positive_weight,
            disable_visual_anchor_geometry=(
                args.disable_visual_anchor_geometry
            ),
            utility_target_mode=args.utility_target_mode,
            expert_bypass_probability=(
                args.expert_bypass_probability
            ),
            expert_bypass_routes=tuple(args.expert_bypass_routes),
            acceptance_thresholds=(),
            cross_modal_guard_z_threshold=(
                args.cross_modal_guard_z_threshold
            ),
        )
        with torch.no_grad():
            eval_result = run_epoch(
                model=model,
                radar_adapter=radar_adapter,
                bearing_model=bearing_model,
                loader=eval_loader,
                device=device,
                optimizer=None,
                loss_weights=loss_weights,
                disable_vlm_cap=args.disable_vlm_cap,
                amp=not args.no_amp,
                max_batches=args.max_eval_batches,
                candidate_positive_weight=args.candidate_positive_weight,
                candidate_presence_weight=args.candidate_presence_weight,
                nearest_x_background_weight=args.nearest_x_background_weight,
                gain_positive_weight=args.gain_positive_weight,
                harm_positive_weight=args.harm_positive_weight,
                candidate_target_mode=args.candidate_target_mode,
                candidate_target_sigma_m=args.candidate_target_sigma_m,
                continuous_positive_weight=args.continuous_positive_weight,
                disable_visual_anchor_geometry=(
                    args.disable_visual_anchor_geometry
                ),
                utility_target_mode=args.utility_target_mode,
                acceptance_thresholds=tuple(args.acceptance_threshold_sweep),
                cross_modal_guard_z_threshold=(
                    args.cross_modal_guard_z_threshold
                ),
                expert_bypass_probability=0.0,
                expert_bypass_routes=(),
            )
        scheduler.step()
        fused_metrics = dict(eval_result["fused"])  # type: ignore[arg-type]
        decision = safe_checkpoint_decision(radar_reference, fused_metrics)
        score = safe_checkpoint_score(fused_metrics)
        if args.phase == "candidate":
            oracle_diagnostics = eval_result["oracle_diagnostics"]  # type: ignore[index]
            location_oracle = oracle_diagnostics[
                "location_head_oracle_gate"
            ]["metrics"]
            phase_score = safe_checkpoint_score(location_oracle)
        elif args.phase == "continuous":
            oracle_diagnostics = eval_result["oracle_diagnostics"]  # type: ignore[index]
            continuous_oracle = oracle_diagnostics[
                "continuous_location_head_oracle_gate"
            ]["metrics"]
            phase_score = safe_checkpoint_score(continuous_oracle)
        elif args.phase == "gain":
            phase_score = -float(eval_result["loss"]["loss"])  # type: ignore[index]
        else:
            phase_score = score
        record = {
            "epoch": epoch,
            "train": train_result,
            "eval": eval_result,
            "checkpoint_guard": {
                "accepted": decision.accepted,
                "reason": decision.reason,
                "score": score,
                "phase_score": phase_score,
            },
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        payload = _checkpoint_payload(
            model=model,
            epoch=epoch,
            phase=args.phase,
            metrics=record,
            metadata=metadata,
        )
        if score > best_unconstrained_score:
            best_unconstrained_score = score
            torch.save(payload, args.output_dir / "best_unconstrained.pt")
        if phase_score > best_phase_score:
            best_phase_score = phase_score
            stale = 0
            torch.save(payload, args.output_dir / "best_phase.pt")
        else:
            stale += 1
        if decision.accepted and score > best_safe_score:
            best_safe_score = score
            torch.save(payload, args.output_dir / "best_safe.pt")
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        if stale >= 3:
            break
    completion = {
        "phase": args.phase,
        "epochs": len(history),
        "best_unconstrained_score": best_unconstrained_score,
        "best_phase_score": best_phase_score,
        "best_safe_score": (
            best_safe_score if best_safe_score > -float("inf") else None
        ),
        "safe_checkpoint_created": (args.output_dir / "best_safe.pt").is_file(),
    }
    (args.output_dir / "training_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()




