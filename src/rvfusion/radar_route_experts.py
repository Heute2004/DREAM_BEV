from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RadarRouteExpertPaths:
    open_indoor_los: Path
    indoor_clutter: Path
    open_outdoor_weak_return: Path
    outdoor_dry_complex: Path
    outdoor_wet_return: Path


class FrozenRadarRouteExpert(nn.Module):
    """Expose one route expert through the common radar-evidence contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        if not hasattr(model, "backbone"):
            raise TypeError("route expert model must expose its frozen backbone")
        self.model = model
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenRadarRouteExpert":
        super().train(False)
        self.model.eval()
        return self

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(radar)
            if not isinstance(output, dict):
                raise TypeError("route expert model must return a dictionary")
            polar_feature = self.model.backbone.forward_polar_feature(radar)
        return {
            "polar_feature": polar_feature,
            "shared_feature": output["shared_feature"],
            "logits": output["logits"],
        }


class FrozenPBMHBRRouteExpert(nn.Module):
    """Expose the causal dry-complex coordinate without altering base confidence."""

    requires_causal_metadata = True

    def __init__(self, model: nn.Module, parameters: object | None = None) -> None:
        super().__init__()
        if not hasattr(model, "backbone"):
            raise TypeError("dry-complex model must expose its frozen backbone")
        from radar_bev.dry_complex_training import PBMHBRParameters, PBMHBRStreamingRefiner

        if parameters is None:
            parameters = PBMHBRParameters(polar_interpolation=0.225)
        if not isinstance(parameters, PBMHBRParameters):
            raise TypeError("parameters must be PBMHBRParameters")
        self.model = model
        self.refiner = PBMHBRStreamingRefiner(parameters)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenPBMHBRRouteExpert":
        super().train(False)
        self.model.eval()
        return self

    def reset_causal_state(self) -> None:
        self.refiner.reset()

    def forward(
        self,
        radar: Tensor,
        *,
        sequences: Sequence[str],
        frame_ids: Sequence[int],
    ) -> dict[str, Tensor]:
        if len(sequences) != radar.shape[0] or len(frame_ids) != radar.shape[0]:
            raise ValueError("dry-complex metadata must match radar batch")
        self.model.eval()
        with torch.no_grad():
            output = self.model(radar)
            if not isinstance(output, dict):
                raise TypeError("dry-complex model must return a dictionary")
            candidates = output["candidate_xy_m"][:, : self.refiner.parameters.candidate_limit]
            refined = torch.stack(
                tuple(
                    self.refiner.step(sequence, int(frame_id), candidate).refined_xy_m
                    for sequence, frame_id, candidate in zip(
                        sequences,
                        frame_ids,
                        candidates,
                    )
                )
            )
            polar_feature = self.model.backbone.forward_polar_feature(radar)
        return {
            "polar_feature": polar_feature,
            "shared_feature": output["shared_feature"],
            "logits": output["base_logits"],
            "position_override_xy_m": refined,
            "position_override_mask": torch.ones(
                radar.shape[0],
                device=radar.device,
                dtype=torch.bool,
            ),
        }


class FrozenGGSRRERecoveryRouteExpert(nn.Module):
    """Apply the frozen global GG-SRRE recovery scale selected on Development Val."""

    def __init__(self, model: nn.Module, recovery_scale: float = 1.15) -> None:
        super().__init__()
        if not hasattr(model, "backbone"):
            raise TypeError("GG-SRRE model must expose its frozen backbone")
        if recovery_scale <= 0.0:
            raise ValueError("recovery_scale must be positive")
        self.model = model
        self.recovery_scale = float(recovery_scale)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenGGSRRERecoveryRouteExpert":
        super().train(False)
        self.model.eval()
        return self

    def forward(self, radar: Tensor) -> dict[str, Tensor]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(radar)
            if not isinstance(output, dict):
                raise TypeError("GG-SRRE model must return a dictionary")
            logits = (
                output["base_logits"]
                + self.recovery_scale * output["delta_bev_logit"]
            )
            polar_feature = self.model.backbone.forward_polar_feature(radar)
        return {
            "polar_feature": polar_feature,
            "shared_feature": output["shared_feature"],
            "logits": logits,
        }


def _load_expert_state(model: nn.Module, checkpoint: Path) -> None:
    if hasattr(model, "load_expert_checkpoint"):
        model.load_expert_checkpoint(checkpoint)
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("expert_state", payload)
    if not isinstance(state, dict) or "expert" not in state:
        raise ValueError(f"invalid route expert checkpoint: {checkpoint}")
    model.expert.load_state_dict(state["expert"], strict=True)


def _load_pbmhbr_parameters(path: Path) -> object:
    from radar_bev.dry_complex_training import PBMHBRParameters

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("runtime_uses_ground_truth") is not False:
        raise ValueError("dry-complex runtime contract must not use ground truth")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("dry-complex parameter file is missing parameters")
    return PBMHBRParameters(
        candidate_limit=int(parameters["candidate_limit"]),
        max_step_m=float(parameters["max_step_m"]),
        base_bonus_m=float(parameters["base_bonus_m"]),
        polar_threshold_intercept_m=float(
            parameters["polar_threshold_intercept_m"]
        ),
        polar_threshold_range_slope_m=float(
            parameters["polar_threshold_range_slope_m"]
        ),
        polar_interpolation=float(parameters["polar_interpolation"]),
    )


def build_frozen_route_experts(
    *,
    base_checkpoint: Path,
    paths: RadarRouteExpertPaths,
    device: torch.device,
    gg_recovery_scale: float = 1.15,
) -> dict[str, nn.Module]:
    from radar_bev.models import build_model

    specifications = {
        "open_indoor_los": (
            "dream_heterogeneous_expert",
            {"expert_kind": "clor"},
            paths.open_indoor_los,
        ),
        "indoor_clutter": (
            "dream_clutter_expert",
            {
                "expert_kind": "tecs",
                "candidate_count": 10,
                "logit_cap": 3.5,
                "acceptance_temperature": 1.5,
                "acceptance_bias": 0.5,
                "residual_scale": 1.0,
            },
            paths.indoor_clutter,
        ),
        "open_outdoor_weak_return": (
            "dream_weak_return_expert",
            {"expert_kind": "ltge_cen", "candidate_count": 8},
            paths.open_outdoor_weak_return,
        ),
    }
    result: dict[str, nn.Module] = {}
    for route, (model_name, kwargs, checkpoint) in specifications.items():
        model = build_model(model_name, **kwargs)
        model.load_base_checkpoint(base_checkpoint)
        _load_expert_state(model, checkpoint)
        result[route] = FrozenRadarRouteExpert(model.to(device))

    pb_model = build_model(
        "dream_dry_complex_expert",
        logit_cap=3.0,
        residual_scale=1.0,
    )
    pb_model.load_base_checkpoint(base_checkpoint)
    result["outdoor_dry_complex"] = FrozenPBMHBRRouteExpert(
        pb_model.to(device),
        _load_pbmhbr_parameters(paths.outdoor_dry_complex),
    )

    gg_model = build_model("dream_recovery_expert", expert_kind="gg_srre")
    gg_model.load_base_checkpoint(base_checkpoint)
    _load_expert_state(gg_model, paths.outdoor_wet_return)
    gg_model.set_training_phase("recovery")
    result["outdoor_wet_return"] = FrozenGGSRRERecoveryRouteExpert(
        gg_model.to(device),
        recovery_scale=gg_recovery_scale,
    )
    return result



