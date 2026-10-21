from __future__ import annotations

import torch
from torch import Tensor, nn

from .contracts import ConditionPrior


ROUTES = (
    "open_indoor_los",
    "indoor_clutter",
    "open_outdoor_weak_return",
    "outdoor_dry_complex",
    "outdoor_wet_return",
    "corner_nlos",
    "outdoor_adverse",
    "unknown",
)
VISUAL_CONDITIONS = (
    "normal",
    "normal_visibility",
    "low_light",
    "low_light_extreme",
    "exposure",
    "fog",
    "dense_fog",
    "lens_droplets",
    "severe_blur",
    "rain_wet",
    "mixed_degradation",
    "unknown",
)
RELIABILITIES = ("usable", "cautious", "unreliable", "unknown")
TRUST_POLICIES = (
    "soft",
    "strict_usable",
    "normal_nonunreliable",
    "normal_or_rain_nonunreliable",
)
WEATHER_SURFACES = (
    "indoor",
    "dry",
    "wet",
    "wet_return",
    "rain",
    "rain_wet",
    "fog",
    "night",
    "unknown",
)


def _indices(values: tuple[str, ...], vocabulary: tuple[str, ...], device: torch.device) -> Tensor:
    lookup = {value: index for index, value in enumerate(vocabulary)}
    unknown = lookup["unknown"]
    return torch.tensor([lookup.get(str(value).strip().lower(), unknown) for value in values], device=device)


def visual_trust_cap(
    prior: ConditionPrior,
    device: torch.device,
    *,
    policy: str = "soft",
) -> Tensor:
    """Convert an optional VLM policy output into a deterministic trust ceiling."""

    if policy not in TRUST_POLICIES:
        raise ValueError(f"unknown visual trust policy: {policy}")
    if policy in {
        "strict_usable",
        "normal_nonunreliable",
        "normal_or_rain_nonunreliable",
    }:
        allowed_conditions = {"normal", "normal_visibility"}
        if policy == "normal_or_rain_nonunreliable":
            allowed_conditions.add("rain_wet")
        values = [
            float(
                (
                    reliability.strip().lower() == "usable"
                    if policy == "strict_usable"
                    else reliability.strip().lower() != "unreliable"
                )
                and condition.strip().lower()
                in allowed_conditions
            )
            for reliability, condition in zip(
                prior.visual_reliability,
                prior.visual_condition,
                strict=True,
            )
        ]
        return torch.tensor(
            values,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)

    reliability_cap = {
        "usable": 1.0,
        "cautious": 0.45,
        "unreliable": 0.0,
        "unknown": 0.25,
    }
    condition_cap = {
        "normal": 1.0,
        "normal_visibility": 1.0,
        "low_light": 0.25,
        "low_light_extreme": 0.10,
        "exposure": 0.15,
        "fog": 0.35,
        "dense_fog": 0.20,
        "lens_droplets": 0.30,
        "severe_blur": 0.20,
        "rain_wet": 0.40,
        "mixed_degradation": 0.15,
        "unknown": 0.25,
    }
    values = [
        min(
            reliability_cap.get(reliability.strip().lower(), reliability_cap["unknown"]),
            condition_cap.get(condition.strip().lower(), condition_cap["unknown"]),
        )
        for reliability, condition in zip(
            prior.visual_reliability, prior.visual_condition, strict=True
        )
    ]
    return torch.tensor(values, device=device, dtype=torch.float32).unsqueeze(1)


class ConditionEncoder(nn.Module):
    """Encode slow VLM context and enforce a deterministic visual trust ceiling."""

    def __init__(
        self,
        embedding_dim: int = 32,
        *,
        trust_policy: str = "soft",
    ) -> None:
        super().__init__()
        if trust_policy not in TRUST_POLICIES:
            raise ValueError(f"unknown visual trust policy: {trust_policy}")
        self.trust_policy = trust_policy
        self.route = nn.Embedding(len(ROUTES), 16)
        self.visual = nn.Embedding(len(VISUAL_CONDITIONS), 8)
        self.reliability = nn.Embedding(len(RELIABILITIES), 4)
        self.weather = nn.Embedding(len(WEATHER_SURFACES), 4)
        self.project = nn.Sequential(nn.Linear(32, embedding_dim), nn.SiLU(), nn.LayerNorm(embedding_dim))

    def _trust_cap(
        self,
        prior: ConditionPrior,
        device: torch.device,
    ) -> Tensor:
        return visual_trust_cap(
            prior,
            device,
            policy=self.trust_policy,
        )

    def forward(self, prior: ConditionPrior) -> tuple[Tensor, Tensor]:
        prior.validate()
        device = self.route.weight.device
        encoded = torch.cat(
            (
                self.route(_indices(prior.global_route, ROUTES, device)),
                self.visual(_indices(prior.visual_condition, VISUAL_CONDITIONS, device)),
                self.reliability(_indices(prior.visual_reliability, RELIABILITIES, device)),
                self.weather(_indices(prior.weather_surface, WEATHER_SURFACES, device)),
            ),
            dim=1,
        )
        return self.project(encoded), self._trust_cap(prior, device)



