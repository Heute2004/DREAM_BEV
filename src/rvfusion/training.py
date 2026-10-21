from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .contracts import ConditionPrior


@dataclass(frozen=True)
class CheckpointDecision:
    accepted: bool
    reason: str


def safe_checkpoint_decision(
    radar: Mapping[str, object],
    fused: Mapping[str, object],
) -> CheckpointDecision:
    """Reject a fusion checkpoint that regresses the frozen radar reference."""
    if float(fused["ap_0p3m"]) < float(radar["ap_0p3m"]):
        return CheckpointDecision(False, "global_ap_0p3_drop")
    if (
        "ap_0p5m" in radar
        and "ap_0p5m" in fused
        and float(fused["ap_0p5m"]) < float(radar["ap_0p5m"]) - 0.001
    ):
        return CheckpointDecision(False, "global_ap_0p5_drop")
    if (
        "p90_location_error_m" in radar
        and "p90_location_error_m" in fused
        and float(fused["p90_location_error_m"])
        > float(radar["p90_location_error_m"]) + 0.01
    ):
        return CheckpointDecision(False, "p90_worse")
    radar_routes = dict(radar.get("per_route", {}))  # type: ignore[arg-type]
    fused_routes = dict(fused.get("per_route", {}))  # type: ignore[arg-type]
    for route, radar_metric in radar_routes.items():
        if route not in fused_routes:
            return CheckpointDecision(False, "missing_route_metric")
        radar_route = dict(radar_metric)  # type: ignore[arg-type]
        fused_route = dict(fused_routes[route])  # type: ignore[arg-type]
        if float(fused_route["ap_0p3m"]) < float(radar_route["ap_0p3m"]) - 0.002:
            return CheckpointDecision(False, "route_ap_0p3_drop")
    return CheckpointDecision(True, "accepted")


def safe_checkpoint_score(metrics: Mapping[str, object]) -> float:
    return (
        float(metrics["ap_0p3m"])
        - 0.1 * float(metrics["mean_location_error_m"])
        - 0.05 * float(metrics["p90_location_error_m"])
    )


def condition_from_collated(value: Mapping[str, object]) -> ConditionPrior:
    def strings(name: str) -> tuple[str, ...]:
        raw = value[name]
        if isinstance(raw, str):
            return (raw,)
        return tuple(str(item) for item in raw)  # type: ignore[arg-type]

    return ConditionPrior(
        global_route=strings("global_route"),
        visual_condition=strings("visual_condition"),
        visual_reliability=strings("visual_reliability"),
        weather_surface=strings("weather_surface"),
    )
