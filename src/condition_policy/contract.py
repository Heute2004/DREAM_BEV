"""Shared output contract for RGB condition policy."""

from __future__ import annotations

from typing import Any


EXPERT_ROUTES = {
    "corner_nlos",
    "indoor_clutter",
    "open_indoor_los",
    "open_outdoor_weak_return",
    "outdoor_dry_complex",
    "outdoor_wet_return",
    "unknown",
}
VISUAL_CONDITIONS = {"normal_visibility", "low_light", "rain_or_wet", "unknown"}
VISUAL_RELIABILITY = {"usable", "cautious", "unreliable", "unknown"}
WEATHER_SURFACE = {"indoor", "dry", "wet_return", "unknown"}
REQUIRED_FIELDS = {
    "expert_route",
    "expert_route_basis",
    "visual_condition",
    "visual_reliability_prior",
    "weather_surface_prior",
    "rationale_zh",
}
TASK_FIELDS = (
    "expert_route",
    "visual_condition",
    "visual_reliability_prior",
    "weather_surface_prior",
)


def validate_prediction(value: dict[str, Any]) -> bool:
    """Return whether a parsed model response exactly satisfies condition-policy schema."""

    return (
        set(value) == REQUIRED_FIELDS
        and value.get("expert_route") in EXPERT_ROUTES
        and value.get("expert_route_basis") in {"current_left_rgb_estimate", "unresolved"}
        and value.get("visual_condition") in VISUAL_CONDITIONS
        and value.get("visual_reliability_prior") in VISUAL_RELIABILITY
        and value.get("weather_surface_prior") in WEATHER_SURFACE
        and isinstance(value.get("rationale_zh"), str)
    )


def safe_dispatch_prediction(value: dict[str, Any] | None) -> dict[str, str]:
    """Return a deployable policy or fail closed to the radar-only fallback."""

    if isinstance(value, dict) and validate_prediction(value):
        return {field: str(value[field]) for field in REQUIRED_FIELDS}
    return {
        "expert_route": "unknown",
        "expert_route_basis": "unresolved",
        "visual_condition": "unknown",
        "visual_reliability_prior": "unreliable",
        "weather_surface_prior": "unknown",
        "rationale_zh": "Invalid visual-policy response; use radar-only identity fallback.",
    }



