"""Materialize route-only counterfactual policy traces for Development audits."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


SUPPORTED_EXPERT_ROUTES = (
    "open_indoor_los",
    "indoor_clutter",
    "open_outdoor_weak_return",
    "outdoor_dry_complex",
    "outdoor_wet_return",
)


class RouteCounterfactualError(ValueError):
    """Raised when a policy trace cannot support a route-only counterfactual."""


def _validated_copy(record: Mapping[str, object]) -> dict[str, object]:
    copied = copy.deepcopy(dict(record))
    if not isinstance(copied.get("segment_rvfusion_admission"), bool):
        raise RouteCounterfactualError(
            "route counterfactuals require a materialized segment policy trace"
        )
    prediction = copied.get("prediction")
    if not isinstance(prediction, dict):
        raise RouteCounterfactualError("policy record is missing its prediction object")
    if not isinstance(prediction.get("expert_route"), str):
        raise RouteCounterfactualError("policy prediction is missing expert_route")
    return copied


def force_expert_route(
    records: Iterable[Mapping[str, object]], target_route: str
) -> list[dict[str, object]]:
    """Clone a materialized trace, replacing only active expert-route values.

    The causal warm-up remains exact radar identity. Segment timing, visual
    admission, conditions, rationale, and every other record field are copied
    verbatim, so the resulting trace is suitable for a route-only audit.
    """

    if target_route not in SUPPORTED_EXPERT_ROUTES:
        raise RouteCounterfactualError(f"unsupported target route: {target_route}")

    forced: list[dict[str, object]] = []
    for record in records:
        copied = _validated_copy(record)
        if copied.get("policy_source") != "causal_warmup_identity":
            prediction = copied["prediction"]
            assert isinstance(prediction, dict)
            prediction["expert_route"] = target_route
        forced.append(copied)
    return forced


def read_policy_trace(path: str | Path) -> list[dict[str, object]]:
    """Load a JSONL materialized policy trace without accepting blank records."""

    source = Path(path)
    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise RouteCounterfactualError(
                    f"policy trace line {line_number} is not a JSON object"
                )
            rows.append(record)
    if not rows:
        raise RouteCounterfactualError("policy trace contains no records")
    return rows


def write_policy_trace(path: str | Path, records: Iterable[Mapping[str, object]]) -> None:
    """Write a route-only counterfactual trace as stable JSONL."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def force_policy_trace(
    source: str | Path, destination: str | Path, target_route: str
) -> None:
    """Read, force, and write one auditable route-only counterfactual trace."""

    write_policy_trace(
        destination,
        force_expert_route(read_policy_trace(source), target_route),
    )
