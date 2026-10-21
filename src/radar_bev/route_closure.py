from __future__ import annotations

import hashlib
import json
from statistics import fmean
from pathlib import Path
from typing import Mapping, Sequence

import torch
import numpy as np

from .evaluation import evaluate_single_person_predictions
from .models.route_experts import MultiScaleDetailAdapter


ROUTES = (
    "open_indoor_los",
    "indoor_clutter",
    "open_outdoor_weak_return",
    "outdoor_dry_complex",
    "outdoor_wet_return",
)

SPECIALIST_PARAMETERS = {
    "open_indoor_los": 75_462,
    "indoor_clutter": 78_549,
    "open_outdoor_weak_return": 28_017,
    "outdoor_dry_complex": 106_209,
    "outdoor_wet_return": 75_011,
}


def assert_development_only_paths(*paths: str | Path) -> None:
    for path in paths:
        normalized = str(path).replace("-", "_").replace(" ", "_").lower()
        if "locked_test" in normalized or "lockedtest" in normalized:
            raise ValueError(f"Locked Test path is forbidden in route-closure work: {path}")


def select_route_sequences(routing_manifest: Path, split: str) -> set[str]:
    """Pool all five named routes while retaining complete-sequence splits."""

    selected: set[str] = set()
    for raw_line in Path(routing_manifest).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if str(record.get("split", "")).strip().lower() != split.strip().lower():
            continue
        if record.get("adapter_route") not in ROUTES:
            continue
        if not bool(record.get("eligible_for_router_training", False)):
            continue
        selected.add(str(record["sequence_name"]))
    if not selected:
        raise ValueError(f"no eligible named-route sequences for split={split!r}")
    return selected


def _adapter_parameters(input_channels: int, detail_channels: int) -> int:
    return sum(
        parameter.numel()
        for parameter in MultiScaleDetailAdapter(
            input_channels, detail_channels
        ).parameters()
    )


def universal_capacity_profiles() -> dict[str, dict[str, int]]:
    bank_parameters = sum(SPECIALIST_PARAMETERS.values())
    active_target = round(bank_parameters / len(SPECIALIST_PARAMETERS))
    profiles = {
        "active_capacity": {
            "polar_detail_channels": 96,
            "bev_detail_channels": 80,
            "target_parameters": active_target,
        },
        "bank_capacity": {
            "polar_detail_channels": 195,
            "bev_detail_channels": 263,
            "target_parameters": bank_parameters,
        },
    }
    for profile in profiles.values():
        profile["trainable_parameters"] = _adapter_parameters(
            48, profile["polar_detail_channels"]
        ) + _adapter_parameters(128, profile["bev_detail_channels"])
    return profiles


def _frame_keys(rows: Sequence[Mapping[str, object]]) -> tuple[tuple[str, int], ...]:
    return tuple((str(row["sequence"]), int(row["frame_id"])) for row in rows)


def validate_route_row_alignment(
    route_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[tuple[str, int], ...]:
    if len(route_rows) < 2:
        raise ValueError("at least two route arms are required")
    reference: tuple[tuple[str, int], ...] | None = None
    for route, rows in route_rows.items():
        if route not in ROUTES:
            raise ValueError(f"unsupported route arm: {route}")
        keys = _frame_keys(rows)
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate frame key in route={route}")
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError("route-arm frame alignment mismatch")
    assert reference is not None
    return reference


def compose_oracle_rows(
    route_rows: Mapping[str, Sequence[Mapping[str, object]]],
    sequence_ledger: Mapping[str, str],
) -> list[dict[str, object]]:
    keys = validate_route_row_alignment(route_rows)
    route_maps = {
        route: {key: row for key, row in zip(keys, rows)}
        for route, rows in route_rows.items()
    }
    output: list[dict[str, object]] = []
    for key in keys:
        route = sequence_ledger.get(key[0])
        if route not in route_maps:
            raise ValueError(f"sequence ledger route unavailable for {key[0]}: {route}")
        output.append({**route_maps[route][key], "selected_route": route})
    return output


def compose_selected_route_rows(
    route_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    selected_routes: Mapping[tuple[str, int], str],
    identity_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compose a policy arm from the same forced-route matrix as its controls."""

    keys = validate_route_row_alignment(route_rows)
    if _frame_keys(identity_rows) != keys:
        raise ValueError("identity and route-arm frame alignment mismatch")
    route_maps = {
        route: {key: row for key, row in zip(keys, rows)}
        for route, rows in route_rows.items()
    }
    identity_map = {key: row for key, row in zip(keys, identity_rows)}
    output: list[dict[str, object]] = []
    for key in keys:
        selected = selected_routes.get(key, "identity")
        if selected in route_maps:
            source = route_maps[selected][key]
        elif selected in {"identity", "unknown", ""}:
            selected = "identity"
            source = identity_map[key]
        else:
            raise ValueError(f"selected route is unavailable for {key}: {selected}")
        output.append({**source, "selected_route": selected})
    return output


def compose_static_mixture_rows(
    route_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    keys = validate_route_row_alignment(route_rows)
    ordered_routes = tuple(route for route in ROUTES if route in route_rows)
    output: list[dict[str, object]] = []
    for index, _ in enumerate(keys):
        members = [route_rows[route][index] for route in ordered_routes]
        row = dict(members[0])
        for field in ("fused_x_m", "fused_y_m", "score"):
            row[field] = fmean(float(member[field]) for member in members)
        row["selected_route"] = "static_uniform_mixture"
        output.append(row)
    return output


def compose_random_rows(
    route_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    identity_rows: Sequence[Mapping[str, object]],
    seed: int,
) -> list[dict[str, object]]:
    keys = validate_route_row_alignment(route_rows)
    selected_routes = {
        key: held_random_route(key[0], key[1], seed=seed) for key in keys
    }
    return compose_selected_route_rows(
        route_rows,
        selected_routes=selected_routes,
        identity_rows=identity_rows,
    )


def held_random_route(
    sequence: str,
    frame_id: int,
    *,
    seed: int,
    warmup_frames: int = 32,
    hold_frames: int = 30,
) -> str:
    if frame_id < warmup_frames:
        return "identity"
    segment = (frame_id - warmup_frames) // hold_frames
    payload = f"{seed}:{sequence}:{segment}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(ROUTES)
    return ROUTES[index]


def evaluate_composed_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot evaluate empty composed rows")
    scores = torch.tensor([float(row["score"]) for row in rows], dtype=torch.float64)
    prediction = torch.tensor(
        [[float(row["fused_x_m"]), float(row["fused_y_m"])] for row in rows],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [[float(row["target_x_m"]), float(row["target_y_m"])] for row in rows],
        dtype=torch.float64,
    )
    return evaluate_single_person_predictions(
        scores,
        prediction,
        target,
        distance_thresholds=(0.25, 0.30, 0.50),
    )


def route_macro_ap30(
    rows: Sequence[Mapping[str, object]], sequence_ledger: Mapping[str, str]
) -> float:
    grouped: dict[str, list[Mapping[str, object]]] = {route: [] for route in ROUTES}
    for row in rows:
        sequence = str(row["sequence"])
        route = sequence_ledger.get(sequence)
        if route in grouped:
            grouped[route].append(row)
    missing = [route for route, members in grouped.items() if not members]
    if missing:
        raise ValueError(f"route macro requires every named route; missing={missing}")
    return fmean(evaluate_composed_rows(grouped[route])["ap_0p3m"] for route in ROUTES)


def paired_route_macro_bootstrap(
    method_rows: Sequence[Mapping[str, object]],
    reference_rows: Sequence[Mapping[str, object]],
    sequence_ledger: Mapping[str, str],
    *,
    repetitions: int = 2_000,
    seed: int = 3407,
) -> dict[str, object]:
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if _frame_keys(method_rows) != _frame_keys(reference_rows):
        raise ValueError("paired bootstrap requires aligned frame rows")
    method_by_sequence: dict[str, list[Mapping[str, object]]] = {}
    reference_by_sequence: dict[str, list[Mapping[str, object]]] = {}
    for method, reference in zip(method_rows, reference_rows):
        sequence = str(method["sequence"])
        method_by_sequence.setdefault(sequence, []).append(method)
        reference_by_sequence.setdefault(sequence, []).append(reference)
    sequences_by_route = {
        route: sorted(
            sequence
            for sequence in method_by_sequence
            if sequence_ledger.get(sequence) == route
        )
        for route in ROUTES
    }
    missing = [route for route, sequences in sequences_by_route.items() if not sequences]
    if missing:
        raise ValueError(f"bootstrap route strata are empty: {missing}")
    point = 100.0 * (
        route_macro_ap30(method_rows, sequence_ledger)
        - route_macro_ap30(reference_rows, sequence_ledger)
    )
    def ap30_arrays(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        errors = np.asarray(
            [
                np.hypot(
                    float(row["fused_x_m"]) - float(row["target_x_m"]),
                    float(row["fused_y_m"]) - float(row["target_y_m"]),
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        return scores, errors <= 0.30

    def average_precision(scores: np.ndarray, matched: np.ndarray) -> float:
        order = np.argsort(-scores, kind="stable")
        ordered = matched[order].astype(np.float64, copy=False)
        true_positive = np.cumsum(ordered)
        false_positive = np.cumsum(1.0 - ordered)
        precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
        recall = true_positive / ordered.size
        precision = np.concatenate(([0.0], precision, [0.0]))
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        changed = recall[1:] != recall[:-1]
        return float(
            np.sum(
                (recall[1:][changed] - recall[:-1][changed])
                * precision[1:][changed]
            )
        )

    method_arrays = {
        sequence: ap30_arrays(rows) for sequence, rows in method_by_sequence.items()
    }
    reference_arrays = {
        sequence: ap30_arrays(rows) for sequence, rows in reference_by_sequence.items()
    }
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(repetitions):
        method_route_ap: list[float] = []
        reference_route_ap: list[float] = []
        for route in ROUTES:
            sequences = sequences_by_route[route]
            sampled = rng.choice(sequences, size=len(sequences), replace=True)
            method_scores = np.concatenate(
                [method_arrays[str(sequence)][0] for sequence in sampled]
            )
            method_matched = np.concatenate(
                [method_arrays[str(sequence)][1] for sequence in sampled]
            )
            reference_scores = np.concatenate(
                [reference_arrays[str(sequence)][0] for sequence in sampled]
            )
            reference_matched = np.concatenate(
                [reference_arrays[str(sequence)][1] for sequence in sampled]
            )
            method_route_ap.append(average_precision(method_scores, method_matched))
            reference_route_ap.append(
                average_precision(reference_scores, reference_matched)
            )
        draws.append(100.0 * (fmean(method_route_ap) - fmean(reference_route_ap)))
    return {
        "bootstrap_unit": "complete_sequence_stratified_by_route",
        "repetitions": repetitions,
        "seed": seed,
        "point_delta_pp": point,
        "confidence_interval_95_pp": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }
