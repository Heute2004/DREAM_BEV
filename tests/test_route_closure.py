from __future__ import annotations

import json
from pathlib import Path

import pytest

from radar_bev.route_closure import (
    ROUTES,
    assert_development_only_paths,
    compose_selected_route_rows,
    compose_oracle_rows,
    compose_random_rows,
    compose_static_mixture_rows,
    held_random_route,
    paired_route_macro_bootstrap,
    route_macro_ap30,
    select_route_sequences,
    universal_capacity_profiles,
    validate_route_row_alignment,
)


def _row(sequence: str, frame_id: int, x_m: float) -> dict[str, object]:
    return {
        "sequence": sequence,
        "frame_id": frame_id,
        "fused_x_m": x_m,
        "fused_y_m": 2.0,
        "target_x_m": 0.0,
        "target_y_m": 2.0,
        "score": 0.8,
    }


def test_select_route_sequences_pools_only_requested_split(tmp_path: Path) -> None:
    manifest = tmp_path / "routes.jsonl"
    records = [
        {
            "sequence_name": "seq_001",
            "split": "Train",
            "adapter_route": ROUTES[0],
            "eligible_for_router_training": True,
        },
        {
            "sequence_name": "seq_002",
            "split": "Development Val",
            "adapter_route": ROUTES[1],
            "eligible_for_router_training": True,
        },
        {
            "sequence_name": "seq_003",
            "split": "Locked Test",
            "adapter_route": ROUTES[2],
            "eligible_for_router_training": True,
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    assert select_route_sequences(manifest, "Train") == {"seq_001"}
    assert select_route_sequences(manifest, "Development Val") == {"seq_002"}


def test_route_closure_refuses_locked_test_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Locked Test"):
        assert_development_only_paths(tmp_path / "Locked_Test" / "predictions.csv")


def test_universal_capacity_profiles_match_frozen_bank() -> None:
    profiles = universal_capacity_profiles()
    assert profiles["active_capacity"]["trainable_parameters"] == 72_652
    assert profiles["bank_capacity"]["trainable_parameters"] == 363_253
    assert abs(profiles["active_capacity"]["target_parameters"] - 72_650) <= 2
    assert abs(profiles["bank_capacity"]["target_parameters"] - 363_248) <= 5


def test_validate_route_rows_rejects_misaligned_frames() -> None:
    rows = {
        ROUTES[0]: [_row("seq_001", 0, 0.0)],
        ROUTES[1]: [_row("seq_001", 1, 1.0)],
    }
    with pytest.raises(ValueError, match="frame alignment"):
        validate_route_row_alignment(rows)


def test_oracle_composition_uses_frozen_sequence_ledger() -> None:
    route_rows = {
        route: [_row("seq_001", 0, float(index))]
        for index, route in enumerate(ROUTES)
    }
    composed = compose_oracle_rows(route_rows, {"seq_001": ROUTES[3]})
    assert composed[0]["fused_x_m"] == 3.0
    assert composed[0]["selected_route"] == ROUTES[3]


def test_random_route_is_identity_during_warmup_and_held_per_segment() -> None:
    assert held_random_route("seq_001", 31, seed=3407) == "identity"
    first = held_random_route("seq_001", 32, seed=3407)
    assert first in ROUTES
    assert held_random_route("seq_001", 61, seed=3407) == first
    assert held_random_route("seq_001", 62, seed=3407) in ROUTES


def test_static_mixture_averages_aligned_route_predictions() -> None:
    route_rows = {
        route: [_row("seq_001", 32, float(index))]
        for index, route in enumerate(ROUTES)
    }
    mixed = compose_static_mixture_rows(route_rows)
    assert mixed[0]["fused_x_m"] == pytest.approx(2.0)
    assert mixed[0]["selected_route"] == "static_uniform_mixture"


def test_random_composition_preserves_identity_warmup() -> None:
    route_rows = {
        route: [_row("seq_001", 0, float(index)), _row("seq_001", 32, float(index))]
        for index, route in enumerate(ROUTES)
    }
    identity_rows = [_row("seq_001", 0, -1.0), _row("seq_001", 32, -1.0)]
    random_rows = compose_random_rows(
        route_rows, identity_rows=identity_rows, seed=3407
    )
    assert random_rows[0]["selected_route"] == "identity"
    assert random_rows[0]["fused_x_m"] == -1.0
    assert random_rows[1]["selected_route"] in ROUTES


def test_selected_route_composition_uses_identity_for_unknown_policy() -> None:
    route_rows = {
        route: [_row("seq_001", 0, float(index)), _row("seq_001", 32, float(index))]
        for index, route in enumerate(ROUTES)
    }
    identity_rows = [_row("seq_001", 0, -1.0), _row("seq_001", 32, -1.0)]
    selected = {
        ("seq_001", 0): "identity",
        ("seq_001", 32): ROUTES[2],
    }

    rows = compose_selected_route_rows(
        route_rows, selected_routes=selected, identity_rows=identity_rows
    )

    assert rows[0]["selected_route"] == "identity"
    assert rows[0]["fused_x_m"] == -1.0
    assert rows[1]["selected_route"] == ROUTES[2]
    assert rows[1]["fused_x_m"] == 2.0


def test_route_macro_ap30_weights_routes_equally() -> None:
    rows = []
    ledger = {}
    for index, route in enumerate(ROUTES, start=1):
        sequence = f"seq_{index:03d}"
        rows.append(_row(sequence, 0, 0.0 if index == 1 else 2.0))
        ledger[sequence] = route
    value = route_macro_ap30(rows, ledger)
    assert value == pytest.approx(0.2)


def test_paired_route_macro_bootstrap_preserves_route_strata() -> None:
    baseline = []
    improved = []
    ledger = {}
    for index, route in enumerate(ROUTES, start=1):
        sequence = f"seq_{index:03d}"
        ledger[sequence] = route
        baseline.append(_row(sequence, 0, 1.0))
        improved.append(_row(sequence, 0, 0.0))
    result = paired_route_macro_bootstrap(
        improved, baseline, ledger, repetitions=20, seed=3407
    )
    assert result["point_delta_pp"] == pytest.approx(100.0)
    assert result["confidence_interval_95_pp"][0] == pytest.approx(100.0)
