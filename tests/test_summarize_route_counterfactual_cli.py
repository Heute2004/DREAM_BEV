from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_writes_reference_route_grouped_metrics(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    audit = tmp_path / "audit.json"
    output = tmp_path / "metrics.json"
    trace.write_text(
        json.dumps(
            {
                "image": "Data_train/seq_082/RGB/frame_0032.jpg",
                "prediction": {"expert_route": "open_indoor_los"},
                "policy_segment_start_frame": 32,
                "segment_rvfusion_admission": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            [
                {
                    "sequence": "seq_082",
                    "frame_id": 32,
                    "target_x_m": 0.0,
                    "target_y_m": 2.0,
                    "radar_x_m": 0.5,
                    "radar_y_m": 2.0,
                    "candidate_x_m": 0.0,
                    "candidate_y_m": 2.0,
                    "radar_score": 0.9,
                    "accepted": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "summarize_route_counterfactual.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--audit-json",
            str(audit),
            "--reference-trace",
            str(trace),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["open_indoor_los"]["frames"] == 1.0
    assert metrics["open_indoor_los"]["ap_0p3m"] == 1.0


def test_cli_can_group_active_frames_by_a_frozen_sequence_route_ledger(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    audit = tmp_path / "audit.json"
    ledger = tmp_path / "ledger.json"
    output = tmp_path / "metrics.json"
    trace.write_text(
        json.dumps(
            {
                "image": "Data_train/seq_082/RGB/frame_0032.jpg",
                "prediction": {"expert_route": "open_indoor_los"},
                "policy_segment_start_frame": 32,
                "segment_rvfusion_admission": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            [
                {
                    "sequence": "seq_082",
                    "frame_id": 32,
                    "target_x_m": 0.0,
                    "target_y_m": 2.0,
                    "radar_x_m": 0.5,
                    "radar_y_m": 2.0,
                    "candidate_x_m": 0.0,
                    "candidate_y_m": 2.0,
                    "radar_score": 0.9,
                    "accepted": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    ledger.write_text(
        json.dumps({"seq_082": "outdoor_wet_return"}), encoding="utf-8"
    )
    script = Path(__file__).parents[1] / "scripts" / "summarize_route_counterfactual.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--audit-json",
            str(audit),
            "--reference-trace",
            str(trace),
            "--sequence-route-map",
            str(ledger),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert set(metrics) == {"outdoor_wet_return"}


def test_cli_accepts_guarded_replay_rows(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    audit = tmp_path / "guarded.csv"
    output = tmp_path / "metrics.json"
    trace.write_text(
        json.dumps(
            {
                "image": "Data_train/seq_082/RGB/frame_0032.jpg",
                "prediction": {"expert_route": "open_indoor_los"},
                "policy_segment_start_frame": 32,
                "segment_rvfusion_admission": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit.write_text(
        "sequence,frame_id,radar_x_m,radar_y_m,candidate_x_m,target_x_m,"
        "target_y_m,score,visual_accepted,fused_x_m,fused_y_m\n"
        "seq_082,32,0.5,2.0,0.0,0.0,2.0,0.9,True,0.0,2.0\n",
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "summarize_route_counterfactual.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--audit-csv",
            str(audit),
            "--reference-trace",
            str(trace),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads(output.read_text(encoding="utf-8"))
    assert metrics["open_indoor_los"]["ap_0p3m"] == 1.0
