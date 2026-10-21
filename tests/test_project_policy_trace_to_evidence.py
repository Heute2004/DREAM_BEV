from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_projected_trace_preserves_segment_admission_and_uses_forced_anchor(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).parents[1]
    policy = tmp_path / "policy.csv"
    evidence = tmp_path / "forced_route.csv"
    output = tmp_path / "projected.csv"
    _write_csv(
        policy,
        [
            "sequence",
            "frame_id",
            "policy_admitted",
            "radar_x_m",
            "radar_y_m",
            "candidate_x_m",
            "target_x_m",
            "target_y_m",
            "score",
            "signed_innovation",
        ],
        [
            {
                "sequence": "seq_022",
                "frame_id": 0,
                "policy_admitted": "False",
                "radar_x_m": 0.0,
                "radar_y_m": 2.0,
                "candidate_x_m": 0.1,
                "target_x_m": 0.2,
                "target_y_m": 1.9,
                "score": 0.1,
                "signed_innovation": 0.0,
            },
            {
                "sequence": "seq_022",
                "frame_id": 1,
                "policy_admitted": "True",
                "radar_x_m": 0.1,
                "radar_y_m": 2.0,
                "candidate_x_m": 0.2,
                "target_x_m": 0.3,
                "target_y_m": 1.9,
                "score": 0.2,
                "signed_innovation": 0.0,
            },
        ],
    )
    _write_csv(
        evidence,
        [
            "sequence",
            "frame_id",
            "target_x_m",
            "target_y_m",
            "radar_x_m",
            "radar_y_m",
            "candidate_x_m",
            "candidate_score",
            "calibrated_visual_x_m",
            "cross_modal_innovation",
        ],
        [
            {
                "sequence": "seq_022",
                "frame_id": 0,
                "target_x_m": 0.2,
                "target_y_m": 1.9,
                "radar_x_m": -0.3,
                "radar_y_m": 2.1,
                "candidate_x_m": -0.2,
                "candidate_score": 0.8,
                "calibrated_visual_x_m": -0.5,
                "cross_modal_innovation": 0.4,
            },
            {
                "sequence": "seq_022",
                "frame_id": 1,
                "target_x_m": 0.3,
                "target_y_m": 1.9,
                "radar_x_m": -0.2,
                "radar_y_m": 2.1,
                "candidate_x_m": 0.0,
                "candidate_score": 0.7,
                "calibrated_visual_x_m": 0.2,
                "cross_modal_innovation": 0.4,
            },
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "project_policy_trace_to_evidence.py"),
            "--policy-causal-replay",
            str(policy),
            "--evidence-audit",
            str(evidence),
            "--output",
            str(output),
        ],
        check=True,
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["policy_admitted"] for row in rows] == ["False", "True"]
    assert [float(row["radar_x_m"]) for row in rows] == [-0.3, -0.2]
    assert [float(row["radar_y_m"]) for row in rows] == [2.1, 2.1]
    assert [float(row["candidate_x_m"]) for row in rows] == [-0.2, 0.0]
    assert [float(row["score"]) for row in rows] == [0.8, 0.7]
    assert [float(row["signed_innovation"]) for row in rows] == [-0.4, 0.4]
