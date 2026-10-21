from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_writes_a_route_only_counterfactual_trace(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "forced.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "policy_0032",
                "image": "Data_train/seq_082/RGB/frame_0032.jpg",
                "prediction": {"expert_route": "open_indoor_los"},
                "policy_source": "causal_response",
                "segment_rvfusion_admission": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "force_expert_route_trace.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(source),
            "--output",
            str(destination),
            "--route",
            "outdoor_wet_return",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(destination.read_text(encoding="utf-8"))
    assert output["prediction"]["expert_route"] == "outdoor_wet_return"
    assert output["segment_rvfusion_admission"] is True
