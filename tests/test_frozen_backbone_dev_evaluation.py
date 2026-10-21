from __future__ import annotations

from pathlib import Path

import torch

from radar_bev.frozen_backbone_evaluation import evaluate_frozen_backbone
from radar_bev.geometry import BEVGeometry


class _FixedPeakModel(torch.nn.Module):
    def __init__(self, x_index: int, y_index: int) -> None:
        super().__init__()
        self.x_index = x_index
        self.y_index = y_index

    def forward(self, radar: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = torch.full((radar.shape[0], 1, 200, 234), -20.0)
        logits[:, 0, self.x_index, self.y_index] = 20.0
        return {"logits": logits}


def test_frozen_backbone_evaluation_reports_strict_metrics_and_frame_identity() -> None:
    geometry = BEVGeometry()
    target_index = torch.tensor([[100, 20]])
    target_xy = geometry.index_to_metric(target_index)
    loader = [
        {
            "radar": torch.zeros(1, 16, 2, 128, 128),
            "target_xy_m": target_xy,
            "sequence": ["seq_001"],
            "frame_id": torch.tensor([7]),
        }
    ]

    result = evaluate_frozen_backbone(
        _FixedPeakModel(100, 20), loader, torch.device("cpu"), latency_warmup_batches=1
    )

    assert result.metrics["ap_0p25m"] == 1.0
    assert result.metrics["ap_0p3m"] == 1.0
    assert result.metrics["ap_0p5m"] == 1.0
    assert result.metrics["ap_1p0m"] == 1.0
    assert result.rows == [
        {
            "sequence": "seq_001",
            "frame_id": 7,
            "score": result.rows[0]["score"],
            "prediction_x_m": 0.0,
            "prediction_y_m": 2.0,
            "target_x_m": 0.0,
            "target_y_m": 2.0,
            "error_m": 0.0,
        }
    ]
    assert result.runtime["measured_batches"] == 0


def test_strict_development_evaluator_is_directly_invocable() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "evaluate_backbone_strict_dev.py"
    assert runner.is_file()


def test_strict_development_evaluator_accepts_erodnet() -> None:
    runner = Path(__file__).parents[1] / "scripts" / "evaluate_backbone_strict_dev.py"

    assert '"e_rodnet"' in runner.read_text(encoding="utf-8")
