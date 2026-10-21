"""CRUW sensor-grid access without importing RODNet training code."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def load_cruw_grids(data_root: str | Path, devkit_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = str(Path(devkit_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from cruw import CRUW

    cruw = CRUW(data_root=str(Path(data_root)), sensor_config_name="sensor_config_rod2021")
    return np.asarray(cruw.range_grid, dtype=np.float32), np.asarray(cruw.angle_grid, dtype=np.float32)
