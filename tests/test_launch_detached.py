from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_launcher():
    path = PROJECT_ROOT / "scripts" / "launch_detached.py"
    spec = importlib.util.spec_from_file_location("launch_detached", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_launch_detached_redirects_child_output_to_requested_log(tmp_path: Path) -> None:
    launcher = _load_launcher()
    log_path = tmp_path / "child.log"

    pid = launcher.launch_detached(
        [sys.executable, "-c", "print('detached-ready')"],
        log_path,
    )

    assert isinstance(pid, int) and pid > 0
    for _ in range(30):
        if log_path.is_file() and "detached-ready" in log_path.read_text(encoding="utf-8"):
            break
        time.sleep(0.1)
    assert log_path.read_text(encoding="utf-8").strip() == "detached-ready"
