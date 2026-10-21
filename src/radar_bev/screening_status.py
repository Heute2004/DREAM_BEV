from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EARLY_STOP_FILENAME = "screening_early_stop.json"
EARLY_STOP_STATUS = "early_stopped_converged"
EARLY_STOP_CRITERION = "no_ap_0p5m_improvement_for_completed_epochs"
MIN_EARLY_STOP_PATIENCE = 3


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _completed_history(history_path: Path, completed_epochs: int) -> list[dict[str, Any]]:
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(history, list):
        raise ValueError(f"Expected a list in {history_path}")
    rows_by_epoch: dict[int, dict[str, Any]] = {}
    for row in history:
        if not isinstance(row, dict):
            raise ValueError(f"Malformed history row in {history_path}")
        epoch = int(row.get("epoch", 0))
        if epoch < 1 or epoch in rows_by_epoch:
            raise ValueError(f"Invalid epoch sequence in {history_path}")
        rows_by_epoch[epoch] = row
    if set(rows_by_epoch) != set(range(1, completed_epochs + 1)):
        raise ValueError(f"History does not exactly cover completed epochs in {history_path}")
    return [rows_by_epoch[epoch] for epoch in range(1, completed_epochs + 1)]


def _validate_early_stop_payload(
    run_dir: Path,
    config: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    requested_epochs = int(config.get("train_epochs", 0))
    completed_epochs = int(payload.get("completed_epochs", 0))
    best_epoch = int(payload.get("best_epoch", 0))
    no_improvement_epochs = int(payload.get("no_improvement_epochs", -1))
    best_ap = float(payload.get("best_ap_0p5m", float("nan")))

    if payload.get("status") != EARLY_STOP_STATUS or payload.get("criterion") != EARLY_STOP_CRITERION:
        raise ValueError(f"Unexpected early-stop status in {run_dir}")
    if requested_epochs <= 0 or int(payload.get("requested_epochs", 0)) != requested_epochs:
        raise ValueError(f"Early-stop requested epoch mismatch in {run_dir}")
    if not 0 < completed_epochs < requested_epochs:
        raise ValueError(f"Early-stop completed epochs are invalid in {run_dir}")
    if not 1 <= best_epoch <= completed_epochs or not math.isfinite(best_ap):
        raise ValueError(f"Early-stop best metrics are invalid in {run_dir}")
    if no_improvement_epochs != completed_epochs - best_epoch or no_improvement_epochs < MIN_EARLY_STOP_PATIENCE:
        raise ValueError(f"Early-stop patience is insufficient in {run_dir}")

    history = _completed_history(run_dir / "history.json", completed_epochs)
    observed_aps = [float(row.get("eval_ap_0p5m", float("nan"))) for row in history]
    if not all(math.isfinite(value) for value in observed_aps):
        raise ValueError(f"Early-stop history is missing AP@0.5 in {run_dir}")
    observed_best = max(observed_aps)
    if not math.isclose(observed_best, best_ap, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Early-stop best AP does not match history in {run_dir}")
    if not math.isclose(observed_aps[best_epoch - 1], best_ap, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Early-stop best epoch does not match history in {run_dir}")

    metrics = _read_json_object(run_dir / "best_metrics.json")
    if int(metrics.get("epoch", 0)) != best_epoch or not math.isclose(
        float(metrics.get("ap_0p5m", float("nan"))), best_ap, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"Early-stop best checkpoint metrics do not match in {run_dir}")
    return payload


def get_verified_converged_early_stop(run_dir: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    marker = Path(run_dir) / EARLY_STOP_FILENAME
    if not marker.is_file():
        return None
    try:
        return _validate_early_stop_payload(Path(run_dir), config, _read_json_object(marker))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def completion_evidence_for_staging(run_dir: Path, config: dict[str, Any]) -> Path | None:
    completion = Path(run_dir) / "training_complete.json"
    if completion.is_file():
        return completion
    if get_verified_converged_early_stop(Path(run_dir), config) is not None:
        return Path(run_dir) / EARLY_STOP_FILENAME
    return None


def build_converged_early_stop_record(run_dir: Path, config: dict[str, Any], reason: str) -> dict[str, Any]:
    requested_epochs = int(config.get("train_epochs", 0))
    if requested_epochs <= 0:
        raise ValueError("train_epochs must be positive before recording an early stop")
    history = json.loads((Path(run_dir) / "history.json").read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history:
        raise ValueError("A nonempty history is required before recording an early stop")
    completed_epochs = max(int(row.get("epoch", 0)) for row in history if isinstance(row, dict))
    checked_history = _completed_history(Path(run_dir) / "history.json", completed_epochs)
    observed_aps = [float(row.get("eval_ap_0p5m", float("nan"))) for row in checked_history]
    if not all(math.isfinite(value) for value in observed_aps):
        raise ValueError("History is missing finite AP@0.5 values")
    best_ap = max(observed_aps)
    best_epoch = observed_aps.index(best_ap) + 1
    record = {
        "status": EARLY_STOP_STATUS,
        "criterion": EARLY_STOP_CRITERION,
        "requested_epochs": requested_epochs,
        "completed_epochs": completed_epochs,
        "best_epoch": best_epoch,
        "best_ap_0p5m": best_ap,
        "no_improvement_epochs": completed_epochs - best_epoch,
        "reason": reason,
        "stopped_utc": datetime.now(timezone.utc).isoformat(),
    }
    _validate_early_stop_payload(Path(run_dir), config, record)
    return record


def write_converged_early_stop_record(run_dir: Path, config: dict[str, Any], reason: str) -> Path:
    marker = Path(run_dir) / EARLY_STOP_FILENAME
    if marker.exists():
        raise FileExistsError(f"Refusing to overwrite early-stop evidence: {marker}")
    record = build_converged_early_stop_record(Path(run_dir), config, reason)
    marker.write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return marker



