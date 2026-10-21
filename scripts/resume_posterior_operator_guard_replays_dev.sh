#!/usr/bin/env bash
# Resume only the guard replays of a completed Development-only operator study.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

OUTPUT_ROOT="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

case "$OUTPUT_ROOT" in
  *locked*|*Locked*|*test*|*Test*)
    echo "Locked Test is forbidden for the posterior operator study." >&2
    exit 3
    ;;
esac

require_env() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || {
    echo "missing required environment variable: $name" >&2
    exit 4
  }
  printf '%s' "$value"
}

FUSION_CODE="$(require_env FUSION_CODE)"
RADAR_CODE="$(require_env RADAR_CODE)"
PYTHON="$(require_env PYTHON)"
MANIFEST="$(require_env MANIFEST)"
POLICY_CAUSAL_REPLAY="$(require_env POLICY_CAUSAL_REPLAY)"
BASELINE_EVIDENCE_AUDIT="$(require_env BASELINE_EVIDENCE_AUDIT)"

for path in \
  "$PYTHON" "$FUSION_CODE" "$RADAR_CODE" "$MANIFEST" \
  "$POLICY_CAUSAL_REPLAY" "$BASELINE_EVIDENCE_AUDIT" \
  "$OUTPUT_ROOT/quantized_lateral_evaluation/acceptance_audit.csv" \
  "$OUTPUT_ROOT/local_posterior_no_op_removed/acceptance_audit.csv"; do
  test -e "$path" || { echo "missing required path: $path" >&2; exit 4; }
done

"$PYTHON" - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8-sig", newline="")))
if len(rows) != 36_300 or {row["split"] for row in rows} != {"Train", "Development Val"}:
    raise RuntimeError("expected fixed IF-only Train/Development manifest")
if any(int(row["sequence"].split("_")[1]) <= 21 for row in rows):
    raise RuntimeError("L-corridor leaked into IF-only manifest")
PY

export PYTHONPATH="$STUDY_CODE_ROOT/src:$FUSION_CODE/src:$RADAR_CODE:$FUSION_CODE${PYTHONPATH:+:$PYTHONPATH}"

replay_guard() {
  local name="$1"
  local evidence="$2"
  local extra="${3:-}"
  local out="$OUTPUT_ROOT/$name"
  test ! -e "$out" || { echo "refusing to overwrite $out" >&2; exit 5; }
  # shellcheck disable=SC2086
  "$PYTHON" "$STUDY_CODE_ROOT/scripts/replay_causal_cross_modal_guard.py" \
    --policy-causal-replay "$POLICY_CAUSAL_REPLAY" \
    --evidence-audit "$evidence" \
    --output-dir "$out" \
    --ema-alpha 0.10 \
    --ema-threshold 0.50 \
    --admission-mode learned \
    --warmup-frames 32 \
    $extra \
    > "$OUTPUT_ROOT/logs/${name}.log" 2>&1
  test -f "$out/metrics.json"
  test -f "$out/causal_guard_replay.csv"
}

replay_guard "posterior_guard_off" "$BASELINE_EVIDENCE_AUDIT" "--disable-guard"
replay_guard "posterior_guarded" "$BASELINE_EVIDENCE_AUDIT"
replay_guard "posterior_no_op_removed_guarded" "$OUTPUT_ROOT/local_posterior_no_op_removed/acceptance_audit.csv"
replay_guard "quantized_residual_guarded" "$OUTPUT_ROOT/quantized_lateral_evaluation/acceptance_audit.csv"

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = {
    "posterior_guard_off": root / "posterior_guard_off" / "metrics.json",
    "posterior_guarded": root / "posterior_guarded" / "metrics.json",
    "posterior_no_op_removed_guarded": root / "posterior_no_op_removed_guarded" / "metrics.json",
    "quantized_residual_guarded": root / "quantized_residual_guarded" / "metrics.json",
}
summary = {}
for name, path in arms.items():
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary[name] = {
        "frames": payload["frames"],
        "ap_0p25m": payload["guarded_fusion"]["ap_0p25m"],
        "ap_0p3m": payload["guarded_fusion"]["ap_0p3m"],
        "ap_0p5m": payload["guarded_fusion"]["ap_0p5m"],
        "p90_location_error_m": payload["guarded_fusion"]["p90_location_error_m"],
        "visual_acceptance_rate": payload["visual_acceptance_rate"],
        "harmed_rate": payload["harmed_rate"],
    }
(root / "operator_study_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

sha256sum "$MANIFEST" "$POLICY_CAUSAL_REPLAY" "$BASELINE_EVIDENCE_AUDIT" \
  "$OUTPUT_ROOT/quantized_lateral_evaluation/acceptance_audit.csv" \
  "$OUTPUT_ROOT/local_posterior_no_op_removed/acceptance_audit.csv" \
  "$OUTPUT_ROOT/posterior_guard_off/metrics.json" \
  "$OUTPUT_ROOT/posterior_guarded/metrics.json" \
  "$OUTPUT_ROOT/posterior_no_op_removed_guarded/metrics.json" \
  "$OUTPUT_ROOT/quantized_residual_guarded/metrics.json" \
  > "$OUTPUT_ROOT/FROZEN_INPUT_AND_RESULT_SHA256SUMS.txt"
