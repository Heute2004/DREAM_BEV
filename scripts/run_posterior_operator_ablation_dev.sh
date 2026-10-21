#!/usr/bin/env bash
# Development-only operator study for the frozen RVFusion posterior.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CUDA_DEVICE OUTPUT_ROOT" >&2
  exit 2
fi

CUDA_DEVICE="$1"
OUTPUT_ROOT="$2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
case "$OUTPUT_ROOT" in
  *locked*|*Locked*|*test*|*Test*)
    echo "Locked Test is forbidden for the posterior operator study." >&2
    exit 3
    ;;
esac
test ! -e "$OUTPUT_ROOT" || {
  echo "refusing to overwrite $OUTPUT_ROOT" >&2
  exit 4
}

require_env() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || {
    echo "missing required environment variable: $name" >&2
    exit 5
  }
  printf '%s' "$value"
}

FUSION_CODE="$(require_env FUSION_CODE)"
RADAR_CODE="$(require_env RADAR_CODE)"
PYTHON="$(require_env PYTHON)"
DATA_ROOT="$(require_env DATA_ROOT)"
MANIFEST="$(require_env MANIFEST)"
TRAIN_POLICY="$(require_env TRAIN_POLICY)"
DEV_POLICY="$(require_env DEV_POLICY)"
VISUAL_CACHE_ROOT="$(require_env VISUAL_CACHE_ROOT)"
RV_BEARING_CACHE_ROOT="$(require_env RV_BEARING_CACHE_ROOT)"
HEALTH_CACHE_ROOT="$(require_env HEALTH_CACHE_ROOT)"
BASE_CHECKPOINT="$(require_env BASE_CHECKPOINT)"
CLOR_CHECKPOINT="$(require_env CLOR_CHECKPOINT)"
TECS_CHECKPOINT="$(require_env TECS_CHECKPOINT)"
LTGE_CHECKPOINT="$(require_env LTGE_CHECKPOINT)"
PB_CONFIGURATION="$(require_env PB_CONFIGURATION)"
GG_CHECKPOINT="$(require_env GG_CHECKPOINT)"
CANDIDATE_CHECKPOINT="$(require_env CANDIDATE_CHECKPOINT)"
POLICY_CAUSAL_REPLAY="$(require_env POLICY_CAUSAL_REPLAY)"
BASELINE_EVIDENCE_AUDIT="$(require_env BASELINE_EVIDENCE_AUDIT)"

for path in \
  "$PYTHON" "$FUSION_CODE" "$RADAR_CODE" "$MANIFEST" \
  "$TRAIN_POLICY" "$DEV_POLICY" "$VISUAL_CACHE_ROOT" \
  "$RV_BEARING_CACHE_ROOT/cache_manifest.json" "$HEALTH_CACHE_ROOT" \
  "$BASE_CHECKPOINT" "$CLOR_CHECKPOINT" "$TECS_CHECKPOINT" \
  "$LTGE_CHECKPOINT" "$PB_CONFIGURATION" "$GG_CHECKPOINT" \
  "$CANDIDATE_CHECKPOINT" "$POLICY_CAUSAL_REPLAY" \
  "$BASELINE_EVIDENCE_AUDIT"; do
  test -e "$path" || { echo "missing required path: $path" >&2; exit 5; }
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

mkdir -p "$OUTPUT_ROOT/logs"
# train_fusion.py runs from the deployed fusion checkout, while the new guard
# replay lives in this isolated study checkout. Both source roots are needed.
export PYTHONPATH="$STUDY_CODE_ROOT/src:$FUSION_CODE/src:$RADAR_CODE:$FUSION_CODE${PYTHONPATH:+:$PYTHONPATH}"

common=(
  --data-root "$DATA_ROOT"
  --manifest "$MANIFEST"
  --visual-cache-root "$VISUAL_CACHE_ROOT"
  --rv-bearing-cache-root "$RV_BEARING_CACHE_ROOT"
  --bearing-source rv_bearing_cache
  --radar-health-cache-root "$HEALTH_CACHE_ROOT"
  --train-condition-predictions "$TRAIN_POLICY"
  --eval-condition-predictions "$DEV_POLICY"
  --base-checkpoint "$BASE_CHECKPOINT"
  --expert-open-indoor-los "$CLOR_CHECKPOINT"
  --expert-indoor-clutter "$TECS_CHECKPOINT"
  --expert-open-outdoor-weak-return "$LTGE_CHECKPOINT"
  --expert-outdoor-dry-complex "$PB_CONFIGURATION"
  --expert-outdoor-wet-return "$GG_CHECKPOINT"
  --gg-recovery-scale 1.15
  --vlm-trust-policy normal_or_rain_nonunreliable
  --assist-policy vlm_segment
  --candidate-geometry dense_local
  --candidate-prior-mode none
  --posterior-temperature 0.5
  --candidate-target-mode nearest_x
  --candidate-target-sigma-m 0.05
  --candidate-positive-weight 10
  --candidate-presence-weight 1
  --acceptance-threshold 0.5
  --acceptance-mode direct_utility
  --correction-source lattice
  --continuous-location-weight 1
  --batch-size 12
  --num-workers 2
  --sequence-cache-size 2
  --seed 3407
  --device "$CUDA_DEVICE"
  --reference-split "Development Val"
  --development-only
  --save-acceptance-audit
)

cd "$FUSION_CODE"

# The sole new learned comparator: fixed bins and bounded lateral residuals,
# trained on Train and checkpointed on Development under the same inputs.
quantized_train="$OUTPUT_ROOT/quantized_lateral_train"
"$PYTHON" -u scripts/train_fusion.py \
  "${common[@]}" \
  --fusion-core quantized_lateral_residual \
  --phase candidate \
  --epochs 4 \
  --learning-rate 0.0003 \
  --output-dir "$quantized_train" \
  > "$OUTPUT_ROOT/logs/quantized_train.log" 2>&1
test -f "$quantized_train/best_unconstrained.pt" || {
  echo "quantized residual did not produce a Development checkpoint" >&2
  exit 6
}

quantized_eval="$OUTPUT_ROOT/quantized_lateral_evaluation"
"$PYTHON" -u scripts/train_fusion.py \
  "${common[@]}" \
  --fusion-core quantized_lateral_residual \
  --phase joint \
  --initial-checkpoint "$quantized_train/best_unconstrained.pt" \
  --epochs 1 \
  --learning-rate 0.0003 \
  --evaluation-only \
  --output-dir "$quantized_eval" \
  > "$OUTPUT_ROOT/logs/quantized_evaluation.log" 2>&1
test -f "$quantized_eval/acceptance_audit.csv"

# Freeze the deployed posterior and remove only its factorized no-op branch.
no_op_eval="$OUTPUT_ROOT/local_posterior_no_op_removed"
"$PYTHON" -u scripts/train_fusion.py \
  "${common[@]}" \
  --fusion-core local_posterior \
  --phase candidate \
  --initial-checkpoint "$CANDIDATE_CHECKPOINT" \
  --epochs 1 \
  --learning-rate 0.001 \
  --evaluation-only \
  --disable-posterior-no-op \
  --output-dir "$no_op_eval" \
  > "$OUTPUT_ROOT/logs/no_op_removed_evaluation.log" 2>&1
test -f "$no_op_eval/evaluation_only.json"
test -f "$no_op_eval/acceptance_audit.csv"
if find "$no_op_eval" -type f -name '*.pt' | grep -q .; then
  echo "frozen no-op operator study wrote a checkpoint" >&2
  exit 7
fi

replay_guard() {
  local name="$1"
  local evidence="$2"
  local extra="${3:-}"
  local out="$OUTPUT_ROOT/$name"
  test ! -e "$out" || { echo "refusing to overwrite $out" >&2; exit 8; }
  # shellcheck disable=SC2086
  "$PYTHON" scripts/replay_causal_cross_modal_guard.py \
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
replay_guard "posterior_no_op_removed_guarded" "$no_op_eval/acceptance_audit.csv"
replay_guard "quantized_residual_guarded" "$quantized_eval/acceptance_audit.csv"

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

sha256sum "$MANIFEST" "$TRAIN_POLICY" "$DEV_POLICY" \
  "$CANDIDATE_CHECKPOINT" "$POLICY_CAUSAL_REPLAY" \
  "$BASELINE_EVIDENCE_AUDIT" > "$OUTPUT_ROOT/FROZEN_INPUT_SHA256SUMS.txt"
