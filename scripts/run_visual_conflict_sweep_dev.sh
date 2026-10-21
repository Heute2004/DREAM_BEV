#!/usr/bin/env bash
# Run the frozen Development-only visual-conflict severity sweep.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_CSV OUTPUT_ROOT" >&2
  exit 2
fi

GPU_CSV="$1"
OUTPUT_ROOT="$2"
case "$OUTPUT_ROOT" in
  *locked_test*|*locked-test*)
    echo "Locked Test is forbidden for the visual-conflict sweep." >&2
    exit 3
    ;;
esac

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_ROOT/.." && pwd)"
SINGLE_RUNNER="$SCRIPT_ROOT/run_bearing_stress_evaluation_dev.sh"
REPLAY="$SCRIPT_ROOT/replay_causal_cross_modal_guard.py"
SUMMARIZER="$SCRIPT_ROOT/summarize_visual_conflict_sweep.py"
EMA_ALPHA="0.10"
EMA_THRESHOLD="0.50"
BIAS_LEVELS=(1 2 3 4 5 6)
LAG_LEVELS=(2 4 8)
DROPOUT_LEVELS=(0.25 0.50 0.75)

require_env() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || { echo "missing required environment variable: $name" >&2; exit 4; }
  printf '%s' "$value"
}

PYTHON="$(require_env PYTHON)"
FUSION_CODE="$(require_env FUSION_CODE)"
RADAR_CODE="$(require_env RADAR_CODE)"
POLICY_CAUSAL_REPLAY="$(require_env POLICY_CAUSAL_REPLAY)"
CLEAN_GUARD_OFF_METRICS="$(require_env CLEAN_GUARD_OFF_METRICS)"
CLEAN_GUARDED_METRICS="$(require_env CLEAN_GUARDED_METRICS)"

for path in "$PYTHON" "$SINGLE_RUNNER" "$REPLAY" "$SUMMARIZER" \
  "$POLICY_CAUSAL_REPLAY" "$CLEAN_GUARD_OFF_METRICS" "$CLEAN_GUARDED_METRICS"; do
  test -e "$path" || { echo "missing required path: $path" >&2; exit 4; }
done
test ! -e "$OUTPUT_ROOT" || { echo "refusing to overwrite $OUTPUT_ROOT" >&2; exit 5; }

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
[[ ${#GPUS[@]} -gt 0 ]] || { echo "at least one GPU is required" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^cuda:[0-9]+$ ]] || { echo "invalid GPU device: $gpu" >&2; exit 2; }
done

POINTS=()
for level in "${BIAS_LEVELS[@]}"; do POINTS+=("bearing_bias|$level|--bias-deg|$level"); done
for level in "${LAG_LEVELS[@]}"; do POINTS+=("temporal_lag|$level|--lag-frames|$level"); done
for level in "${DROPOUT_LEVELS[@]}"; do
  label="$("$PYTHON" -c "print(int(float('$level') * 100))")"
  POINTS+=("visual_dropout|$label|--dropout-probability|$level")
done

mkdir -p "$OUTPUT_ROOT/logs"
export PYTHONPATH="$PROJECT_ROOT/src:$FUSION_CODE/src:$RADAR_CODE:$FUSION_CODE"

run_point() {
  local point="$1"
  local gpu="$2"
  IFS='|' read -r mode level argument value <<< "$point"
  local point_root="$OUTPUT_ROOT/raw/$mode/$level"
  local audit="$point_root/evaluation/$mode/acceptance_audit.csv"
  local replay_root="$OUTPUT_ROOT/replay/$mode/$level"
  "$SINGLE_RUNNER" "$mode" "$gpu" "$point_root" "$argument" "$value"
  test -f "$audit" || { echo "missing acceptance audit: $audit" >&2; exit 7; }
  "$PYTHON" "$REPLAY" \
    --policy-causal-replay "$POLICY_CAUSAL_REPLAY" \
    --evidence-audit "$audit" \
    --output-dir "$replay_root/guard_off" \
    --ema-alpha "$EMA_ALPHA" \
    --ema-threshold "$EMA_THRESHOLD" \
    --disable-guard \
    > "$OUTPUT_ROOT/logs/${mode}_${level}_guard_off.log" 2>&1
  "$PYTHON" "$REPLAY" \
    --policy-causal-replay "$POLICY_CAUSAL_REPLAY" \
    --evidence-audit "$audit" \
    --output-dir "$replay_root/protected" \
    --ema-alpha "$EMA_ALPHA" \
    --ema-threshold "$EMA_THRESHOLD" \
    > "$OUTPUT_ROOT/logs/${mode}_${level}_protected.log" 2>&1
}

for ((offset=0; offset<${#POINTS[@]}; offset+=${#GPUS[@]})); do
  pids=()
  for ((slot=0; slot<${#GPUS[@]} && offset+slot<${#POINTS[@]}; slot++)); do
    run_point "${POINTS[$((offset+slot))]}" "${GPUS[$slot]}" \
      > "$OUTPUT_ROOT/logs/point_$((offset+slot)).log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
done

"$PYTHON" "$SUMMARIZER" \
  --run-root "$OUTPUT_ROOT" \
  --clean-guard-off "$CLEAN_GUARD_OFF_METRICS" \
  --clean-protected "$CLEAN_GUARDED_METRICS" \
  --output-dir "$OUTPUT_ROOT/summary" \
  --expected-frames 9969

find "$OUTPUT_ROOT" -type f -name '*.pt' -print -quit | grep -q . && {
  echo "evaluation-only sweep wrote a checkpoint" >&2
  exit 8
}
