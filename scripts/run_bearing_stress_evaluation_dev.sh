#!/usr/bin/env bash
# Evaluate one frozen Development-only RVFusion checkpoint under bearing stress.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 MODE GPU OUTPUT_ROOT [mode arguments]" >&2
  exit 2
fi

MODE="$1"
CUDA_DEVICE="$2"
OUTPUT_ROOT="$3"
shift 3

case "$MODE" in
  bearing_bias|temporal_lag|visual_dropout) ;;
  *) echo "unsupported stress mode: $MODE" >&2; exit 2 ;;
esac
case "$OUTPUT_ROOT" in
  *locked_test*|*locked-test*)
    echo "Locked Test is forbidden for bearing-stress evaluation." >&2
    exit 3
    ;;
esac

require_env() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || { echo "missing required environment variable: $name" >&2; exit 4; }
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
SOURCE_BEARING="$(require_env SOURCE_BEARING_CACHE_ROOT)"
HEALTH_CACHE_ROOT="$(require_env HEALTH_CACHE_ROOT)"
BASE_CHECKPOINT="$(require_env BASE_CHECKPOINT)"
CLOR_CHECKPOINT="$(require_env CLOR_CHECKPOINT)"
TECS_CHECKPOINT="$(require_env TECS_CHECKPOINT)"
LTGE_CHECKPOINT="$(require_env LTGE_CHECKPOINT)"
PB_CONFIGURATION="$(require_env PB_CONFIGURATION)"
GG_CHECKPOINT="$(require_env GG_CHECKPOINT)"
CANDIDATE="$(require_env CANDIDATE_CHECKPOINT)"
STRESS_CACHE="$OUTPUT_ROOT/cache/$MODE"
EVALUATION="$OUTPUT_ROOT/evaluation/$MODE"

for path in "$PYTHON" "$FUSION_CODE" "$RADAR_CODE" "$MANIFEST" "$TRAIN_POLICY" "$DEV_POLICY" "$CANDIDATE" "$SOURCE_BEARING/cache_manifest.json"; do
  test -e "$path" || { echo "missing required path: $path" >&2; exit 4; }
done
test ! -e "$STRESS_CACHE" || { echo "refusing to overwrite $STRESS_CACHE" >&2; exit 5; }
test ! -e "$EVALUATION" || { echo "refusing to overwrite $EVALUATION" >&2; exit 5; }

"$PYTHON" - "$MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8-sig", newline="")))
if len(rows) != 36_300 or {row["split"] for row in rows} != {"Train", "Development Val"}:
    raise RuntimeError("expected fixed Train/Development manifest")
if any(int(row["sequence"].split("_")[1]) <= 21 for row in rows):
    raise RuntimeError("L-corridor leaked into IF-only manifest")
PY

mkdir -p "$OUTPUT_ROOT/logs"
# The cache builder imports the study package too, so its import path must be
# available before the stress cache is materialized, not just for evaluation.
export PYTHONPATH="$FUSION_CODE/src:$RADAR_CODE:$FUSION_CODE"
export LD_LIBRARY_PATH="/home/hyc/.conda/envs/rvmg/lib:${LD_LIBRARY_PATH:-}"
"$PYTHON" "$FUSION_CODE/scripts/build_bearing_stress_cache.py" \
  --source-cache-root "$SOURCE_BEARING" \
  --output-cache-root "$STRESS_CACHE" \
  --mode "$MODE" "$@" \
  > "$OUTPUT_ROOT/logs/${MODE}_cache.log" 2>&1

cd "$FUSION_CODE"
# Stress evaluation uses the deployed candidate-stage contract; only the
# materialized visual evidence is perturbed.
"$PYTHON" -u scripts/train_fusion.py \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --visual-cache-root "$VISUAL_CACHE_ROOT" \
  --rv-bearing-cache-root "$STRESS_CACHE" \
  --bearing-source rv_bearing_cache \
  --radar-health-cache-root "$HEALTH_CACHE_ROOT" \
  --train-condition-predictions "$TRAIN_POLICY" \
  --eval-condition-predictions "$DEV_POLICY" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --expert-open-indoor-los "$CLOR_CHECKPOINT" \
  --expert-indoor-clutter "$TECS_CHECKPOINT" \
  --expert-open-outdoor-weak-return "$LTGE_CHECKPOINT" \
  --expert-outdoor-dry-complex "$PB_CONFIGURATION" \
  --expert-outdoor-wet-return "$GG_CHECKPOINT" \
  --gg-recovery-scale 1.15 \
  --vlm-trust-policy normal_or_rain_nonunreliable \
  --assist-policy vlm_segment \
  --candidate-geometry dense_local \
  --candidate-prior-mode none \
  --posterior-temperature 0.5 \
  --candidate-target-mode nearest_x \
  --candidate-target-sigma-m 0.05 \
  --candidate-positive-weight 10 \
  --candidate-presence-weight 1 \
  --num-workers 2 \
  --sequence-cache-size 2 \
  --device "$CUDA_DEVICE" \
  --phase candidate \
  --initial-checkpoint "$CANDIDATE" \
  --acceptance-threshold 0.5 \
  --acceptance-mode direct_utility \
  --correction-source lattice \
  --save-acceptance-audit \
  --evaluation-only \
  --development-only \
  --reference-split "Development Val" \
  --output-dir "$EVALUATION" \
  --epochs 1 \
  --learning-rate 0.001 \
  --batch-size 12 \
  > "$OUTPUT_ROOT/logs/${MODE}_evaluation.log" 2>&1

test -f "$EVALUATION/evaluation_only.json"
if find "$EVALUATION" -type f -name '*.pt' | grep -q .; then
  echo "evaluation-only stress run wrote a checkpoint" >&2
  exit 6
fi
sha256sum "$MANIFEST" "$TRAIN_POLICY" "$DEV_POLICY" "$CANDIDATE" \
  "$STRESS_CACHE/cache_manifest.json" \
  "$STRESS_CACHE/bearing_stress_metadata.json" > "$EVALUATION/FROZEN_INPUT_SHA256SUMS.txt"
