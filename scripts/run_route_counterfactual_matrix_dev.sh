#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?Set PYTHON to the experiment Python executable.}"
: "${FUSION_CODE:?Set FUSION_CODE to the RVFusion source directory.}"
: "${RADAR_CODE:?Set RADAR_CODE to the radar evaluator source directory.}"
: "${DATA_ROOT:?Set DATA_ROOT to the IF-only Data_train directory.}"
: "${MANIFEST:?Set MANIFEST to the Train/Development manifest.}"
: "${TRAIN_POLICY:?Set TRAIN_POLICY to the canonical train materialized trace.}"
: "${DEV_POLICY:?Set DEV_POLICY to the canonical Development materialized trace.}"
: "${VISUAL_CACHE:?Set VISUAL_CACHE to the Development visual cache.}"
: "${BEARING_CACHE:?Set BEARING_CACHE to the Development bearing cache.}"
: "${HEALTH_CACHE:?Set HEALTH_CACHE to the Development radar-health cache.}"
: "${BASE:?Set BASE to the frozen DREAM-BEV checkpoint.}"
: "${CLOR:?Set CLOR to the frozen CLOR checkpoint.}"
: "${TECS:?Set TECS to the frozen TECS checkpoint.}"
: "${LTGE:?Set LTGE to the frozen LTGE checkpoint.}"
: "${PB_CONFIG:?Set PB_CONFIG to the frozen PB-MHBR config.}"
: "${GG:?Set GG to the frozen GG-SRRE checkpoint.}"
: "${CANDIDATE:?Set CANDIDATE to the frozen RVFusion checkpoint.}"
: "${SEQUENCE_ROUTE_MAP:?Set SEQUENCE_ROUTE_MAP to the frozen Development sequence-to-route ledger.}"
: "${POLICY_CAUSAL_REPLAY:?Set POLICY_CAUSAL_REPLAY to the frozen Development segment trace.}"
: "${OUT_ROOT:?Set OUT_ROOT to a new Development-only output directory.}"

case "${MANIFEST}:${DEV_POLICY}:${OUT_ROOT}" in
  *locked*|*Locked*)
    echo "route counterfactual runner refuses locked-test paths" >&2
    exit 2
    ;;
esac

test ! -e "${OUT_ROOT}" || {
  echo "refusing to overwrite ${OUT_ROOT}" >&2
  exit 3
}

routes=(
  open_indoor_los
  indoor_clutter
  open_outdoor_weak_return
  outdoor_dry_complex
  outdoor_wet_return
)
gpus=(0 2 3)

mkdir -p "${OUT_ROOT}/traces" "${OUT_ROOT}/evaluations" "${OUT_ROOT}/logs"
export PYTHONPATH="${FUSION_CODE}/src:${RADAR_CODE}:${FUSION_CODE}"

for route in "${routes[@]}"; do
  "${PYTHON}" "${FUSION_CODE}/scripts/force_expert_route_trace.py" \
    --input "${DEV_POLICY}" \
    --output "${OUT_ROOT}/traces/${route}.jsonl" \
    --route "${route}"
done

common=(
  --data-root "${DATA_ROOT}"
  --manifest "${MANIFEST}"
  --visual-cache-root "${VISUAL_CACHE}"
  --rv-bearing-cache-root "${BEARING_CACHE}"
  --bearing-source rv_bearing_cache
  --radar-health-cache-root "${HEALTH_CACHE}"
  --train-condition-predictions "${TRAIN_POLICY}"
  --base-checkpoint "${BASE}"
  --expert-open-indoor-los "${CLOR}"
  --expert-indoor-clutter "${TECS}"
  --expert-open-outdoor-weak-return "${LTGE}"
  --expert-outdoor-dry-complex "${PB_CONFIG}"
  --expert-outdoor-wet-return "${GG}"
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
  # Every route arm replays the same deployed candidate-stage model.  The
  # forced trace changes only expert allocation; it must not revive the
  # superseded continuous/factorized path.
  --phase candidate
  --initial-checkpoint "${CANDIDATE}"
  --acceptance-threshold 0.5
  --acceptance-mode direct_utility
  --correction-source lattice
  --save-acceptance-audit
  --evaluation-only
  --development-only
  --reference-split "Development Val"
  --epochs 1
  --learning-rate 0.001
  --batch-size 12
  --num-workers 2
  --sequence-cache-size 2
  --device cuda:0
)

pids=()
for index in "${!routes[@]}"; do
  route="${routes[$index]}"
  gpu="${gpus[$((index % ${#gpus[@]}))]}"
  output="${OUT_ROOT}/evaluations/${route}"
  log="${OUT_ROOT}/logs/${route}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    "${FUSION_CODE}/scripts/train_fusion.py" \
    "${common[@]}" \
    --eval-condition-predictions "${OUT_ROOT}/traces/${route}.jsonl" \
    --output-dir "${output}" \
    >"${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
test "${status}" -eq 0 || exit "${status}"

for route in "${routes[@]}"; do
  output="${OUT_ROOT}/evaluations/${route}"
  projected_trace="${output}/forced_route_policy_replay.csv"
  guarded="${output}/guarded"
  test -f "${output}/evaluation_only.json"
  test -f "${output}/acceptance_audit.csv"
  # Preserve frozen segment admission while replacing only the expert-radar
  # anchor emitted by the forced route, then apply the deployed guard.
  "${PYTHON}" "${FUSION_CODE}/scripts/project_policy_trace_to_evidence.py" \
    --policy-causal-replay "${POLICY_CAUSAL_REPLAY}" \
    --evidence-audit "${output}/acceptance_audit.csv" \
    --output "${projected_trace}"
  "${PYTHON}" "${FUSION_CODE}/scripts/replay_causal_cross_modal_guard.py" \
    --policy-causal-replay "${projected_trace}" \
    --evidence-audit "${output}/acceptance_audit.csv" \
    --output-dir "${guarded}" \
    --ema-alpha 0.10 \
    --ema-threshold 0.50 \
    --admission-mode learned \
    --warmup-frames 32
  test -f "${guarded}/causal_guard_replay.csv"
  test -f "${guarded}/metrics.json"
  "${PYTHON}" "${FUSION_CODE}/scripts/summarize_route_counterfactual.py" \
    --audit-csv "${guarded}/causal_guard_replay.csv" \
    --reference-trace "${DEV_POLICY}" \
    --sequence-route-map "${SEQUENCE_ROUTE_MAP}" \
    --output "${output}/metrics_by_ledger_route.json"
done

"${PYTHON}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
matrix = {}
for evaluation in sorted((root / "evaluations").iterdir()):
    metrics = json.loads((evaluation / "metrics_by_ledger_route.json").read_text())
    for ledger_route, values in metrics.items():
        matrix.setdefault(ledger_route, {})[evaluation.name] = values
(root / "route_counterfactual_matrix_ledger.json").write_text(
    json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY

sha256sum "${MANIFEST}" "${TRAIN_POLICY}" "${DEV_POLICY}" "${SEQUENCE_ROUTE_MAP}" "${POLICY_CAUSAL_REPLAY}" "${BASE}" \
  "${CLOR}" "${TECS}" "${LTGE}" "${PB_CONFIG}" "${GG}" "${CANDIDATE}" \
  "${OUT_ROOT}/traces/"*.jsonl > "${OUT_ROOT}/SHA256SUMS.txt"
echo "Development route counterfactual matrix complete: ${OUT_ROOT}"
