#!/usr/bin/env bash
# Replay interchangeable policy backends through one frozen RVFusion contract.
set -euo pipefail

: "${PYTHON:?Set PYTHON to the experiment Python executable.}"
: "${FUSION_CODE:?Set FUSION_CODE to the RVFusion source directory.}"
: "${RADAR_CODE:?Set RADAR_CODE to the radar evaluator source directory.}"
: "${DATA_ROOT:?Set DATA_ROOT to the IF-only Data_train directory.}"
: "${MANIFEST:?Set MANIFEST to the Train/Development manifest.}"
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
: "${VLM_TRAIN_POLICY:?Set VLM_TRAIN_POLICY to the semantic Train trace.}"
: "${VLM_DEV_POLICY:?Set VLM_DEV_POLICY to the semantic Development trace.}"
: "${MOBILENET_TRAIN_POLICY:?Set MOBILENET_TRAIN_POLICY to the efficient-CNN Train trace.}"
: "${MOBILENET_DEV_POLICY:?Set MOBILENET_DEV_POLICY to the efficient-CNN Development trace.}"
: "${RESNET_TRAIN_POLICY:?Set RESNET_TRAIN_POLICY to the capacity-CNN Train trace.}"
: "${RESNET_DEV_POLICY:?Set RESNET_DEV_POLICY to the capacity-CNN Development trace.}"
: "${OUT_ROOT:?Set OUT_ROOT to a new Development-only output directory.}"

case "${MANIFEST}:${OUT_ROOT}" in
  *locked*|*Locked*|*test*|*Test*)
    echo "policy frontier runner refuses Locked Test paths" >&2
    exit 2
    ;;
esac

test ! -e "${OUT_ROOT}" || {
  echo "refusing to overwrite ${OUT_ROOT}" >&2
  exit 3
}

"${PYTHON}" - "${MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="utf-8-sig", newline="")))
if len(rows) != 36_300 or {row["split"] for row in rows} != {"Train", "Development Val"}:
    raise RuntimeError("expected fixed IF-only Train/Development manifest")
if any(int(row["sequence"].split("_")[1]) <= 21 for row in rows):
    raise RuntimeError("L-corridor leaked into IF-only manifest")
PY

mkdir -p "${OUT_ROOT}/evaluations" "${OUT_ROOT}/guarded" "${OUT_ROOT}/logs"
export PYTHONPATH="${FUSION_CODE}/src:${RADAR_CODE}:${FUSION_CODE}${PYTHONPATH:+:${PYTHONPATH}}"

common=(
  --data-root "${DATA_ROOT}"
  --manifest "${MANIFEST}"
  --visual-cache-root "${VISUAL_CACHE}"
  --rv-bearing-cache-root "${BEARING_CACHE}"
  --bearing-source rv_bearing_cache
  --radar-health-cache-root "${HEALTH_CACHE}"
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
)

names=(semantic_vlm efficient_mobilenet capacity_resnet)
train_policies=("${VLM_TRAIN_POLICY}" "${MOBILENET_TRAIN_POLICY}" "${RESNET_TRAIN_POLICY}")
dev_policies=("${VLM_DEV_POLICY}" "${MOBILENET_DEV_POLICY}" "${RESNET_DEV_POLICY}")
gpus=(0 2 3)
pids=()

for index in "${!names[@]}"; do
  name="${names[$index]}"
  CUDA_VISIBLE_DEVICES="${gpus[$index]}" "${PYTHON}" -u "${FUSION_CODE}/scripts/train_fusion.py" \
    "${common[@]}" \
    --train-condition-predictions "${train_policies[$index]}" \
    --eval-condition-predictions "${dev_policies[$index]}" \
    --device cuda:0 \
    --output-dir "${OUT_ROOT}/evaluations/${name}" \
    > "${OUT_ROOT}/logs/${name}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
test "${status}" -eq 0 || exit "${status}"

for name in "${names[@]}"; do
  audit="${OUT_ROOT}/evaluations/${name}/acceptance_audit.csv"
  out="${OUT_ROOT}/guarded/${name}"
  test -f "${OUT_ROOT}/evaluations/${name}/evaluation_only.json"
  test -f "${audit}"
  "${PYTHON}" "${FUSION_CODE}/scripts/replay_causal_cross_modal_guard.py" \
    --policy-audit "${audit}" \
    --evidence-audit "${audit}" \
    --output-dir "${out}" \
    --ema-alpha 0.10 \
    --ema-threshold 0.50 \
    --admission-mode learned \
    --warmup-frames 32 \
    > "${OUT_ROOT}/logs/${name}_guard.log" 2>&1
  test -f "${out}/metrics.json"
  test -f "${out}/causal_guard_replay.csv"
done

"${PYTHON}" - "${OUT_ROOT}" "${names[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {}
for name in sys.argv[2:]:
    payload = json.loads((root / "guarded" / name / "metrics.json").read_text())
    summary[name] = {
        "frames": payload["frames"],
        "ap_0p25m": payload["guarded_fusion"]["ap_0p25m"],
        "ap_0p3m": payload["guarded_fusion"]["ap_0p3m"],
        "ap_0p5m": payload["guarded_fusion"]["ap_0p5m"],
        "p90_location_error_m": payload["guarded_fusion"]["p90_location_error_m"],
        "visual_acceptance_rate": payload["visual_acceptance_rate"],
        "harmed_rate": payload["harmed_rate"],
    }
(root / "policy_backend_frontier_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

sha256sum "${MANIFEST}" "${BASE}" "${CLOR}" "${TECS}" "${LTGE}" "${PB_CONFIG}" "${GG}" "${CANDIDATE}" \
  "${VLM_TRAIN_POLICY}" "${VLM_DEV_POLICY}" \
  "${MOBILENET_TRAIN_POLICY}" "${MOBILENET_DEV_POLICY}" \
  "${RESNET_TRAIN_POLICY}" "${RESNET_DEV_POLICY}" \
  "${OUT_ROOT}/evaluations/"*/acceptance_audit.csv \
  "${OUT_ROOT}/guarded/"*/metrics.json \
  > "${OUT_ROOT}/FROZEN_INPUT_AND_RESULT_SHA256SUMS.txt"

echo "Development policy-backend frontier complete: ${OUT_ROOT}"
