#!/usr/bin/env bash
set -euo pipefail

# Runs the manuscript's public radar controls under the private DREAM-BEV
# Train/Development contract.
ROOT="${ROOT:-/data_hard_driver/hyc/Radar+RIS/workspaces/DREAM_BEV_PUBLIC_BACKBONE_TRANSFER}"
PROJECT="${PROJECT:-${ROOT}/project}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs}"
CONTRACT="${CONTRACT:-${PROJECT}/protocols/carrada_public_backbone_private_transfer.json}"
DATA_ROOT="${DATA_ROOT:-/data_hard_driver/hyc/Radar+RIS/Radar_RIS_VLM_20260724/Data_train}"
MANIFEST="${MANIFEST:-${PROJECT}/contracts/frame_manifest.csv}"
PYTHON_BIN="${PYTHON_BIN:-/home/hyc/.conda/envs/rvmg/bin/python}"
MODEL="${MODEL:?set MODEL to dream_bev, t_rodnet_compact, rodnet_cdc, rodnet_hg, fcn8s, or e_rodnet}"
GPU="${GPU:?set GPU to one physical GPU index}"

case "${MODEL}" in
  dream_bev|t_rodnet_compact|rodnet_cdc|rodnet_hg|fcn8s|e_rodnet) ;;
  *) echo "unsupported MODEL=${MODEL}" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOG_PATH="${LOG_DIR}/${MODEL}.log"
mkdir -p "${LOG_DIR}"
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
cp "${CONTRACT}" "${OUTPUT_DIR}/transfer_contract.json"
sha256sum "${CONTRACT}" > "${OUTPUT_DIR}/transfer_contract.sha256"
export PYTHONPATH="${PROJECT}/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/hyc/.conda/envs/rvmg/lib:${LD_LIBRARY_PATH:-}"

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --manifest "${MANIFEST}"
  --output-dir "${OUTPUT_DIR}"
  --train-epochs 16
  --batch-size 1
  --gradient-accumulation 8
  --num-workers 4
  --learning-rate 0.0003
  --weight-decay 0.0001
  --seed 3407
  --coordinate-weight 0.05
  --early-stop-patience 4
  --progress-every-batches 500
  --device cuda
)

if [[ "${MODEL}" == "dream_bev" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u "${PROJECT}/scripts/train_radar.py" \
    --model dream_bev "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
else
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u "${PROJECT}/scripts/train_public_full_adapted.py" \
    --model "${MODEL}" \
    --source-root "${PROJECT}/third_party" \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
fi
