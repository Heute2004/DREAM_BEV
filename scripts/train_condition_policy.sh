#!/usr/bin/env bash
set -euo pipefail

# Train the RGB-only condition policy with the pinned upstream InternVL source.
# The annotation package is constructed locally; the sealed test annotation is
# intentionally not passed to this command.

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INTERNVL_ROOT="${INTERNVL_ROOT:-$PROJECT_ROOT/third_party/InternVL/internvl_chat}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the pristine InternVL3-1B model}"
META_PATH="${META_PATH:?set META_PATH to the generated Train metadata JSON}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to an empty output directory}"
GPU_ID="${GPU_ID:-0}"
MASTER_PORT="${MASTER_PORT:-29682}"

test -d "$INTERNVL_ROOT"
test -f "$INTERNVL_ROOT/internvl/train/internvl_chat_finetune.py"
test -f "$MODEL_PATH/config.json"
test -f "$META_PATH"
test ! -e "$OUTPUT_DIR/trainer_state.json"
test ! -e "$OUTPUT_DIR/model.safetensors"
mkdir -p "$OUTPUT_DIR"

cd "$INTERNVL_ROOT"
export PYTHONPATH="$INTERNVL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

exec "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "$MODEL_PATH" \
  --conv_style internvl2_5 \
  --use_fast_tokenizer False \
  --output_dir "$OUTPUT_DIR" \
  --meta_path "$META_PATH" \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --tf32 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_strategy steps \
  --logging_steps 5 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --report_to tensorboard \
  --seed 3407
