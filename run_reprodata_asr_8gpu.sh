#!/usr/bin/env bash
set -euo pipefail

# 8-GPU DDP training for Kimi-Audio ASR on ms-swift.
# Note: per_device_train_batch_size=1 per GPU, so the effective global batch size becomes 8.
# If you want to keep the original single-card optimization dynamics, consider
# dividing learning_rate by 8 or using gradient_accumulation_steps=1 (as below).

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
KIMI_AUDIO_TRAIN_WHISPER="${KIMI_AUDIO_TRAIN_WHISPER:-1}" \
swift sft \
  --custom_register_path custom/kimi_audio_swift_register.py \
  --model /workspace/model/Qwen2.5-7B \
  --model_type kimi_audio_text \
  --dataset reprodata_asr_zh_existing \
  --train_type full \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --learning_rate 1e-5 \
  --lr_scheduler_type constant \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 512 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 1 \
  --save_only_model true \
  --output_dir output/reprodata_asr_zh_existing_train_whisper_8gpu \
  --report_to none
