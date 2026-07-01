#!/usr/bin/env bash
set -euo pipefail

# Memory stress test: batch_size=8 with 30s audio x 100 entries.
# 8-GPU DDP.

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
  --custom_register_path custom/kimi_audio_swift_register.py \
  --model /workspace/model/Qwen2.5-7B \
  --model_type kimi_audio_text \
  --dataset test_audio_30s_x100 \
  --train_type full \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --lr_scheduler_type constant \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 2048 \
  --logging_steps 1 \
  --dataloader_num_workers 8 \
  --save_steps 10000 \
  --save_total_limit 1 \
  --save_only_model true \
  --output_dir output/test_audio_30s_bs8 \
  --report_to none
