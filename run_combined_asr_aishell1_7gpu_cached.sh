#!/usr/bin/env bash
set -euo pipefail

# MiMo-Audio 7B + AISHELL-1 134k full training with cached audio tokens
# 7 GPUs, batch size 8 per device, gradient accumulation 4
# Effective batch size = 7 * 8 * 4 = 224

NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --custom_register_path custom/mimo_audio_swift_register.py \
  --model /workspace/model/MiMo-Audio-7B-Base-merged-v3 \
  --model_type mimo_audio \
  --dataset combined_asr_aishell_1_cached \
  --train_type full \
  --split_dataset_ratio 0 \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --trainable_parameters lm_head \
  --ddp_find_unused_parameters true \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 2048 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir output/combined_asr_aishell1_7gpu_cached \
  --report_to none \
  2>&1 | tee output/combined_asr_aishell1_7gpu_cached.log
