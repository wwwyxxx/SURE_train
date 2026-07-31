#!/usr/bin/env bash
set -euo pipefail

# 4-GPU DDP training for MiMo-Audio ASR on pre-cached audio tokens.
# Trains speech_embeddings, input_local_transformer, speech_group_downcast, lm_head.

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
swift sft \
  --custom_register_path custom/mimo_audio_swift_register.py \
  --model_type mimo_audio \
  --dataset combined_asr_local_cached \
  --train_type full \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 2048 \
  --dataloader_num_workers 4 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir output/mimo_audio_combined_asr_local_cached \
  --report_to none
