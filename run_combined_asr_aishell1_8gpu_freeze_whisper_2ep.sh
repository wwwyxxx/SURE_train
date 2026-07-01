#!/usr/bin/env bash
set -euo pipefail

# 8-GPU DDP training: freeze whisper, train adaptor + mimo_layers + text head.
# One epoch baseline to verify audio-to-text mapping works on AISHELL-1.

NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --custom_register_path custom/kimi_audio_swift_register.py \
  --model /workspace/SURE_train/model/Qwen2.5-7B \
  --model_type kimi_audio_text \
  --dataset combined_asr_local \
  --train_type full \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --dataloader_num_workers 8 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir output/run1 \
  --report_to none
