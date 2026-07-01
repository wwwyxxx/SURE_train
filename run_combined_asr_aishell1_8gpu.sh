#!/usr/bin/env bash
set -euo pipefail

# 8-GPU DDP training on full AISHELL-1 ASR data.
# Trains whisper encoder + adaptor + mimo_layers + text head; shared LLM frozen.

NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
KIMI_AUDIO_TRAIN_WHISPER=1 \
swift sft \
  --custom_register_path custom/kimi_audio_swift_register.py \
  --model /workspace/model/Qwen2.5-7B \
  --model_type kimi_audio_text \
  --dataset combined_asr_aishell_1 \
  --train_type full \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir output/combined_asr_aishell1_8gpu \
  --report_to none
