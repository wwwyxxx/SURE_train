#!/usr/bin/env bash
set -euo pipefail

KIMI_AUDIO_TRAIN_WHISPER="${KIMI_AUDIO_TRAIN_WHISPER:-1}" swift sft \
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
  --output_dir output/reprodata_asr_zh_existing_train_whisper \
  --report_to none
