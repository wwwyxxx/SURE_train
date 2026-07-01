#!/usr/bin/env bash
set -euo pipefail

# LoRA smoke test on AISHELL-93 Chinese subset
# 1 GPU, constant LR, no warmup, 20 epochs
# In a Docker container that exposes only one GPU, override with CUDA_VISIBLE_DEVICES=0.

MIMO_AUDIO_FREEZE_LLM=0 \
MIMO_AUDIO_LLM_BACKBONE=/workspace/model/MiMo-7B-Base \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7} \
swift sft \
  --custom_register_path custom/mimo_audio_swift_register.py \
  --model /workspace/model/MiMo-Audio-7B-Base-merged-v3 \
  --model_type mimo_audio \
  --dataset reprodata_asr_zh_existing_aishell93 \
  --train_type lora \
  --target_modules all-linear \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --modules_to_save speech_embeddings.0 speech_embeddings.1 speech_embeddings.2 speech_embeddings.3 speech_embeddings.4 speech_embeddings.5 speech_embeddings.6 speech_embeddings.7 speech_group_downcast lm_head \
  --freeze_vit true \
  --split_dataset_ratio 0 \
  --ddp_find_unused_parameters true \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 4 \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --lr_scheduler_type constant \
  --warmup_ratio 0 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 2048 \
  --logging_steps 5 \
  --save_steps 100 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir output/reprodata_asr_zh_existing_aishell93_lora \
  --report_to none \
  2>&1 | tee output/reprodata_asr_zh_existing_aishell93_lora.log
