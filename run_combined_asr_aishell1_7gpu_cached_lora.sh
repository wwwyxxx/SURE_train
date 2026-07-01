#!/usr/bin/env bash
set -euo pipefail

# MiMo-Audio 7B + AISHELL-1 134k training with cached audio tokens
# LoRA on LLM + full training on audio aligner + lm_head
# 7 GPUs, batch size 8 per device, gradient accumulation 4
# Effective batch size = 7 * 8 * 4 = 224

# Disable the custom full-parameter freezing in mimo_audio_swift_register.py;
# ms-swift's LoRA tuner will freeze base weights and train LoRA adapters + modules_to_save.
MIMO_AUDIO_FREEZE_LLM=0 \
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --custom_register_path custom/mimo_audio_swift_register.py \
  --model /workspace/model/MiMo-Audio-7B-Base-merged-v3 \
  --model_type mimo_audio \
  --dataset combined_asr_aishell_1_cached \
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
  --output_dir output/combined_asr_aishell1_7gpu_cached_lora \
  --report_to none \
  2>&1 | tee output/combined_asr_aishell1_7gpu_cached_lora.log
