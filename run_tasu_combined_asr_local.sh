#!/usr/bin/env bash
set -euo pipefail

# TASU projector-only SFT on combined_asr_local via ms-swift.
# LLM (Qwen2.5-1.5B) and audio encoder (SenseVoiceSmall) are frozen;
# only encoder_projector is trained.

NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=0,1,2 \
swift sft \
  --custom_register_path custom/tasu_swift_register.py \
  --model model/Qwen2.5-1.5B \
  --model_type tasu \
  --dataset combined_asr_local \
  --train_type full \
  --freeze_parameters_regex '.*' \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --trainable_parameters encoder_projector \
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
  --max_length 4096 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --save_safetensors false \
  --resume_from_checkpoint outputs/20260708-203711/tasu_combined_asr_local/v1-20260709-144001/checkpoint-20000 \
  --output_dir outputs/20260708-203711/tasu_combined_asr_local \
  --report_to none
