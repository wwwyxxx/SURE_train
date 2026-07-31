#!/usr/bin/env bash
set -euo pipefail

# SLAM-LLM projector-only SFT on combined_asr_local via ms-swift.
# LLM (vicuna-7b-v1.5) and audio encoder (wavlm-large) are frozen;
# only encoder_projector is trained.

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=3,4,5,6 \
swift sft \
  --custom_register_path custom/slam_llm_swift_register.py \
  --model model/vicuna-7b-v1.5 \
  --model_type slam_llm_asr \
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
  --dataloader_num_workers 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --max_length 4096 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 2 \
  --save_safetensors false \
  --resume_from_checkpoint outputs/20260709-144622/slam_llm_combined_asr_local/v2-20260709-211646/checkpoint-2000 \
  --output_dir outputs/20260709-144622/slam_llm_combined_asr_local \
  --report_to none
