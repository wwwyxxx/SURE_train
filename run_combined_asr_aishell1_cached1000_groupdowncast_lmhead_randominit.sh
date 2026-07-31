#!/usr/bin/env bash
set -euo pipefail

# Smoke test: 1000 samples from combined_asr_aishell_1_cached
# Random init lm_head and speech_group_downcast, freeze everything else.
# 1 GPU (GPU 4), constant LR, no warmup, 50 epochs, lr=2e-4

MIMO_AUDIO_FREEZE_LLM=0 \
MIMO_AUDIO_RANDOM_INIT_LM_HEAD=1 \
MIMO_AUDIO_RANDOM_INIT_SPEECH_GROUP_DOWNCAST=1 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --custom_register_path custom/mimo_audio_swift_register.py \
  --model /workspace/model/MiMo-Audio-7B-Base-merged-v3 \
  --model_type mimo_audio \
  --dataset combined_asr_aishell_1_cached_1000 \
  --train_type full \
  --split_dataset_ratio 0 \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_parameters local_transformer hidden_states_downcast local_transformer_lm_heads \
  --trainable_parameters lm_head speech_group_downcast \
  --ddp_find_unused_parameters true \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 4 \
  --num_train_epochs 50 \
  --learning_rate 2e-4 \
  --lr_scheduler_type constant \
  --warmup_ratio 0 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 2048 \
  --logging_steps 5 \
  --save_steps 500 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir output/combined_asr_aishell1_cached1000_groupdowncast_lmhead_randominit \
  --report_to none \
  2>&1 | tee output/combined_asr_aishell1_cached1000_groupdowncast_lmhead_randominit.log
