#!/usr/bin/env bash
set -euo pipefail

# Smoke test: 100 samples from combined_asr_aishell_1_cached
# Freeze LLM / speech_embeddings / input_local_transformer,
# train only speech_group_downcast + lm_head.
# 1 GPU, constant LR, no warmup, 20 epochs

# Disable custom _freeze_for_asr; rely on ms-swift's native freeze flags.
MIMO_AUDIO_FREEZE_LLM=0 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7} \
swift sft \
  --custom_register_path custom/mimo_audio_swift_register.py \
  --model /workspace/model/MiMo-Audio-7B-Base-merged-v3 \
  --model_type mimo_audio \
  --dataset combined_asr_aishell_1_cached_100 \
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
  --output_dir output/combined_asr_aishell1_cached100_groupdowncast_lmhead \
  --report_to none \
  2>&1 | tee output/combined_asr_aishell1_cached100_groupdowncast_lmhead.log
