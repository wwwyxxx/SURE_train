#!/usr/bin/env bash
set -euo pipefail

# Max 7 GPUs allowed.
NPROC_PER_NODE={{n_gpu}} \
CUDA_VISIBLE_DEVICES={{gpu_list}} \
swift sft \
  --custom_register_path {{register_path}} \
  --model {{model_path}} \
  --model_type {{model_type}} \
  --dataset {{dataset_name}} \
  --train_type {{train_type}} \
  --freeze_llm {{freeze_llm}} \
  --freeze_vit {{freeze_vit}} \
  --freeze_aligner {{freeze_aligner}} \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size {{per_device_bs}} \
  --gradient_accumulation_steps {{grad_accum}} \
  --num_train_epochs {{epochs}} \
  --learning_rate {{lr}} \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length {{max_length}} \
  --logging_steps 10 \
  --save_steps {{save_steps}} \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir {{output_dir}} \
  --report_to none
