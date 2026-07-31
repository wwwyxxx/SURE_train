#!/usr/bin/env bash
set -euo pipefail

# Max 7 GPUs allowed.
# Registered ms-swift model with optional runtime component registration.
NPROC_PER_NODE={{n_gpu}} \
CUDA_VISIBLE_DEVICES={{gpu_list}} \
{{env_vars}}
swift sft \
  --model_type {{model_type}} \
  --model {{model_path}} \
  --dataset {{dataset_arg}} \
  {{external_plugins}} \
  {{custom_register_path}} \
  --train_type {{train_type}} \
  {{lora_args}} \
  {{freeze_args}} \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size {{per_device_bs}} \
  --gradient_accumulation_steps {{grad_accum}} \
  --num_train_epochs {{epochs}} \
  --learning_rate {{lr}} \
  --lr_scheduler_type {{lr_scheduler}} \
  --warmup_ratio {{warmup_ratio}} \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length {{max_length}} \
  --logging_steps 10 \
  --save_steps {{save_steps}} \
  --save_total_limit 2 \
  --output_dir {{output_dir}} \
  --report_to none
