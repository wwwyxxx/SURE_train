import json, os

v1_args = json.load(open('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554/args.json'))
random_args = json.load(open('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj/v0-20260702-053257/args.json'))

keys_to_compare = [
    'model_type', 'template_type', 'model_id_or_path', 'model_revision', 'torch_dtype',
    'max_length', 'truncation_strategy', 'padding_side', 'max_new_tokens',
    'train_type', 'trainable_parameters', 'freeze_parameters_regex',
    'freeze_llm', 'freeze_vit', 'freeze_aligner',
    'per_device_train_batch_size', 'gradient_accumulation_steps', 'num_train_epochs',
    'learning_rate', 'lr_scheduler_type', 'warmup_ratio', 'max_grad_norm',
    'bf16', 'fp16', 'attn_impl', 'gradient_checkpointing',
    'logging_steps', 'save_steps', 'save_total_limit', 'save_only_model',
    'split_dataset_ratio', 'seed', 'data_seed',
    'custom_register_path', 'external_plugins',
    'device_map', 'local_rank', 'rank', 'world_size', 'n_gpu',
]

print(f'{"key":<40} {"v1":<60} {"random":<60}')
for k in keys_to_compare:
    v1 = v1_args.get(k, 'MISSING')
    r = random_args.get(k, 'MISSING')
    marker = '' if str(v1) == str(r) else ' ***'
    print(f'{k:<40} {str(v1)[:58]:<60} {str(r)[:58]:<60}{marker}')
