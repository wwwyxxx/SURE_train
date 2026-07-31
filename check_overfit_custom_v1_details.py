import json, os

base = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554'
args = json.load(open(os.path.join(base, 'args.json')))
keys = ['model_type', 'train_type', 'trainable_parameters', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'num_train_epochs', 'learning_rate', 'lr_scheduler_type', 'max_length', 'bf16', 'attn_impl', 'gradient_checkpointing', 'freeze_llm', 'freeze_vit', 'freeze_aligner', 'freeze_parameters_regex', 'external_plugins', 'custom_register_path']
for k in keys:
    print(f'{k}: {args.get(k)}')

# check logging
print('\nFirst 5 training logs:')
with open(os.path.join(base, 'logging.jsonl')) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(json.loads(line))
