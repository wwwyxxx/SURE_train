import json
args = json.load(open('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj/v0-20260702-053257/args.json'))
for k in ['max_length', 'truncation_strategy', 'padding_side', 'max_new_tokens', 'label_smoothing_factor', 'model_type', 'template_type', 'trainable_parameters', 'num_train_epochs', 'per_device_train_batch_size']:
    print(k, ':', args.get(k))
