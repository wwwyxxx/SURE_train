import sys
sys.path.insert(0, '/workspace/ms-swift')
from swift.llm.argument.train_args import TrainArguments
from swift.trainers import TrainerFactory
from swift.utils import parse_args

args_list = [
    '--model_type', 'qwen2_5_omni_custom',
    '--model', '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B',
    '--dataset', 'combined_asr_local_smoke_overfit1',
    '--external_plugins', '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py',
    '--custom_register_path', '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_smoke_dataset_register.py',
    '--train_type', 'full',
    '--freeze_llm', 'true',
    '--freeze_vit', 'true',
    '--freeze_aligner', 'true',
    '--freeze_parameters_regex', '.*',
    '--trainable_parameters', 'thinker.lm_head', 'thinker.audio_tower.proj',
    '--split_dataset_ratio', '0',
    '--per_device_train_batch_size', '2',
    '--gradient_accumulation_steps', '1',
    '--num_train_epochs', '10',
    '--learning_rate', '1e-4',
    '--lr_scheduler_type', 'constant',
    '--warmup_ratio', '0',
    '--max_grad_norm', '1.0',
    '--bf16', 'true',
    '--attn_impl', 'flash_attn',
    '--gradient_checkpointing', 'true',
    '--max_length', '1024',
    '--logging_steps', '1',
    '--save_steps', '500',
    '--save_total_limit', '1',
    '--output_dir', '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj',
    '--report_to', 'none',
]
args, _ = parse_args(TrainArguments, args_list)
print('args.bf16:', args.bf16)
print('args.fp16:', args.fp16)
try:
    ta = TrainerFactory.get_training_args(args)
    print('training_args.bf16:', ta.bf16)
    print('training_args.fp16:', ta.fp16)
    print('OK')
except Exception as e:
    print('ERROR:', e)
