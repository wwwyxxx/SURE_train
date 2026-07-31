from pathlib import Path

log = Path('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/infer_overfit1_custom_random_proj.log').read_text()

# Check if external_plugins was loaded
if 'qwen2_5_omni_model_register.py' in log:
    print('model register loaded: YES')
else:
    print('model register loaded: NO')

# Check model type
idx = log.find('model_type')
print('model_type snippet:', log[idx:idx+100] if idx != -1 else 'not found')

# Check if random init log message appears
if 'Randomly re-initializing' in log:
    print('random init executed during inference: YES')
else:
    print('random init executed during inference: NO')
