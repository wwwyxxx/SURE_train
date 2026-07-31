import os, re

roots = [
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom',
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-191248/custom',
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs',
]

for root in roots:
    if not os.path.exists(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if 'qwen2_5_omni' in fn and 'model_register' in fn and fn.endswith('.py'):
                path = os.path.join(dirpath, fn)
                with open(path) as f:
                    content = f.read()
                has_random = 'random' in content.lower() or 'init.normal' in content or 'init.xavier' in content or 'init.kaiming' in content
                has_proj = 'audio_tower.proj' in content
                print(f'{path}')
                print(f'  has_proj_mention: {has_proj}, has_random_init: {has_random}')
