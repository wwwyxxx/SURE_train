import json, os
from safetensors.torch import load_file
import torch

# checkpoint
ckpt = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj/v0-20260702-053257/checkpoint-500'
index = json.load(open(os.path.join(ckpt, 'model.safetensors.index.json')))

# official Omni
omni = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B'
omni_index = json.load(open(os.path.join(omni, 'model.safetensors.index.json')))

def load_state(base, idx, key):
    fn = idx['weight_map'][key]
    path = os.path.join(base, fn)
    return load_file(path)[key]

keys = [
    'thinker.audio_tower.proj.weight',
    'thinker.audio_tower.proj.bias',
    'thinker.lm_head.weight',
]

for key in keys:
    print(f'\n=== {key} ===')
    c = load_state(ckpt, index, key).float()
    o = load_state(omni, omni_index, key).float()
    print(f'checkpoint: mean={c.mean():.6f} std={c.std():.6f} shape={tuple(c.shape)}')
    print(f'official:   mean={o.mean():.6f} std={o.std():.6f} shape={tuple(o.shape)}')
    print(f'diff norm:  {(c-o).norm().item():.6f}')
    print(f'relative diff:  {((c-o).norm()/o.norm()).item():.6f}')
