import json, os
from safetensors.torch import load_file
import torch

# v1 checkpoint
ckpt = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554/checkpoint-500'
index = json.load(open(os.path.join(ckpt, 'model.safetensors.index.json')))

# official Omni
omni = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B'
omni_index = json.load(open(os.path.join(omni, 'model.safetensors.index.json')))

def load_state(base, idx, key):
    fn = idx['weight_map'][key]
    return load_file(os.path.join(base, fn))[key]

key = 'thinker.audio_tower.proj.weight'
c = load_state(ckpt, index, key).float()
o = load_state(omni, omni_index, key).float()
print('v1 checkpoint proj:')
print(f'  mean={c.mean():.6f} std={c.std():.6f}')
print(f'  official mean={o.mean():.6f} std={o.std():.6f}')
print(f'  diff norm={(c-o).norm().item():.6f} relative={((c-o).norm()/o.norm()).item():.6f}')

# Check if it looks like random init std=0.02
print(f'  is close to random(0,0.02)? std ratio to 0.02: {c.std().item()/0.02:.3f}')
