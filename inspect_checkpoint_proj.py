from safetensors.torch import load_file
import os, json, torch

checkpoint = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj_ddp/v0-20260702-070509/checkpoint-500'
index_path = os.path.join(checkpoint, 'model.safetensors.index.json')
with open(index_path) as f:
    weight_map = json.load(f)['weight_map']

print('Total keys:', len(weight_map))
print('\nKeys containing proj:')
for k in sorted(weight_map):
    if 'proj' in k:
        print(' ', k, '->', weight_map[k])

print('\nKeys containing lm_head:')
for k in sorted(weight_map):
    if 'lm_head' in k:
        print(' ', k, '->', weight_map[k])

# Load proj and lm_head weights
state = {}
for shard in sorted(set(weight_map.values())):
    state.update(load_file(os.path.join(checkpoint, shard)))

for name in ['thinker.audio_tower.proj.weight', 'thinker.audio_tower.proj.bias', 'thinker.lm_head.weight']:
    if name in state:
        w = state[name]
        print(f'\n{name}: shape={tuple(w.shape)}, mean={w.mean().item():.6f}, std={w.std().item():.6f}, min={w.min().item():.6f}, max={w.max().item():.6f}')
