from safetensors.torch import load_file
import os, json, torch

checkpoint = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj_ddp/v0-20260702-070509/checkpoint-500'
index_path = os.path.join(checkpoint, 'model.safetensors.index.json')
with open(index_path) as f:
    weight_map = json.load(f)['weight_map']

state = {}
for shard in sorted(set(weight_map.values())):
    state.update(load_file(os.path.join(checkpoint, shard)))

for name in ['thinker.audio_tower.proj.weight', 'thinker.audio_tower.proj.bias', 'thinker.lm_head.weight']:
    if name in state:
        w = state[name]
        print(f'{name}: shape={tuple(w.shape)}, mean={w.mean().item():.6f}, std={w.std().item():.6f}, min={w.min().item():.6f}, max={w.max().item():.6f}')

# Also check official Omni proj for comparison
official = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B'
with open(os.path.join(official, 'model.safetensors.index.json')) as f:
    weight_map2 = json.load(f)['weight_map']
state2 = {}
for shard in sorted(set(weight_map2.values())):
    state2.update(load_file(os.path.join(official, shard)))

if 'thinker.audio_tower.proj.weight' in state2:
    w = state2['thinker.audio_tower.proj.weight']
    print(f'\nOFFICIAL thinker.audio_tower.proj.weight: shape={tuple(w.shape)}, mean={w.mean().item():.6f}, std={w.std().item():.6f}, min={w.min().item():.6f}, max={w.max().item():.6f}')
