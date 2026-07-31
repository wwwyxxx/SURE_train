from safetensors.torch import load_file
import os, json, torch

official = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B'
with open(os.path.join(official, 'model.safetensors.index.json')) as f:
    weight_map = json.load(f)['weight_map']
state = {}
for shard in sorted(set(weight_map.values())):
    state.update(load_file(os.path.join(official, shard)))

for name in ['thinker.audio_tower.proj.weight', 'thinker.audio_tower.proj.bias']:
    if name in state:
        w = state[name]
        print(f'{name}: shape={tuple(w.shape)}, mean={w.mean().item():.6f}, std={w.std().item():.6f}, min={w.min().item():.6f}, max={w.max().item():.6f}')
