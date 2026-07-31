from safetensors.torch import load_file
import os, json, torch

def stats(path):
    with open(os.path.join(path, 'model.safetensors.index.json')) as f:
        wm = json.load(f)['weight_map']
    state = {}
    for shard in sorted(set(wm.values())):
        state.update(load_file(os.path.join(path, shard)))
    w = state['thinker.audio_tower.proj.weight']
    b = state['thinker.audio_tower.proj.bias']
    return w.mean().item(), w.std().item(), b.mean().item(), b.std().item()

v6 = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/mini100_v6/v0-20260701-185720/checkpoint-1000'
rand = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/mini100_v6_random_proj/v0-20260702-055056/checkpoint-1000'
official = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B'

for name, path in [('mini100_v6', v6), ('mini100_v6_random_proj', rand), ('official', official)]:
    try:
        wm, ws, bm, bs = stats(path)
        print(f'{name}: proj.weight mean={wm:.6f} std={ws:.6f}, proj.bias mean={bm:.6f} std={bs:.6f}')
    except Exception as e:
        print(f'{name}: ERROR {e}')
