import json, os
from safetensors.torch import load_file
import torch

ckpt = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj/v0-20260702-053257/checkpoint-500'
index = json.load(open(os.path.join(ckpt, 'model.safetensors.index.json')))
key = 'thinker.audio_tower.proj.weight'
fn = index['weight_map'][key]
state = load_file(os.path.join(ckpt, fn))
w = state[key].float()
print('shape:', tuple(w.shape))
print('mean:', w.mean().item())
print('std:', w.std().item())
print('min:', w.min().item())
print('max:', w.max().item())
