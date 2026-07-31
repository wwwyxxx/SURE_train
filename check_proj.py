import torch
from safetensors.torch import load_file
import json, os

print("start", flush=True)
index = json.load(open("/workspace/SURE_train/model/Qwen2.5-Omni-7B/model.safetensors.index.json"))
proj_key = "thinker.audio_tower.proj.weight"
file_name = index["weight_map"].get(proj_key)
print(f"{proj_key} in {file_name}", flush=True)

full_path = os.path.join("/workspace/SURE_train/model/Qwen2.5-Omni-7B", file_name)
state = load_file(full_path)
w = state[proj_key]
print("mean:", w.float().mean().item(), flush=True)
print("std:", w.float().std().item(), flush=True)
print("shape:", tuple(w.shape), flush=True)
print("min:", w.float().min().item(), flush=True)
print("max:", w.float().max().item(), flush=True)

# compare with random init of same shape
r = torch.randn(w.shape, dtype=w.dtype)
print("random mean/std:", r.float().mean().item(), r.float().std().item(), flush=True)
