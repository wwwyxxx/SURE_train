import torch
from safetensors.torch import load_file
import json, os

index = json.load(open("/workspace/SURE_train/model/Qwen2.5-Omni-7B/model.safetensors.index.json"))
proj_key = "thinker.audio_tower.proj.weight"
file_name = index["weight_map"][proj_key]
full_path = os.path.join("/workspace/SURE_train/model/Qwen2.5-Omni-7B", file_name)
state = load_file(full_path)
w = state[proj_key].float()
print("shape:", tuple(w.shape))
print("mean:", w.mean().item())
print("std:", w.std().item())
print("min:", w.min().item())
print("max:", w.max().item())
print("abs mean:", w.abs().mean().item())

# Check distribution vs typical random init scaled by fan_in
fan_in = w.shape[1]
print("Kaiming uniform bound sqrt(6/fan_in):", (6/fan_in)**0.5)
print("Xavier std sqrt(2/(fan_in+fan_out)):", (2/(w.shape[0]+w.shape[1]))**0.5)
print("He std sqrt(2/fan_in):", (2/fan_in)**0.5)
