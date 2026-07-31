import torch
from safetensors.torch import load_file
import json, os

index = json.load(open("/workspace/SURE_train/model/Qwen2.5-Omni-7B/model.safetensors.index.json"))
proj_key = "thinker.audio_tower.proj.weight"
file_name = index["weight_map"][proj_key]
full_path = os.path.join("/workspace/SURE_train/model/Qwen2.5-Omni-7B", file_name)
state = load_file(full_path)
w = state[proj_key].float()

# SVD to check if structure looks trained vs random
u, s, v = torch.svd(w)
print("Top 10 singular values:", s[:10].tolist())
print("Ratio s[0]/s[-1]:", (s[0]/s[-1]).item())

# Compare with random
r = torch.randn_like(w)
_, sr, _ = torch.svd(r)
print("Random top 10 singular values:", sr[:10].tolist())
print("Random ratio:", (sr[0]/sr[-1]).item())
