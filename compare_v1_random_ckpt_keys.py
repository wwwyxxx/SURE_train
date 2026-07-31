import json, os
from safetensors.torch import load_file

v1_index = json.load(open('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554/checkpoint-500/model.safetensors.index.json'))
rand_index = json.load(open('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj/v0-20260702-053257/checkpoint-500/model.safetensors.index.json'))

print('Keys only in v1:', len(set(v1_index['weight_map'].keys()) - set(rand_index['weight_map'].keys())))
print('Keys only in random:', len(set(rand_index['weight_map'].keys()) - set(v1_index['weight_map'].keys())))

# Load and compare lm_head
key = 'thinker.lm_head.weight'
v1_fn = v1_index['weight_map'][key]
rand_fn = rand_index['weight_map'][key]
v1_w = load_file(os.path.join('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554/checkpoint-500', v1_fn))[key].float()
rand_w = load_file(os.path.join('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj/v0-20260702-053257/checkpoint-500', rand_fn))[key].float()

print(f'\nlm_head v1: mean={v1_w.mean():.6f} std={v1_w.std():.6f}')
print(f'lm_head random: mean={rand_w.mean():.6f} std={rand_w.std():.6f}')
print(f'lm_head diff norm={(v1_w-rand_w).norm().item():.6f}')

# find top tokens by norm change
norm_change = (v1_w - rand_w).norm(dim=1)
top_changed = norm_change.topk(10)
print('Top 10 token ids with largest lm_head change:', top_changed.indices.tolist())
print('Their change norms:', top_changed.values.tolist())
