import json, os

v1 = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554/args.json'
control = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_control/v0-20260702-060512/args.json'

a1 = json.load(open(v1))
a2 = json.load(open(control))

all_keys = set(a1.keys()) | set(a2.keys())
for k in sorted(all_keys):
    v1_val = a1.get(k, 'MISSING')
    v2_val = a2.get(k, 'MISSING')
    if v1_val != v2_val:
        print(f'DIFF {k}:')
        print(f'  v1: {v1_val}')
        print(f'  control: {v2_val}')
