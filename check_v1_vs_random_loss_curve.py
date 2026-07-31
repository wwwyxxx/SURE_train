import json, os

paths = {
    'v1': '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554/logging.jsonl',
    'random': '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj/v0-20260702-053257/logging.jsonl',
}

for name, path in paths.items():
    print(f'\n=== {name} ===')
    with open(path) as f:
        lines = f.readlines()
    print(f'total steps: {len(lines)}')
    for line in lines[:5]:
        print(json.loads(line))
    print('...')
    for line in lines[-3:]:
        print(json.loads(line))
