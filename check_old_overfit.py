import json, os
from collections import Counter

# find latest infer result in old overfit1_custom
base = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom'
for v in sorted(os.listdir(base)):
    infer_dir = os.path.join(base, v, 'checkpoint-500', 'infer_result')
    if os.path.exists(infer_dir):
        files = sorted(os.listdir(infer_dir))
        if files:
            path = os.path.join(infer_dir, files[-1])
            responses = []
            labels = []
            with open(path) as f:
                for line in f:
                    row = json.loads(line)
                    responses.append(row.get('response', ''))
                    labels.append(row.get('labels', ''))
            print(f'{v}:')
            print('  responses:', Counter(responses))
            print('  labels:', Counter(labels))
            print('  first pair:', responses[0], '|', labels[0])
