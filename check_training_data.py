import json
from collections import Counter

path = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1.jsonl'
txts = []
with open(path) as f:
    for line in f:
        row = json.loads(line)
        txts.append(row['txt'])
print('Total:', len(txts))
print('Unique texts:', Counter(txts))
