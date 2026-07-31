import json
from collections import Counter

path = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj/v0-20260702-053257/checkpoint-500/infer_result/20260702-053721.jsonl'
responses = []
labels = []
with open(path) as f:
    for line in f:
        row = json.loads(line)
        responses.append(row['response'])
        labels.append(row['labels'])

print('Unique responses:', Counter(responses))
print('Unique labels:', Counter(labels))
print('First 5 response-label pairs:')
for r, l in zip(responses[:5], labels[:5]):
    print(f'  pred: {r!r} | label: {l!r}')
