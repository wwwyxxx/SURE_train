import os, json
from collections import Counter

base = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom'
for v in sorted(os.listdir(base)):
    ckpt_dir = os.path.join(base, v)
    if not os.path.isdir(ckpt_dir):
        continue
    infer_dir = os.path.join(ckpt_dir, 'checkpoint-500', 'infer_result')
    if not os.path.exists(infer_dir):
        continue
    files = sorted(os.listdir(infer_dir))
    if not files:
        continue
    path = os.path.join(infer_dir, files[-1])
    responses = []
    labels = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            responses.append(row.get('response', ''))
            labels.append(row.get('labels', ''))
    print(f'=== {v} ===')
    print('  responses:', Counter(responses))
    print('  labels:', Counter(labels))
    # find args
    args_path = os.path.join(ckpt_dir, 'args.json')
    if os.path.exists(args_path):
        args = json.load(open(args_path))
        print('  trainable_parameters:', args.get('trainable_parameters'))
        print('  output_dir:', args.get('output_dir'))
