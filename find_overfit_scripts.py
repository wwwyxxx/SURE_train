import os, json

base = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test'
for fn in sorted(os.listdir(base)):
    if 'run_overfit1' in fn and fn.endswith('.sh'):
        path = os.path.join(base, fn)
        with open(path) as f:
            content = f.read()
        print(f'=== {fn} ===')
        for line in content.split('\n'):
            if 'external_plugins' in line or 'model_type' in line or 'output_dir' in line or 'trainable_parameters' in line or 'random' in line.lower():
                print(line.strip())
        print()
