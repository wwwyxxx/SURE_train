import os, subprocess

# v1 was created on 2026-07-01 19:15:54
# Look for any backup or git version of the register script around that time
paths_to_check = [
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py.bak',
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register_v1.py',
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register_v2.py',
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register_v3.py',
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register_v4.py',
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register_v5.py',
]

for p in paths_to_check:
    if os.path.exists(p):
        print(f'EXISTS: {p}')
        with open(p) as f:
            content = f.read()
        if 'random' in content.lower() or 'init.normal' in content or 'init.xavier' in content:
            print('  -> has random init code')
        if 'audio_tower.proj' in content:
            print('  -> mentions audio_tower.proj')
    else:
        print(f'NOT FOUND: {p}')

# Check file timestamps in custom dir
custom_dir = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom'
print('\nFiles in custom dir:')
for fn in sorted(os.listdir(custom_dir)):
    p = os.path.join(custom_dir, fn)
    st = os.stat(p)
    print(f'  {fn}: mtime={st.st_mtime} size={st.st_size}')
