from pathlib import Path

bak = Path('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py.bak').read_text()
cur = Path('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py').read_text()

bak_lines = bak.split('\n')
cur_lines = cur.split('\n')

for i, (b, c) in enumerate(zip(bak_lines, cur_lines)):
    if b != c:
        print(f'Line {i+1}:')
        print(f'  BAK: {b}')
        print(f'  CUR: {c}')
