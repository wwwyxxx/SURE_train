import subprocess
import os

path = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py'
# Check if it's in git
result = subprocess.run(['git', '-C', '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train', 'log', '--oneline', '-5', '--', path], capture_output=True, text=True)
print('git log for model_register.py:')
print(result.stdout)
print(result.stderr)
