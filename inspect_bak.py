from pathlib import Path
import datetime

p = Path('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py.bak')
st = p.stat()
print('bak mtime:', st.st_mtime)
print('bak datetime:', datetime.datetime.fromtimestamp(st.st_mtime))

p2 = Path('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py')
st2 = p2.stat()
print('current mtime:', st2.st_mtime)
print('current datetime:', datetime.datetime.fromtimestamp(st2.st_mtime))

# v1 args timestamp
import json
args = json.load(open('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554/args.json'))
print('v1 args output_dir:', args.get('output_dir'))
