import json
args = json.load(open('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj_ddp/v0-20260702-070209/args.json'))
print('rank:', args.get('rank'))
print('local_rank:', args.get('local_rank'))
print('world_size:', args.get('world_size'))
