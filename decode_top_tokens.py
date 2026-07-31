from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B', trust_remote_code=True)
ids = [198, 102317, 16628, 151645, 69526]
for i in ids:
    print(f'{i}: {tok.convert_ids_to_tokens([i])} decode={tok.decode([i])!r}')
