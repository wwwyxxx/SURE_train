from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B', trust_remote_code=True)

# What token id corresponds to string '2'
ids = tok.encode('2', add_special_tokens=False)
print('encode 2:', ids)
print('decode 2 ids:', [tok.decode([i]) for i in ids])

# What is id 2
tok2 = tok.convert_ids_to_tokens([2])[0]
print('token id 2:', repr(tok2))
