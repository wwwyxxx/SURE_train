from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B', trust_remote_code=True)
text = '新科空调'
ids = tok.encode(text, add_special_tokens=False)
print('text:', text)
print('ids:', ids)
print('decoded:', tok.decode(ids, skip_special_tokens=True))
print('id 2:', tok.decode([2]) if 2 in ids else 'not in ids')
print('vocab 2:', tok.convert_ids_to_tokens([2]))
