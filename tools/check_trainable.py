import os, sys
ROOT = '/workspace'
sys.path.insert(0, os.path.join(ROOT, 'MiMo-Audio'))
from src.mimo_audio.modeling_mimo_audio import MiMoAudioForCausalLM, MiMoAudioArguments
from transformers import AutoTokenizer
import torch

checkpoint = '/workspace/output/combined_asr_aishell1_cached100_groupdowncast_lmhead/v1-20260629-043155/checkpoint-80'
print('checkpoint', checkpoint, 'exists', os.path.exists(checkpoint))
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
for tok in ['<|sosp|>','<|eosp|>','<|empty|>','<|sostm|>','<|eostm|>','<|eot|>']:
    if tok not in tokenizer.get_vocab():
        tokenizer.add_tokens([tok], special_tokens=True)
args = MiMoAudioArguments(
    model_name_or_path=checkpoint,
    sosp_idx=tokenizer.convert_tokens_to_ids('<|sosp|>'),
    eosp_idx=tokenizer.convert_tokens_to_ids('<|eosp|>'),
    empty_idx=tokenizer.convert_tokens_to_ids('<|empty|>'),
    sostm_idx=tokenizer.convert_tokens_to_ids('<|sostm|>'),
    eostm_idx=tokenizer.convert_tokens_to_ids('<|eostm|>'),
    eot_idx=tokenizer.convert_tokens_to_ids('<|eot|>'),
)
print('loading model')
model = MiMoAudioForCausalLM.from_pretrained(checkpoint, args=args, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
print('loaded')
model.train()
model.requires_grad_(True)

freeze_parameters = ['model.embed_tokens', 'model.layers', 'model.norm', 'speech_embeddings', 'input_local_transformer', 'speech_group_downcast', 'lm_head']
trainable_parameters = ['lm_head', 'speech_group_downcast']

for n, p in model.named_parameters():
    for fp in freeze_parameters:
        if n.startswith(fp):
            p.requires_grad = False
for n, p in model.named_parameters():
    for tp in trainable_parameters:
        if n.startswith(tp):
            p.requires_grad = True

prefixes = {}
for n, p in model.named_parameters():
    if p.requires_grad:
        prefix = n.split('.')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + p.numel()
print('Trainable params by top-level prefix:')
for k, v in sorted(prefixes.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v/1e6:.1f}M')
print('Total trainable:', sum(prefixes.values())/1e6, 'M')
