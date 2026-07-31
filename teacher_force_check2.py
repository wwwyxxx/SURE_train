import os
import sys
import torch

sys.path.insert(0, '/workspace/ms-swift')
sys.path.insert(0, '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/ms-swift')
sys.path.insert(0, '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train')

plugin_path = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register_randmatch.py'
spec = __import__('importlib.util').util.spec_from_file_location('plugin', plugin_path)
mod = __import__('importlib.util').util.module_from_spec(spec)
spec.loader.exec_module(mod)

from swift.llm import PtEngine, get_template
from swift.llm.template import get_template as get_template_fn

checkpoint = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj_randmatch/v0-20260702-072854/checkpoint-500'
audio_path = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/example/BAC009S0002W0263.wav'
ref = '新科空调'

os.environ['MAX_PIXELS'] = '1003520'
os.environ['ENABLE_AUDIO_OUTPUT'] = '0'

engine = PtEngine(
    '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B',
    model_type='qwen2_5_omni_custom',
    attn_impl='flash_attention_2',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    adapters=[checkpoint],
)
template = get_template('qwen2_5_omni', engine.processor)
engine.default_template = template

# Encode a training example with labels
example = {
    'messages': [
        {'role': 'user', 'content': [{'type': 'audio', 'audio_url': audio_path}, {'type': 'text', 'text': 'Transcribe the speech to text.'}]},
        {'role': 'assistant', 'content': ref}
    ]
}
encoded = template.encode(example)
print('Encoded keys:', list(encoded.keys()))
for k, v in encoded.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}')
    elif isinstance(v, list):
        print(f'  {k}: list len={len(v)}')
    else:
        print(f'  {k}: {type(v)}')

# Move to device and run forward
model = engine.model
model.eval()
inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in encoded.items() if not isinstance(v, (list, str)) or (isinstance(v, torch.Tensor))}
print('\nInputs for model:')
for k, v in inputs.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={tuple(v.shape)}')

with torch.no_grad():
    outputs = model(**inputs, return_dict=True)
    print('\nLogits shape:', outputs.logits.shape)
    preds = outputs.logits.argmax(dim=-1)
    print('Predicted token ids (last 10):', preds[0, -10:].tolist())
    print('Predicted text:', engine.processor.tokenizer.decode(preds[0], skip_special_tokens=True))
