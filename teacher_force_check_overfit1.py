import os
import sys
import json
import torch

sys.path.insert(0, '/workspace/ms-swift')
sys.path.insert(0, '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/ms-swift')
sys.path.insert(0, '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train')

# Load plugin
plugin_path = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py'
spec = __import__('importlib.util').util.spec_from_file_location('plugin', plugin_path)
mod = __import__('importlib.util').util.module_from_spec(spec)
spec.loader.exec_module(mod)

from swift.llm import PtEngine, InferRequest, get_template
from swift.llm.dataset.utils import load_audio

checkpoint = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_random_proj_ddp/v0-20260702-070509/checkpoint-500'
audio_path = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/example/BAC009S0002W0263.wav'
prompt = 'Transcribe the speech to text.'
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

# Get token ids for reference
ref_ids = engine.processor.tokenizer.encode(ref, add_special_tokens=False)
print('Reference token ids:', ref_ids)
print('Reference text:', engine.processor.tokenizer.decode(ref_ids, skip_special_tokens=True))

# Use swift infer to get response
request = InferRequest(messages=[{
    'role': 'user',
    'content': [{'type': 'audio', 'audio_url': audio_path}, {'type': 'text', 'text': prompt}]
}])
resp_list = engine.infer([request])
response = resp_list[0].choices[0].message.content
print('Autoregressive response:', repr(response))

# Now do teacher forcing: manually run forward with reference labels
# We need to access the model directly
model = engine.model
model.eval()

# Build inputs using template encode
from swift.llm import encode
example = {'messages': [{'role': 'user', 'content': [{'type': 'audio', 'audio_url': audio_path}, {'type': 'text', 'text': prompt}]}, {'role': 'assistant', 'content': ref}]}
encoded = template.encode(example)
print('Encoded keys:', encoded.keys())
print('Input ids shape:', encoded['input_ids'].shape if hasattr(encoded['input_ids'], 'shape') else type(encoded['input_ids']))
