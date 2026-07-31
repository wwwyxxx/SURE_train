import argparse
import os
import sys
import torch

def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--audio', default='example/BAC009S0002W0263.wav')
    parser.add_argument('--prompt', default='Transcribe the speech to text.')
    parser.add_argument('--attn_impl', default='flash_attention_2')
    args = parser.parse_args()

    sys.path.insert(0, '/workspace/ms-swift')
    sys.path.insert(0, '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/ms-swift')
    sys.path.insert(0, '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train')

    plugin_path = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/custom/qwen2_5_omni_model_register.py'
    spec = __import__('importlib.util').util.spec_from_file_location('plugin', plugin_path)
    mod = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from swift.llm import PtEngine, InferRequest, get_template

    checkpoint = args.checkpoint
    audio = args.audio
    os.environ['MAX_PIXELS'] = '1003520'
    os.environ['ENABLE_AUDIO_OUTPUT'] = '0'

    print(f'Loading with attn_impl={args.attn_impl}...')
    engine = PtEngine(
        '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B',
        model_type='qwen2_5_omni_custom',
        attn_impl=args.attn_impl,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        adapters=[checkpoint],
    )
    template = get_template('qwen2_5_omni', engine.processor)
    engine.default_template = template

    request = InferRequest(messages=[{
        'role': 'user',
        'content': [{'type': 'audio', 'audio_url': audio}, {'type': 'text', 'text': args.prompt}]
    }])
    resp_list = engine.infer([request])
    response = resp_list[0].choices[0].message.content
    print('Response:', response)

    ref = '新科空调'
    cer = levenshtein(response, ref) / max(len(ref), 1)
    print(f'REF: {ref}')
    print(f'HYP: {response}')
    print(f'CER: {cer:.4f}')

if __name__ == '__main__':
    main()
