import argparse
import os
import sys
import json
import torch

def levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
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
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--model_dir', default='/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-Omni-7B')
    parser.add_argument('--plugins', nargs='+', default=[])
    args = parser.parse_args()

    # Make ms-swift importable inside docker
    sys.path.insert(0, '/workspace/ms-swift')
    sys.path.insert(0, '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/ms-swift')
    sys.path.insert(0, '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train')

    # Load plugins if provided
    for plugin in args.plugins:
        plugin_path = plugin if os.path.isabs(plugin) else os.path.join('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train', plugin)
        spec = __import__('importlib.util').util.spec_from_file_location('plugin', plugin_path)
        mod = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    from swift.llm import PtEngine, InferRequest, get_template

    model_dir = args.model_dir
    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train', args.checkpoint)
    audio = args.audio if os.path.isabs(args.audio) else os.path.join('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train', args.audio)

    os.environ['MAX_PIXELS'] = '1003520'
    os.environ['ENABLE_AUDIO_OUTPUT'] = '0'

    print(f'Loading model from {model_dir} with checkpoint {checkpoint}...')

    # Use official qwen2_5_omni template; the custom model uses same template
    model_type = 'qwen2_5_omni_custom' if args.plugins else 'qwen2_5_omni'
    engine = PtEngine(
        model_dir,
        model_type=model_type,
        attn_impl='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map='auto',
        adapters=[checkpoint],
    )
    print('model class:', engine.model.__class__.__name__)
    print('processor class:', engine.processor.__class__.__name__)

    template = get_template('qwen2_5_omni', engine.processor)
    engine.default_template = template

    request = InferRequest(messages=[{
        'role': 'user',
        'content': [{'type': 'audio', 'audio_url': audio}, {'type': 'text', 'text': args.prompt}]
    }])
    print('Query:', request.messages[0]['content'])
    resp_list = engine.infer([request])
    response = resp_list[0].choices[0].message.content
    print('Response:', response)

    # Compute CER
    ref = '新科空调'
    cer = levenshtein(response, ref) / max(len(ref), 1)
    print(f'REF: {ref}')
    print(f'HYP: {response}')
    print(f'CER: {cer:.4f}')

if __name__ == '__main__':
    main()
