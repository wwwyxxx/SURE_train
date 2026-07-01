import os
import sys
import torch

# Allow running directly from host source tree when ms-swift is mounted at /workspace/ms-swift
sys.path.insert(0, '/workspace/ms-swift')

from swift.llm import PtEngine, InferRequest, get_template

MODEL_DIR = '/workspace/models/Qwen2.5-Omni-7B'

def main():
    if not os.path.isdir(MODEL_DIR):
        print(f'Model directory not found: {MODEL_DIR}', file=sys.stderr)
        sys.exit(1)

    os.environ['MAX_PIXELS'] = '1003520'

    print('Loading model and processor...')
    engine = PtEngine(
        MODEL_DIR,
        model_type='qwen2_5_omni',
        attn_impl='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    print('model class:', engine.model.__class__.__name__)
    print('processor class:', engine.processor.__class__.__name__)

    template = get_template('qwen2_5_omni', engine.processor)
    engine.default_template = template

    request = InferRequest(messages=[{
        'role': 'user',
        'content': '你好，请简单介绍一下自己。',
    }])
    print('Query:', request.messages[0]['content'])
    resp_list = engine.infer([request])
    response = resp_list[0].choices[0].message.content
    print('Response:', response)


if __name__ == '__main__':
    main()
