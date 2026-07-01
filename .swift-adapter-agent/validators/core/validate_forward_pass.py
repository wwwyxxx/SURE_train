#!/usr/bin/env python3
"""Validate that model forward pass works end-to-end.

Usage:
    python validate_forward_pass.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1

Checks:
    1. Template can encode a sample.
    2. data_collator can produce a batch.
    3. model(**batch) runs without error.
    4. outputs contain loss/logits of expected shape.
"""

import argparse
import importlib.util
import sys

import torch


def load_register_module(path: str):
    spec = importlib.util.spec_from_file_location('custom_register', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template, load_dataset

    model, tokenizer = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )
    model = model.to(args.device)
    model.train()

    template = get_template(args.model_type, load_model_tokenizer=None)
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)

    encoded = template.encode(train_dataset[0], return_length=True)
    batch = template.data_collator([encoded])

    # Move tensors to device.
    batch = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)

    assert outputs.loss is not None, 'loss is None'
    assert outputs.loss.dim() == 0, 'loss is not scalar'
    assert hasattr(outputs, 'logits'), 'outputs missing logits'
    print(f'[OK] Forward pass succeeded')
    print(f'[OK] loss={outputs.loss.item():.4f}')
    print(f'[OK] logits type={type(outputs.logits)}')


if __name__ == '__main__':
    main()
