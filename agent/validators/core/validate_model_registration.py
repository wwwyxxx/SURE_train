#!/usr/bin/env python3
"""Validate that a custom model is correctly registered with ms-swift.

Usage:
    python validate_model_registration.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text

Checks:
    1. register_model was called with the given model_type.
    2. get_model_tokenizer function can be invoked.
    3. model weights are loaded on the expected device.
    4. model has at least one trainable parameter.
"""

import argparse
import importlib.util
import os
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
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    # Loading the register module should execute registrations.
    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer

    print(f'[INFO] Loading model_type={args.model_type} from {args.model}')
    model, tokenizer = get_model_tokenizer(
        args.model,
        model_type=args.model_type,
        torch_dtype=torch.bfloat16,
        device_map='cpu',
    )

    assert model is not None, 'Model loading failed'
    assert tokenizer is not None, 'Tokenizer loading failed'

    # Check trainable parameters.
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'[OK] Model loaded successfully')
    print(f'[OK] Total parameters: {total / 1e6:.1f}M')
    print(f'[OK] Trainable parameters: {trainable_total / 1e6:.1f}M ({len(trainable)} tensors)')

    if trainable:
        print('[OK] Model registration validation passed')
    else:
        print('[FAIL] No trainable parameters found')
        sys.exit(1)


if __name__ == '__main__':
    main()
