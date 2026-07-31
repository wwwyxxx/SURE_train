#!/usr/bin/env python3
"""Validate freeze/unfreeze strategy is applied correctly.

Usage:
    python validate_freeze_unfreeze.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --expected-frozen-prefixes model.embed_tokens. model.layers. \
        --expected-trainable-prefixes model.vq_adaptor. mimo_output.

Checks:
    1. Expected frozen prefixes have requires_grad=False.
    2. Expected trainable prefixes have requires_grad=True.
"""

import argparse
import importlib.util
import sys


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
    parser.add_argument('--dataset-name', default=None)
    parser.add_argument('--expected-frozen-prefixes', nargs='+', default=['llm.', 'encoder.'])
    parser.add_argument('--expected-trainable-prefixes', nargs='+', default=['encoder_projector.'])
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer

    model, _ = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype='bfloat16', device_map=None
    )

    errors = []
    for name, p in model.named_parameters():
        frozen_match = any(name.startswith(prefix) for prefix in args.expected_frozen_prefixes)
        trainable_match = any(name.startswith(prefix) for prefix in args.expected_trainable_prefixes)

        if frozen_match and p.requires_grad:
            errors.append(f'Expected frozen but trainable: {name}')
        if trainable_match and not p.requires_grad:
            errors.append(f'Expected trainable but frozen: {name}')

    if errors:
        print('[FAIL] Freeze/unfreeze validation failed:')
        for e in errors:
            print(f'  {e}')
        sys.exit(1)

    print('[OK] Freeze/unfreeze validation passed')
    print(f'[OK] Frozen prefixes checked: {args.expected_frozen_prefixes}')
    print(f'[OK] Trainable prefixes checked: {args.expected_trainable_prefixes}')


if __name__ == '__main__':
    main()
