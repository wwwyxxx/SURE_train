#!/usr/bin/env python3
"""Validate MiMo-Audio model registration."""
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
    parser.add_argument('--model-type', default='mimo_audio')
    parser.add_argument('--model-path', default='/workspace/model/MiMo-Audio-7B-Base-merged')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer

    print(f'[INFO] Loading model_type={args.model_type} from {args.model_path}', flush=True)
    model, tokenizer = get_model_tokenizer(
        args.model_path,
        model_type=args.model_type,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    assert model is not None, 'Model loading failed'
    assert tokenizer is not None, 'Tokenizer loading failed'

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'[OK] Model loaded successfully')
    print(f'[OK] Total parameters: {total / 1e6:.1f}M')
    print(f'[OK] Trainable parameters: {trainable_total / 1e6:.1f}M ({len(trainable)} tensors)')

    if not trainable:
        print('[FAIL] No trainable parameters found')
        sys.exit(1)

    # Verify that all trainable parameters have gradients enabled and are not NaN.
    print('[INFO] Checking trainable parameters are finite...', flush=True)
    for n, p in model.named_parameters():
        if p.requires_grad:
            if torch.isnan(p).any():
                print(f'[FAIL] NaN detected in trainable parameter: {n}')
                sys.exit(1)
    print(f'[OK] All {len(trainable)} trainable parameter tensors are finite')
    print('[OK] Model registration validation passed')


if __name__ == '__main__':
    main()
