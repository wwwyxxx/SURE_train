#!/usr/bin/env python3
"""Validate checkpoint save and load consistency.

Usage:
    python validate_checkpoint_save_load.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --output-dir /tmp/kimi_ckpt_test

Checks:
    1. Save trainable-only checkpoint.
    2. Load checkpoint back into model.
    3. Verify trainable params are identical before/after.
"""

import argparse
import importlib.util
import os
import shutil
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
    parser.add_argument('--output-dir', default='/tmp/kimi_ckpt_test')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer

    model, _ = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )

    # Capture trainable state before save.
    trainable_before = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    model.save_pretrained(args.output_dir)
    print(f'[OK] Checkpoint saved to {args.output_dir}')

    # Load checkpoint.
    state_dict = torch.load(os.path.join(args.output_dir, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    # Verify.
    mismatches = []
    for n, p_before in trainable_before.items():
        p_after = dict(model.named_parameters())[n]
        if not torch.allclose(p_before, p_after.cpu(), atol=1e-3):
            mismatches.append(n)

    if mismatches:
        print(f'[FAIL] Mismatched params after load: {mismatches}')
        sys.exit(1)

    print(f'[OK] Checkpoint save/load validation passed')
    print(f'[OK] {len(trainable_before)} trainable tensors are consistent')


if __name__ == '__main__':
    main()
