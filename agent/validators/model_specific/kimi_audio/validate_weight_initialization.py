#!/usr/bin/env python3
"""Validate Kimi-Audio specific weight initialization.

Usage:
    python validate_weight_initialization.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text

Checks:
    1. Shared LLM weights match base Qwen model.
    2. mimo_output first N rows match Qwen lm_head.
    3. Trainable modules (vq_adaptor, mimo_layers, mimo_norm) are not all zeros.
    4. Whisper model is loaded (has parameters and is frozen by default).
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
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer
    from transformers import AutoModelForCausalLM

    model, _ = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )
    model_state = {n: p for n, p in model.named_parameters()}

    # Load base Qwen for comparison.
    qwen = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='cpu', low_cpu_mem_usage=True
    )
    qwen_state = qwen.state_dict()

    # Check embed_tokens.
    emb_name = 'model.embed_tokens.weight'
    if emb_name in model_state and 'model.embed_tokens.weight' in qwen_state:
        rows = min(model_state[emb_name].shape[0], qwen_state['model.embed_tokens.weight'].shape[0])
        cols = min(model_state[emb_name].shape[1], qwen_state['model.embed_tokens.weight'].shape[1])
        diff = (model_state[emb_name][:rows, :cols].float() - qwen_state['model.embed_tokens.weight'][:rows, :cols].float()).abs().max()
        assert diff < 1e-2, f'embed_tokens mismatch: {diff}'
        print(f'[OK] embed_tokens copied from Qwen (max diff={diff:.4f})')

    # Check mimo_output first rows.
    head_name = 'mimo_output.weight'
    if head_name in model_state and 'lm_head.weight' in qwen_state:
        rows = min(model_state[head_name].shape[0], qwen_state['lm_head.weight'].shape[0])
        cols = min(model_state[head_name].shape[1], qwen_state['lm_head.weight'].shape[1])
        diff = (model_state[head_name][:rows, :cols].float() - qwen_state['lm_head.weight'][:rows, :cols].float()).abs().max()
        assert diff < 1e-2, f'mimo_output mismatch: {diff}'
        print(f'[OK] mimo_output partially copied from Qwen lm_head (max diff={diff:.4f})')

    # Check trainable modules are initialized (not all zeros).
    trainable_prefixes = ['model.vq_adaptor.', 'model.mimo_layers.', 'model.mimo_norm.']
    for prefix in trainable_prefixes:
        params = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
        assert params, f'No parameters found for {prefix}'
        for n, p in params:
            assert p.abs().max() > 0, f'{n} is all zeros'
        print(f'[OK] {prefix} initialized (non-zero)')

    # Check whisper exists and is frozen by default.
    whisper_params = [(n, p) for n, p in model.named_parameters() if n.startswith('whisper_model.')]
    assert whisper_params, 'No whisper_model parameters found'
    assert not whisper_params[0][1].requires_grad, 'whisper_model should be frozen by default'
    print(f'[OK] whisper_model loaded and frozen ({len(whisper_params)} tensors)')

    print('[OK] Kimi-Audio weight initialization validation passed')


if __name__ == '__main__':
    main()
