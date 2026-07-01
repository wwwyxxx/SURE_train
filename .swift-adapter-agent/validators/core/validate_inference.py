#!/usr/bin/env python3
"""Validate inference pipeline produces reasonable text.

Usage:
    python validate_inference.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1 \
        --max-new-tokens 128

Checks:
    1. Model can do greedy generation.
    2. Output is non-empty and contains text tokens.
    3. Output does not collapse to a single repeated token.
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
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template, load_dataset

    model, tokenizer = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )
    model = model.to(args.device)
    model.eval()

    template = get_template(args.model_type, load_model_tokenizer=None)
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)

    encoded = template.encode(train_dataset[0], return_length=True)
    batch = template.data_collator([encoded])
    batch = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}

    input_ids = batch['input_ids']
    attention_mask = batch.get('attention_mask')

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f'[INFO] Generated text: {output_text[:200]}')

    assert len(output_text.strip()) > 0, 'Generated text is empty'

    # Check for collapse (e.g. same token repeated > 80%).
    tokens = tokenizer.encode(output_text)
    if tokens:
        most_common_ratio = max(tokens.count(t) for t in set(tokens)) / len(tokens)
        assert most_common_ratio < 0.8, f'Output collapsed, most common token ratio={most_common_ratio:.2f}'

    print('[OK] Inference validation passed')


if __name__ == '__main__':
    main()
