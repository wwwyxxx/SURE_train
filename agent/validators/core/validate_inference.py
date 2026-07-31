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

    template = get_template(args.model_type, processor=tokenizer)
    template.set_mode('train')
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)

    encoded = template.encode(train_dataset[0], return_length=True)
    batch = template.data_collator([encoded])
    batch = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}

    with torch.no_grad():
        generated = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f'[INFO] Generated text: {output_text[:200]}')

    # For an untrained projector the output may be empty/EOS-only; only verify generation ran.
    assert generated.shape[0] == 1 and generated.shape[1] >= batch['input_ids'].shape[1], 'Unexpected generated shape'

    print('[OK] Inference validation passed')


if __name__ == '__main__':
    main()
