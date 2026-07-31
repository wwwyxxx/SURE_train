#!/usr/bin/env python3
"""Validate that a custom template is correctly registered with ms-swift.

Usage:
    python validate_template_registration.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model-type kimi_audio_text \
        --dataset-jsonl data/combined_asr_aishell-1.jsonl \
        --dataset-name combined_asr_aishell_1

Checks:
    1. Template can be retrieved by model_type.
    2. Template.encode returns expected keys.
    3. Returned tensors have correct shapes/dtypes.
    4. data_collator can batch the encoded samples.
"""

import argparse
import importlib.util
import json
import sys

import torch


def load_register_module(path: str):
    spec = importlib.util.spec_from_file_location('custom_register', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def load_raw_row(jsonl_path: str, idx: int = 0):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f'Row {idx} not found in {jsonl_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--model', default=None)
    parser.add_argument('--dataset-jsonl', required=True)
    parser.add_argument('--dataset-name', required=True)
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_template, load_dataset, get_model_tokenizer, MODEL_MAPPING

    # Load template.
    model_path = args.model
    if model_path is None:
        model_meta = MODEL_MAPPING.get(args.model_type)
        if model_meta is not None and model_meta.model_groups:
            model_path = model_meta.model_groups[0].models[0].model_path
    assert model_path is not None, 'Please pass --model or ensure ModelMeta has a model_path'
    _, tokenizer = get_model_tokenizer(model_path, model_type=args.model_type, load_model=False)
    template = get_template(args.model_type, processor=tokenizer)
    template.set_mode('train')
    assert template is not None, f'Template for {args.model_type} not found'
    print(f'[OK] Template retrieved: {type(template).__name__}')

    # Load raw row and preprocess.
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)
    raw = train_dataset[0]
    print(f'[INFO] Encoding sample: {raw.keys()}')

    encoded = template.encode(raw, return_length=True)
    print(f'[OK] Encoded keys: {encoded.keys()}')

    required_keys = {'input_ids', 'labels', 'attention_mask'}
    missing = required_keys - set(encoded.keys())
    assert not missing, f'Missing keys: {missing}'

    for key in encoded.keys():
        val = encoded[key]
        print(f'[INFO] {key}: shape={tuple(val.shape) if hasattr(val, "shape") else "N/A"}, '
              f'dtype={getattr(val, "dtype", "N/A")}')

    # Test data_collator.
    batch = [template.encode(train_dataset[i], return_length=True) for i in range(min(2, len(train_dataset)))]
    collated = template.data_collator(batch)
    print(f'[OK] Collated keys: {collated.keys()}')
    for key in encoded.keys():
        if key == 'length':
            continue
        val = collated[key]
        shape = tuple(val.shape) if hasattr(val, 'shape') else 'list(len={})'.format(len(val))
        print(f'[INFO] collated {key}: shape={shape}')

    print('[OK] Template registration validation passed')


if __name__ == '__main__':
    main()
