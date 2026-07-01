#!/usr/bin/env python3
"""Validate MiMo-Audio template registration."""
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
    parser.add_argument('--dataset-name', default='combined_asr_aishell_1')
    parser.add_argument('--num-samples', type=int, default=2)
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_template, get_model_tokenizer, load_dataset

    _, processor = get_model_tokenizer(
        '/workspace/model/MiMo-Audio-7B-Base-merged',
        model_type=args.model_type,
        load_model=False,
    )
    template = get_template(args.model_type, processor=processor)
    assert template is not None, f'Template for {args.model_type} not found'
    template.mode = 'train'
    print(f'[OK] Template retrieved: {type(template).__name__}')

    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)
    print(f'[OK] Dataset loaded, size={len(train_dataset)}')

    encoded_samples = []
    for i in range(min(args.num_samples, len(train_dataset))):
        raw = train_dataset[i]
        encoded = template.encode(raw, return_length=True)
        print(f'[INFO] Sample {i} keys: {encoded.keys()}')
        print(f'[INFO]   input_ids shape: {tuple(encoded["input_ids"].shape)}')
        print(f'[INFO]   labels shape: {tuple(encoded["labels"].shape)}')
        print(f'[INFO]   text_loss_mask sum: {encoded["text_loss_mask"].sum().item()}')
        encoded_samples.append(encoded)

    batch = template.data_collator(encoded_samples)
    print(f'[OK] Collated keys: {batch.keys()}')
    for key, val in batch.items():
        print(f'[INFO]   collated {key}: shape={tuple(val.shape)}, dtype={val.dtype}')

    # Check loss positions are non-padding
    labels = batch['labels']
    text_loss_mask = batch['text_loss_mask']
    assert (labels[text_loss_mask] != -100).all(), 'Loss positions should not be -100'
    print('[OK] Template registration validation passed')


if __name__ == '__main__':
    main()
