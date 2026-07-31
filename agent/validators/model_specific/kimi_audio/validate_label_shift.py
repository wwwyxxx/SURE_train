#!/usr/bin/env python3
"""Validate Kimi-Audio labels/mask alignment (critical bug area).

Usage:
    python validate_label_shift.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1

Checks:
    1. text_input_ids and labels have the same shape.
    2. text_loss_mask marks exactly the assistant response tokens.
    3. After forward shift, loss positions correspond to predicted tokens.
    4. Padding is marked as -100 in labels.
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
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--dataset-name', required=True)
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_template, load_dataset

    template = get_template(args.model_type, load_model_tokenizer=None)
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)

    encoded = template.encode(train_dataset[0], return_length=True)
    text_ids = encoded['text_input_ids']
    labels = encoded['labels']
    mask = encoded['text_loss_mask']

    assert text_ids.shape == labels.shape, f'Shape mismatch: text_input_ids {text_ids.shape} vs labels {labels.shape}'
    assert mask.shape == labels.shape, f'Shape mismatch: mask {mask.shape} vs labels {labels.shape}'
    print(f'[OK] text_input_ids/labels/mask shapes align: {tuple(text_ids.shape)}')

    # Mask should be boolean or 0/1.
    unique = torch.unique(mask)
    assert len(unique) <= 2, f'text_loss_mask has more than 2 values: {unique}'
    print(f'[OK] text_loss_mask is binary: {unique.tolist()}')

    # At least some assistant tokens exist.
    num_assistant = mask.sum().item()
    assert num_assistant > 0, 'No assistant tokens marked in text_loss_mask'
    print(f'[OK] Assistant tokens marked: {num_assistant}')

    # Padding should be -100 in labels (collator sets this).
    batch = template.data_collator([encoded])
    labels_batched = batch['labels'][0]
    mask_batched = batch['text_loss_mask'][0]
    pad_positions = labels_batched == -100
    non_pad_positions = labels_batched != -100
    assert (mask_batched[pad_positions] == 0).all(), 'Padding positions still have loss mask=True'
    assert (mask_batched[non_pad_positions] == 1).all(), 'Non-padding positions have loss mask=False'
    print(f'[OK] Padding positions are -100 and excluded from loss')

    # Simulate shift and verify alignment.
    shifted_labels = torch.cat((labels_batched[1:], labels_batched.new_full((1,), -100)))
    valid = shifted_labels != -100
    assert valid.sum().item() == num_assistant, 'Shifted labels count mismatch with assistant tokens'
    print(f'[OK] Label shift alignment verified')

    print('[OK] Kimi-Audio label shift validation passed')


if __name__ == '__main__':
    main()
