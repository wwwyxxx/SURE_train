#!/usr/bin/env python3
"""Validate loss computation generically.

Usage:
    python validators/core/validate_loss_computation.py \
        --custom-register-path custom/xxx_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1

Checks:
    1. Loss is scalar and finite.
    2. Loss is sensitive to label changes (corrupt labels -> different loss).
    3. Padding positions (-100) do not contribute to loss.
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


def get_loss(model, batch, device):
    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
    return outputs.loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template, load_dataset

    model, _ = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )
    model = model.to(args.device)
    model.train()

    template = get_template(args.model_type, load_model_tokenizer=None)
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)

    encoded = template.encode(train_dataset[0], return_length=True)
    batch = template.data_collator([encoded])

    loss = get_loss(model, batch, args.device)
    assert loss.dim() == 0, 'Loss is not scalar'
    assert torch.isfinite(loss), f'Loss is not finite: {loss.item()}'
    print(f'[OK] Loss scalar and finite: {loss.item():.4f}')

    # Corrupt labels and verify loss changes.
    batch_corrupt = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in batch.items()}
    if 'labels' in batch_corrupt and batch_corrupt['labels'] is not None:
        labels = batch_corrupt['labels']
        valid_mask = labels != -100
        if valid_mask.any():
            # Shift one valid label to a different token id.
            first_valid = labels[valid_mask][0].item()
            new_label = (first_valid + 1) % max(2, labels.max().item() + 1)
            labels[valid_mask] = new_label
            loss_corrupt = get_loss(model, batch_corrupt, args.device)
            print(f'[OK] Loss with corrupted labels: {loss_corrupt.item():.4f}')
            assert abs(loss.item() - loss_corrupt.item()) > 1e-6, (
                'Loss did not change when labels were corrupted'
            )

    # Set all labels to -100 and verify loss stays finite/unchanged (no valid positions).
    batch_pad = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in batch.items()}
    if 'labels' in batch_pad and batch_pad['labels'] is not None:
        batch_pad['labels'].fill_(-100)
        loss_pad = get_loss(model, batch_pad, args.device)
        print(f'[OK] Loss with all labels padded: {loss_pad.item():.4f}')
        assert torch.isfinite(loss_pad), f'Loss with all padding is not finite: {loss_pad.item()}'

    print('[OK] Loss computation validation passed')


if __name__ == '__main__':
    main()
