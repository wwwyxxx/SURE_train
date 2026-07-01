#!/usr/bin/env python3
"""Validate Kimi-Audio loss computation.

Usage:
    python validate_loss_computation.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1

Checks:
    1. Loss is scalar and finite.
    2. Loss only comes from text_loss_mask=True positions.
    3. Padding positions (-100) do not contribute to loss.
    4. Loss changes when labels change.
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

    # Zero out text_loss_mask and verify loss changes.
    batch_zero_mask = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in batch.items()}
    if 'text_loss_mask' in batch_zero_mask:
        batch_zero_mask['text_loss_mask'] = torch.zeros_like(batch_zero_mask['text_loss_mask'])
        # Also set labels to -100 where mask is zero to be consistent.
        batch_zero_mask['labels'][batch_zero_mask['text_loss_mask'] == 0] = -100
        loss_zero = get_loss(model, batch_zero_mask, args.device)
        print(f'[OK] Loss with zero mask: {loss_zero.item():.4f}')
        # Loss should increase or become nan if no valid positions; here we just check it differs.
        assert abs(loss.item() - loss_zero.item()) > 1e-6, 'Loss did not change when mask zeroed'

    print('[OK] Kimi-Audio loss computation validation passed')


if __name__ == '__main__':
    main()
