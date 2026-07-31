#!/usr/bin/env python3
"""Validate that a single training step (forward + backward) works.

Usage:
    python validate_single_step_training.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1 \
        --lr 1e-4
"""

import argparse
import importlib.util
import sys

import torch
from torch.optim import AdamW


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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template, load_dataset

    model, tokenizer = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )
    model = model.to(args.device)
    model.train()

    template = get_template(args.model_type, processor=tokenizer)
    template.set_mode('train')
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)

    encoded = template.encode(train_dataset[0], return_length=True)
    batch = template.data_collator([encoded])
    batch = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)

    loss = outputs.loss
    loss.backward()

    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, 'No gradients produced'

    optimizer.step()
    optimizer.zero_grad()

    print(f'[OK] Single training step succeeded')
    print(f'[OK] loss={loss.item():.4f}')
    print(f'[OK] Gradients produced and optimizer stepped')


if __name__ == '__main__':
    main()
