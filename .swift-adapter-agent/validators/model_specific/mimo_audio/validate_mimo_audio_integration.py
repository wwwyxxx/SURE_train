#!/usr/bin/env python3
"""Comprehensive integration test for MiMo-Audio ASR registration.

Validates forward, loss, single-step training, freeze/unfreeze, and inference.
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
    parser.add_argument('--model', default='/workspace/model/MiMo-Audio-7B-Base-merged')
    parser.add_argument('--model-type', default='mimo_audio')
    parser.add_argument('--dataset-name', default='combined_asr_aishell_1')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template
    from swift.llm.dataset import load_dataset

    print(f'[INFO] Loading model_type={args.model_type} from {args.model}', flush=True)
    model, tokenizer = get_model_tokenizer(
        args.model,
        model_type=args.model_type,
        torch_dtype=torch.bfloat16,
    )
    assert model is not None, 'Model loading failed'

    # 1. Freeze/unfreeze check
    print('[INFO] Checking freeze/unfreeze...', flush=True)
    frozen_llm = all(not p.requires_grad for n, p in model.named_parameters()
                     if n.startswith('model.'))
    trainable_aligner = any(p.requires_grad for n, p in model.named_parameters()
                            if n.startswith(('speech_embeddings.', 'input_local_transformer.', 'speech_group_downcast.')))
    trainable_head = any(p.requires_grad for n, p in model.named_parameters()
                         if n.startswith('lm_head.'))
    assert frozen_llm, 'LLM backbone should be frozen'
    assert trainable_aligner, 'Audio aligner should be trainable'
    assert trainable_head, 'LM head should be trainable'
    print('[OK] Freeze/unfreeze correct')

    template = get_template(args.model_type, processor=tokenizer)
    template.mode = 'train'

    print('[INFO] Loading one sample from dataset...', flush=True)
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)
    sample = train_dataset[0]
    assert 'messages' in sample and 'audios' in sample

    encoded = template.encode(sample)
    input_ids = encoded['input_ids'].clone().detach().unsqueeze(0)
    attention_mask = encoded['attention_mask'].clone().detach().unsqueeze(0)
    position_ids = encoded['position_ids'].clone().detach().unsqueeze(0)
    labels = encoded['labels'].clone().detach().unsqueeze(0)
    text_loss_mask = encoded['text_loss_mask'].clone().detach().unsqueeze(0)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
    labels = labels.to(device)
    text_loss_mask = text_loss_mask.to(device)

    # 2. Forward + loss
    print('[INFO] Checking forward and loss...', flush=True)
    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        text_loss_mask=text_loss_mask,
    )
    loss = outputs.loss
    assert loss is not None and not torch.isnan(loss) and not torch.isinf(loss)
    print(f'[OK] Forward + loss passed. loss={loss.item():.4f}')

    # 3. Single-step training
    print('[INFO] Checking single-step training...', flush=True)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    loss0 = loss.item()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    outputs2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        text_loss_mask=text_loss_mask,
    )
    loss1 = outputs2.loss.item()
    print(f'[OK] Single-step training passed. loss before={loss0:.4f}, after={loss1:.4f}')

    # 4. Inference / generate readiness (just forward in eval mode)
    print('[INFO] Checking inference forward...', flush=True)
    model.eval()
    with torch.no_grad():
        outputs3 = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    assert outputs3.logits is not None
    print(f'[OK] Inference forward passed. logits shape={tuple(outputs3.logits.shape)}')

    print('[OK] MiMo-Audio integration test passed')


if __name__ == '__main__':
    main()
