#!/usr/bin/env python3
"""Validate MiMo-Audio forward pass and loss on a real sample."""
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
    print(f'[INFO] speech_vocab_size={model.config.speech_vocab_size}', flush=True)
    print(f'[INFO] speech_zeroemb_idx={model.config.speech_zeroemb_idx}', flush=True)
    for i, emb in enumerate(model.speech_embeddings):
        print(f'[INFO] speech_embeddings[{i}] num_embeddings={emb.num_embeddings}', flush=True)

    template = get_template(args.model_type, processor=tokenizer)
    template.mode = 'train'

    print('[INFO] Loading one sample from dataset...', flush=True)
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)
    sample = train_dataset[0]
    assert 'messages' in sample and 'audios' in sample

    encoded = template.encode(sample)
    input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.bool).unsqueeze(0)
    position_ids = torch.tensor(encoded['position_ids'], dtype=torch.long).unsqueeze(0)
    labels = torch.tensor(encoded['labels'], dtype=torch.long).unsqueeze(0)
    text_loss_mask = torch.tensor(encoded['text_loss_mask'], dtype=torch.bool).unsqueeze(0)

    print(f'[INFO] input_ids shape: {tuple(input_ids.shape)}', flush=True)
    for c in range(input_ids.shape[1]):
        print(f'[INFO] channel {c}: min={input_ids[0, c].min().item()}, max={input_ids[0, c].max().item()}', flush=True)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
    labels = labels.to(device)
    text_loss_mask = text_loss_mask.to(device)

    model.train()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        labels=labels,
        text_loss_mask=text_loss_mask,
    )
    loss = outputs.loss
    assert loss is not None, 'Loss is None'
    assert not torch.isnan(loss), 'Loss is NaN'
    assert not torch.isinf(loss), 'Loss is Inf'

    print(f'[OK] Forward + loss passed. loss={loss.item():.4f}')
    print('[OK] MiMo-Audio forward validation passed')


if __name__ == '__main__':
    main()
