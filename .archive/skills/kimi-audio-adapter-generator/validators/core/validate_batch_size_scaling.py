#!/usr/bin/env python3
"""Find maximum per-device batch size that fits in GPU memory.

Usage:
    python validate_batch_size_scaling.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1 \
        --batch-sizes 1 2 4 8 16 \
        --device cuda:0

Checks:
    1. For each batch size, try forward + backward.
    2. Report peak memory and whether OOM occurred.
"""

import argparse
import importlib.util

import torch


def load_register_module(path: str):
    spec = importlib.util.spec_from_file_location('custom_register', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def try_batch_size(model, template, dataset, bs, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        encoded = [template.encode(dataset[i], return_length=True) for i in range(min(bs, len(dataset)))]
        while len(encoded) < bs:
            encoded.append(encoded[0])
        batch = template.data_collator(encoded)
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
        outputs.loss.backward()
        model.zero_grad()

        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        return True, peak
    except torch.OutOfMemoryError:
        return False, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16])
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

    print(f'{"Batch Size":>12} | {"Status":>10} | {"Peak Memory (GiB)":>18}')
    print('-' * 50)
    for bs in args.batch_sizes:
        ok, peak = try_batch_size(model, template, train_dataset, bs, args.device)
        status = 'OK' if ok else 'OOM'
        peak_str = f'{peak:.2f}' if ok else 'N/A'
        print(f'{bs:>12} | {status:>10} | {peak_str:>18}')


if __name__ == '__main__':
    main()
