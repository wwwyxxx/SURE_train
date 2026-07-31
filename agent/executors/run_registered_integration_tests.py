#!/usr/bin/env python3
"""Run lightweight integration test for ms-swift registered (native) models.

This script does NOT require a custom register file. It only needs:
- model_type
- model path
- dataset name or path

Usage:
    python executors/run_registered_integration_tests.py \
        --model-type qwen2_audio \
        --model /workspace/model/Qwen2-Audio-7B \
        --dataset-name combined_asr_custom \
        --custom-register-path outputs/{run_id}/custom/qwen2_audio_dataset_register.py \
        --output outputs/{run_id}/integration_test_report.json

The --custom-register-path is optional; use it only if the dataset was registered
via a custom Python file.
"""

import argparse
import importlib.util
import json
import os
import sys


def load_register_module(path: str):
    spec = importlib.util.spec_from_file_location('custom_register', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def test_load_model(model_path: str, model_type: str, device: str, external_plugins: str | None = None):
    import torch
    if external_plugins:
        load_register_module(external_plugins)
    from swift.llm import get_model_tokenizer
    model, tokenizer = get_model_tokenizer(
        model_path,
        model_type=model_type,
        torch_dtype=torch.bfloat16,
        device_map=None,
        model_kwargs={'device_map': None, 'low_cpu_mem_usage': True},
    )
    model = model.to(device)
    return model, tokenizer


def test_load_dataset(dataset_name: str, custom_register_path: str = None):
    if custom_register_path:
        load_register_module(custom_register_path)
    from swift.llm import load_dataset
    train_dataset, _ = load_dataset([dataset_name], split_dataset_ratio=0.0)
    return train_dataset


def test_forward(model, template, train_dataset, device: str):
    import torch
    model.train()
    encoded = template.encode(train_dataset[0], return_length=True)
    batch = template.data_collator([encoded])
    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

    # Some templates (e.g. step_audio2_mini) do not return labels from encode().
    # For the integration test we can use input_ids shifted by one as labels.
    if 'labels' not in batch:
        batch['labels'] = batch['input_ids'].clone()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)

    if outputs.loss is None:
        # Manually compute loss for models that don't return it
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        outputs.loss = loss

    assert outputs.loss is not None, 'loss is None'
    assert outputs.loss.dim() == 0, 'loss is not scalar'
    return outputs.loss.item()


def test_single_step(model, template, train_dataset, device: str, lr: float = 1e-4):
    import torch
    from torch.optim import AdamW
    model.train()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    encoded = template.encode(train_dataset[0], return_length=True)
    batch = template.data_collator([encoded])
    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

    if 'labels' not in batch:
        batch['labels'] = batch['input_ids'].clone()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)

    if outputs.loss is None:
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        outputs.loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs.loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--custom-register-path', default=None)
    parser.add_argument('--external-plugins', default=None,
                        help='Path to custom model-register Python file to load before get_model_tokenizer.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    results = []
    all_passed = True

    try:
        from swift.llm import get_template

        model, processor = test_load_model(args.model, args.model_type, args.device, args.external_plugins)
        results.append({'name': 'load_model', 'passed': True, 'message': 'Model loaded'})

        train_dataset = test_load_dataset(args.dataset_name, args.custom_register_path)
        results.append({'name': 'load_dataset', 'passed': True, 'message': f'Dataset loaded: {len(train_dataset)} samples'})

        template = get_template(args.model_type, processor)
        loss = test_forward(model, template, train_dataset, args.device)
        results.append({'name': 'forward_pass', 'passed': True, 'message': f'loss={loss:.4f}'})

        loss = test_single_step(model, template, train_dataset, args.device, lr=args.lr)
        results.append({'name': 'single_step_training', 'passed': True, 'message': f'loss={loss:.4f}'})

    except Exception as e:
        import traceback
        results.append({
            'name': 'integration_test',
            'passed': False,
            'message': str(e),
            'traceback': traceback.format_exc(),
        })
        all_passed = False

    report = {
        'passed': all_passed,
        'model_type': args.model_type,
        'model': args.model,
        'dataset_name': args.dataset_name,
        'results': results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
