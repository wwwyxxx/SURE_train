#!/usr/bin/env python3
"""End-to-end validator for TASU in the ms-swift harness.

Loads the custom registration, runs dataset/model/template/forward/loss/single-step/
checkpoint/freeze/inference checks, and exits non-zero on any failure.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile

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
    parser.add_argument('--model', default='model/Qwen2.5-1.5B')
    parser.add_argument('--model-type', default='tasu')
    parser.add_argument('--dataset-name', default='combined_asr_local')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template, load_dataset

    results = {}

    # 1. Model loading
    model, tokenizer = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )
    model = model.to(args.device)
    assert model is not None
    results['model_load'] = {'total_params_M': sum(p.numel() for p in model.parameters()) / 1e6}

    # 2. Dataset loading
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)
    assert train_dataset is not None and len(train_dataset) > 0
    results['dataset_load'] = {'size': len(train_dataset)}

    # 3. Template encode + collator
    template = get_template(args.model_type, tokenizer)
    encoded = template.encode(train_dataset[0], return_length=True)
    required = {'input_ids', 'labels', 'input_features', 'input_feature_length'}
    assert required <= set(encoded.keys()), f'Missing keys: {required - set(encoded.keys())}'
    batch = template.data_collator([encoded])
    assert 'input_features' in batch and 'input_feature_length' in batch
    results['template_encode'] = {
        'input_ids_len': len(encoded['input_ids']),
        'input_features_shape': list(encoded['input_features'].shape),
    }

    # 4. Forward pass
    batch_device = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}
    with torch.autocast(args.device, dtype=torch.bfloat16):
        outputs = model(**batch_device)
    assert outputs.loss is not None and outputs.loss.dim() == 0
    assert hasattr(outputs, 'logits')
    loss_val = outputs.loss.item()
    results['forward'] = {'loss': loss_val}

    # 5. Loss sensitivity
    batch_corrupt = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in batch_device.items()}
    labels = batch_corrupt['labels']
    valid_mask = labels != -100
    if valid_mask.any():
        first_valid = labels[valid_mask][0].item()
        new_label = (first_valid + 1) % max(2, labels.max().item() + 1)
        labels[valid_mask] = new_label
        with torch.autocast(args.device, dtype=torch.bfloat16):
            loss_corrupt = model(**batch_corrupt).loss.item()
        assert abs(loss_val - loss_corrupt) > 1e-6
        results['loss_sensitivity'] = {'loss_corrupt': loss_corrupt}

    # 6. Padding positions do not contribute
    batch_pad = {k: (v.clone() if hasattr(v, 'clone') else v) for k, v in batch_device.items()}
    batch_pad['labels'].fill_(-100)
    with torch.autocast(args.device, dtype=torch.bfloat16):
        loss_pad = model(**batch_pad).loss.item()
    assert torch.isfinite(torch.tensor(loss_pad))
    results['loss_all_pad'] = {'loss_pad': loss_pad}

    # 7. Single training step
    model.train()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad()
    with torch.autocast(args.device, dtype=torch.bfloat16):
        outputs = model(**batch_device)
    outputs.loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad
    optimizer.step()
    results['single_step'] = {'ok': True}

    # 8. Checkpoint save/load
    ckpt_dir = tempfile.mkdtemp(prefix='tasu_ckpt_test_')
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    trainable_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.save_pretrained(ckpt_dir)
    state_dict = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    mismatches = []
    for n, p_before in trainable_before.items():
        p_after = dict(model.named_parameters())[n]
        if not torch.allclose(p_before, p_after.cpu(), atol=1e-3):
            mismatches.append(n)
    assert not mismatches, f'Mismatched params: {mismatches}'
    results['checkpoint'] = {'ok': True, 'path': ckpt_dir}

    # 9. Freeze/unfreeze check
    frozen_ok = all(not p.requires_grad for n, p in model.named_parameters()
                    if n.startswith('llm.') or n.startswith('encoder.'))
    trainable_ok = any(p.requires_grad for n, p in model.named_parameters() if n.startswith('encoder_projector.'))
    assert frozen_ok and trainable_ok
    results['freeze'] = {'frozen_llm_encoder': frozen_ok, 'trainable_projector': trainable_ok}

    # 10. Inference (pass full batch so audio features are available)
    model.eval()
    infer_batch = {k: v[:1].to(args.device) if hasattr(v, 'to') else v[:1] for k, v in batch.items()}
    with torch.no_grad():
        generated = model.generate(
            input_ids=infer_batch['input_ids'],
            attention_mask=infer_batch['attention_mask'],
            input_features=infer_batch['input_features'],
            input_feature_length=infer_batch['input_feature_length'],
            max_new_tokens=20,
            do_sample=False,
        )
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    assert len(output_text.strip()) > 0
    results['inference'] = {'output': output_text[:200]}

    report = {
        'passed': True,
        'results': results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(json.dumps({'passed': False, 'error': str(e)}, indent=2, ensure_ascii=False), file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
