#!/usr/bin/env python3
"""Model-specific inference script for SLAM-LLM ASR in the ms-swift harness.

Loads a SLAM-LLM checkpoint trained via ms-swift and runs greedy generation on a
jsonl dataset. Each row is expected to contain `wav`/`audio` and `txt`/`text`/`target`.
"""
import argparse
import importlib.util
import json
import os
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
    parser.add_argument('--checkpoint', required=True, help='Path to trained checkpoint directory')
    parser.add_argument('--dataset', required=True, help='Path to jsonl dataset')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--output', default=None, help='Optional output jsonl path')
    parser.add_argument('--custom-register-path',
                        default='outputs/20260709-144622/custom/slam_llm_swift_register.py')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)
    from swift.llm import get_model_tokenizer, get_template
    from swift.llm.template import TemplateInputs

    base_model_path = os.environ.get('SLAM_LLM_BASE_MODEL', 'model/vicuna-7b-v1.5')
    model, tokenizer = get_model_tokenizer(
        base_model_path,
        model_type='slam_llm_asr',
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    ckpt_path = os.path.join(args.checkpoint, 'pytorch_model.bin')
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        #DEBUG
        print('[CKPT] path:', ckpt_path, file=sys.stderr)
        print('[CKPT] num tensors:', len(state_dict), file=sys.stderr)
        print('[CKPT] first 20 keys:', list(state_dict.keys())[:20], file=sys.stderr)
        print(
            '[CKPT] projector keys:',
            [k for k in state_dict.keys() if 'encoder_projector' in k],
            file=sys.stderr,
        )
        if missing:
            print(f'[WARN] Missing keys when loading checkpoint: {missing[:10]}', file=sys.stderr)
        if unexpected:
            print(f'[WARN] Unexpected keys when loading checkpoint: {unexpected[:10]}', file=sys.stderr)
    else:
        print(f'[WARN] No pytorch_model.bin found at {args.checkpoint}; using base model only.', file=sys.stderr)

    if args.device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.to(args.device)
    model.eval()

    template = get_template('slam_llm_asr', processor=tokenizer)

    with open(args.dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    rows = rows[:args.num_samples]

    results = []
    exact_match_count = 0
    readable_count = 0
    total_response_tokens = 0
    correct_tokens = 0

    for idx, row in enumerate(rows):
        wav = row.get('wav') or row.get('audio') or row.get('audio_path') or row.get('source') or row.get('path')
        text = row.get('txt') or row.get('text') or row.get('target') or row.get('response') or ''
        prompt = row.get('prompt') or 'Transcribe the speech to text.'

        inputs = TemplateInputs.from_dict({
            'messages': [
                {'role': 'user', 'content': f'{prompt} <audio>'},
            ],
            'audios': [wav],
        })
        encoded = template.encode(inputs, return_length=True)
        batch = template.data_collator([encoded])
        batch = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        input_len = batch['input_ids'].shape[1]

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                audio=batch['audio'],
                audio_mask=batch['audio_mask'],
                modality_mask=batch['modality_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        if idx < 3:
            print('[GEN] input_len:', input_len, file=sys.stderr)
            print('[GEN] generated shape:', tuple(generated.shape), file=sys.stderr)
            print('[GEN] decoded full:', tokenizer.decode(generated[0], skip_special_tokens=False)[:500], file=sys.stderr)
            print('[GEN] decoded new:', tokenizer.decode(generated[0, input_len:], skip_special_tokens=False)[:500], file=sys.stderr)
        pred_text = tokenizer.decode(generated[0, input_len:], skip_special_tokens=True).strip()
        gt_text = text.strip()

        # Token-level accuracy on the response (for reporting, not strict).
        gt_ids = tokenizer.encode(gt_text, add_special_tokens=False)
        pred_ids = tokenizer.encode(pred_text, add_special_tokens=False)
        min_len = min(len(gt_ids), len(pred_ids))
        if min_len > 0:
            correct_tokens += sum(a == b for a, b in zip(gt_ids[:min_len], pred_ids[:min_len]))
            total_response_tokens += len(gt_ids)

        is_exact = pred_text == gt_text
        is_readable = len(pred_text) > 0 and not all(c in {'<', '|', '>', ' ', '\n'} for c in pred_text)
        if is_exact:
            exact_match_count += 1
        if is_readable:
            readable_count += 1

        results.append({
            'audio': wav,
            'ground_truth': gt_text,
            'prediction': pred_text,
            'exact_match': is_exact,
            'readable': is_readable,
        })

    total = len(results)
    summary = {
        'num_samples': total,
        'total_response_tokens': total_response_tokens,
        'correct_tokens': correct_tokens,
        'token_acc': correct_tokens / total_response_tokens if total_response_tokens else 0.0,
        'exact_match_count': exact_match_count,
        'exact_match_rate': exact_match_count / total if total else 0.0,
        'readable_count': readable_count,
        'readable_rate': readable_count / total if total else 0.0,
    }
    print(json.dumps(summary, ensure_ascii=False))

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    if readable_count == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
