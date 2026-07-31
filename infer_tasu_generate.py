#!/usr/bin/env python3
"""Autoregressive generation inference for TASU checkpoint evaluation.

Runs the TASU model in autoregressive generate mode and reports per-sample
predictions plus CER (character error rate) for Chinese ASR.  Supports
incremental resume so that long-running full-set evals can survive pre-emption.
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


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m]


def normalize(text: str) -> str:
    return text.strip().replace(' ', '').replace('\n', '')


def flush_output(path: str, results: list):
    """Write results incrementally so that partial progress survives pre-emption."""
    total_edits = sum(r.get('edits', 0) for r in results)
    total_ref_chars = sum(r.get('ref_chars', 0) for r in results)
    summary = {
        'num_samples': len(results),
        'total_edits': total_edits,
        'total_ref_chars': total_ref_chars,
        'cer': total_edits / total_ref_chars if total_ref_chars else 0.0,
    }
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp, path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=200)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--resume', action='store_true',
                        help='Skip samples already present in the output file.')
    parser.add_argument('--custom-register-path',
                        default='custom/tasu_swift_register.py')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    load_register_module(args.custom_register_path)
    from swift.llm import get_model_tokenizer, get_template
    from swift.llm.template import TemplateInputs

    base_model_path = os.environ.get('TASU_BASE_MODEL', 'model/Qwen2.5-1.5B')
    model, tokenizer = get_model_tokenizer(
        base_model_path,
        model_type='tasu',
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    ckpt_path = os.path.join(args.checkpoint, 'pytorch_model.bin')
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f'[INFO] loaded checkpoint {ckpt_path}, missing={len(missing)}, unexpected={len(unexpected)}',
              file=sys.stderr)
    else:
        print(f'[WARN] No checkpoint at {ckpt_path}; using base weights.', file=sys.stderr)

    if args.device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.to(args.device)
    model.eval()

    template = get_template('tasu', tokenizer)

    with open(args.dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    if args.num_samples is not None:
        rows = rows[:args.num_samples]

    # Resume: load already-processed samples from the output file.
    processed = set()
    results = []
    if args.resume and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if 'audio' in obj and 'prediction' in obj:
                    processed.add(obj['audio'])
                    results.append(obj)
        print(f'[INFO] resume: {len(processed)} samples already processed', file=sys.stderr)

    pending = [row for row in rows if (row.get('wav') or row.get('audio') or row.get('audio_path') or row.get('path')) not in processed]
    print(f'[INFO] total={len(rows)}, already_done={len(results)}, pending={len(pending)}', file=sys.stderr)

    for i, row in enumerate(pending):
        wav = row.get('wav') or row.get('audio') or row.get('audio_path') or row.get('path')
        text = row.get('txt') or row.get('text') or row.get('response') or row.get('target') or ''
        prompt = row.get('prompt') or 'Transcribe the speech to text.'

        inputs = TemplateInputs.from_dict({
            'messages': [
                {'role': 'user', 'content': f'{prompt}<speech>'},
                {'role': 'assistant', 'content': text},
            ],
            'audios': [wav],
        })
        encoded = template.encode(inputs, return_length=True)
        batch = template.data_collator([encoded])
        batch = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}

        prompt_str = (
            f'<|im_start|>user\n{prompt}<speech><|im_end|>\n'
            f'<|im_start|>assistant\n'
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
        prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.bool)

        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
            generated = model.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                input_features=batch['input_features'],
                input_feature_length=batch['input_feature_length'],
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )

        # TASU's generate() returns only the newly generated tokens.
        gen_ids = generated[0]
        pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        ref = normalize(text)
        hyp = normalize(pred_text)
        edits = levenshtein(ref, hyp)
        ref_chars = len(ref)

        results.append({
            'audio': wav,
            'ground_truth': text,
            'prediction': pred_text,
            'cer': edits / ref_chars if ref_chars else 0.0,
            'edits': edits,
            'ref_chars': ref_chars,
        })

        if (i + 1) % 50 == 0:
            summary = flush_output(args.output, results)
            done = len(results)
            print(f'[{done}/{len(rows)}] running CER={summary["cer"]:.4f}', file=sys.stderr)

    summary = flush_output(args.output, results)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
