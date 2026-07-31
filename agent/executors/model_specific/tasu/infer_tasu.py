#!/usr/bin/env python3
"""Model-specific inference script for TASU in the ms-swift harness.

Loads a TASU checkpoint trained via ms-swift and runs greedy generation on a
jsonl dataset.  Each row is expected to contain at least `wav`/`audio` and
`txt`/`text`/`response` fields.
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
                        default='outputs/20260708-203711/custom/tasu_swift_register.py')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)
    from swift.llm import get_model_tokenizer, get_template
    from swift.llm.template import TemplateInputs

    # TASU's save_pretrained only stores trainable tensors, so the checkpoint
    # directory itself is not a complete swift model path.  Load the base model
    # from the original LLM path and then overlay the trained projector weights.
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

    template = get_template('tasu', tokenizer)

    with open(args.dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    rows = rows[:args.num_samples]

    results = []
    exact_match_count = 0
    readable_count = 0
    for row in rows:
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        text = row.get('txt') or row.get('text') or row.get('response') or ''
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

        with torch.no_grad():
            generated = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                input_features=batch['input_features'],
                input_feature_length=batch['input_feature_length'],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        pred_text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        gt_text = text.strip()

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
        'exact_match_count': exact_match_count,
        'exact_match_rate': exact_match_count / total if total else 0.0,
        'readable_count': readable_count,
        'readable_rate': readable_count / total if total else 0.0,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Non-zero exit if no readable outputs at all.
    if readable_count == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
