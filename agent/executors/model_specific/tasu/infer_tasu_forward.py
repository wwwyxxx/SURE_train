#!/usr/bin/env python3
"""Forward-pass inference/evaluation for TASU smoke tests.

Avoids the model's autoregressive generate() path (which is sensitive to
prompt formatting in the current ms-swift integration) and instead checks
whether the model predicts the assistant response tokens correctly given the
full input sequence.  This is a reliable signal that the projector has learned
the audio -> text mapping.
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
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--custom-register-path',
                        default='outputs/20260708-203711/custom/tasu_swift_register.py')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

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
        if missing:
            print(f'[WARN] Missing keys: {missing[:10]}', file=sys.stderr)
        if unexpected:
            print(f'[WARN] Unexpected keys: {unexpected[:10]}', file=sys.stderr)
    else:
        print(f'[WARN] No checkpoint found at {ckpt_path}; using base model.', file=sys.stderr)

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
    total_tokens = 0
    correct_tokens = 0
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

        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                input_features=batch['input_features'],
                input_feature_length=batch['input_feature_length'],
                labels=batch['labels'],
            )

        # The TASU wrapper expands the single <speech> placeholder into multiple
        # audio tokens inside the model.  The logits therefore correspond to the
        # expanded sequence, while the template's labels still use the original
        # (unexpanded) positions.  Re-align the response positions.
        input_len = batch['input_ids'].shape[1]
        logits_len = outputs.logits.shape[1]
        audio_tokens = logits_len - input_len + 1
        speech_token_id = tokenizer.convert_tokens_to_ids('<speech>')
        speech_positions = (batch['input_ids'][0] == speech_token_id).nonzero(as_tuple=True)[0]
        speech_offset = audio_tokens - 1

        labels = batch['labels'][0]
        response_positions = (labels != -100).nonzero(as_tuple=True)[0]
        if len(response_positions) == 0:
            continue
        # All response tokens appear after the <speech> placeholder.
        final_response_positions = response_positions + speech_offset

        # Predictions are logits at position t-1 for token t.
        pred_ids = outputs.logits[0, final_response_positions - 1].argmax(dim=-1)
        target_ids = labels[response_positions]

        correct = (pred_ids == target_ids).sum().item()
        total = len(response_positions)
        total_tokens += total
        correct_tokens += correct

        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
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
            'token_acc': correct / total if total else 0.0,
            'exact_match': is_exact,
            'readable': is_readable,
        })

    total = len(results)
    summary = {
        'num_samples': total,
        'total_response_tokens': total_tokens,
        'correct_tokens': correct_tokens,
        'token_acc': correct_tokens / total_tokens if total_tokens else 0.0,
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


if __name__ == '__main__':
    main()
