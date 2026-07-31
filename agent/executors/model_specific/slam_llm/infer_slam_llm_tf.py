#!/usr/bin/env python3
"""Model-specific inference + teacher-forcing check for SLAM-LLM ASR in ms-swift.

This script:
1. Loads the custom ms-swift registration file.
2. Builds the SLAM-LLM ASR wrapper from the frozen base Vicuna + WavLM.
3. Strongly loads only encoder_projector.* tensors from a trained checkpoint.
4. Runs greedy generation on a jsonl dataset.
5. Runs teacher-forcing next-token accuracy on the same dataset.

Each jsonl row is expected to contain one of:
  audio path: wav / audio / audio_path / source
  text:       txt / text / target / response
Optional:
  prompt
"""

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import torch


def load_register_module(path: str):
    spec = importlib.util.spec_from_file_location('custom_register', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load custom register module from: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def get_row_fields(row: Dict[str, Any]) -> Tuple[str, str, str]:
    wav = row.get('wav') or row.get('audio') or row.get('audio_path') or row.get('source') or row.get('path')
    text = row.get('txt') or row.get('text') or row.get('target') or row.get('response') or ''
    prompt = row.get('prompt') or 'Transcribe the speech to text.'

    if wav is None:
        raise ValueError(f'Cannot find audio path in row keys: {list(row.keys())}')
    return wav, str(text).strip(), str(prompt)


def normalize_ckpt_key(k: str) -> str:
    """Normalize common wrapper/DDP prefixes and keep encoder_projector.*."""
    for prefix in ['module.', 'model.', '_orig_mod.']:
        if k.startswith(prefix):
            k = k[len(prefix):]

    pos = k.find('encoder_projector.')
    if pos >= 0:
        k = k[pos:]
    return k


def load_projector_checkpoint(model, checkpoint_dir: str):
    """Load only encoder_projector.* tensors, with hard validation.

    This avoids strict=False silently swallowing mismatched keys.
    """
    ckpt_path = checkpoint_dir
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, 'pytorch_model.bin')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No pytorch_model.bin found at: {ckpt_path}')

    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']

    print('[CKPT] path:', ckpt_path, file=sys.stderr)
    print('[CKPT] num tensors:', len(state), file=sys.stderr)
    print('[CKPT] first 20 keys:', list(state.keys())[:20], file=sys.stderr)

    projector_state = {}
    for k, v in state.items():
        nk = normalize_ckpt_key(k)
        if nk.startswith('encoder_projector.'):
            projector_state[nk] = v

    expected = {
        'encoder_projector.linear1.weight',
        'encoder_projector.linear1.bias',
        'encoder_projector.linear2.weight',
        'encoder_projector.linear2.bias',
    }

    print('[CKPT] normalized projector keys:', sorted(projector_state.keys()), file=sys.stderr)

    missing_projector = expected - set(projector_state.keys())
    extra_projector = set(projector_state.keys()) - expected

    if missing_projector:
        raise RuntimeError(f'Missing projector tensors in checkpoint: {sorted(missing_projector)}')
    if extra_projector:
        print('[CKPT] extra projector tensors:', sorted(extra_projector), file=sys.stderr)

    before = {}
    for n, p in model.encoder_projector.named_parameters():
        full_name = 'encoder_projector.' + n
        before[full_name] = (
            float(p.detach().float().abs().mean().cpu()),
            float(p.detach().float().std().cpu()),
        )

    own_state = model.state_dict()
    with torch.no_grad():
        for k, v in projector_state.items():
            if k not in own_state:
                raise RuntimeError(f'Projector key not found in current model: {k}')
            own_state[k].copy_(v.to(device=own_state[k].device, dtype=own_state[k].dtype))

    after = {}
    for n, p in model.encoder_projector.named_parameters():
        full_name = 'encoder_projector.' + n
        after[full_name] = (
            float(p.detach().float().abs().mean().cpu()),
            float(p.detach().float().std().cpu()),
        )

    print('[CKPT] projector checksum before:', before, file=sys.stderr)
    print('[CKPT] projector checksum after:', after, file=sys.stderr)

    changed = any(before[k] != after[k] for k in before)
    if not changed:
        print(
            '[WARN] Projector checksum did not change after loading. '
            'This can be OK only if the model was already initialized from the same ckpt.',
            file=sys.stderr,
        )

    print('[CKPT] projector checkpoint loaded successfully.', file=sys.stderr)


def build_generation_batch(template, row: Dict[str, Any], device: torch.device):
    from swift.llm.template import TemplateInputs

    wav, gt_text, prompt = get_row_fields(row)
    inputs = TemplateInputs.from_dict({
        'messages': [
            {'role': 'user', 'content': f'{prompt} <audio>'},
        ],
        'audios': [wav],
    })

    encoded = template.encode(inputs, return_length=True)
    batch = template.data_collator([encoded])
    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    return batch, wav, gt_text


def build_teacher_forcing_batch(template, row: Dict[str, Any], device: torch.device):
    from swift.llm.template import TemplateInputs

    wav, gt_text, prompt = get_row_fields(row)
    inputs = TemplateInputs.from_dict({
        'messages': [
            {'role': 'user', 'content': f'{prompt} <audio>'},
            {'role': 'assistant', 'content': gt_text},
        ],
        'audios': [wav],
    })

    encoded = template.encode(inputs, return_length=True)
    batch = template.data_collator([encoded])
    batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
    return batch


@torch.no_grad()
def teacher_forcing_check(model, tokenizer, template, row: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Teacher-forcing next-token argmax accuracy on response tokens.

    This checks whether the trained projector can predict the gold response
    when all previous gold tokens are fed as context.
    """
    batch = build_teacher_forcing_batch(template, row, device)

    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
        audio=batch['audio'],
        audio_mask=batch['audio_mask'],
        modality_mask=batch['modality_mask'],
    )

    logits = outputs.logits
    labels = batch['labels']

    # Causal LM next-token shift.
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels != -100
    pred_ids = shift_logits.argmax(dim=-1)

    correct = int((pred_ids[mask] == shift_labels[mask]).sum().item())
    total = int(mask.sum().item())

    pred_resp_ids = pred_ids[mask]
    gold_resp_ids = shift_labels[mask]

    pred_text_tf = tokenizer.decode(pred_resp_ids, skip_special_tokens=True).strip()
    gold_text_tf = tokenizer.decode(gold_resp_ids, skip_special_tokens=True).strip()

    return {
        'tf_correct': correct,
        'tf_total': total,
        'tf_token_acc': correct / total if total else 0.0,
        'tf_pred_text': pred_text_tf,
        'tf_gold_text': gold_text_tf,
    }


def split_generated_new_tokens(generated: torch.Tensor, input_len: int) -> torch.Tensor:
    """Handle wrappers that return full sequence and wrappers that return only new tokens."""
    if generated.shape[1] > input_len:
        return generated[:, input_len:]
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to trained checkpoint directory or pytorch_model.bin')
    parser.add_argument('--dataset', required=True, help='Path to jsonl dataset')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--output', default=None, help='Optional output jsonl path')
    parser.add_argument('--custom-register-path',
                        default='outputs/20260709-144622/custom/slam_llm_swift_register.py')
    parser.add_argument('--base-model', default=None,
                        help='Base LLM path. Defaults to SLAM_LLM_BASE_MODEL or model/vicuna-7b-v1.5')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--debug-gen-samples', type=int, default=3)
    parser.add_argument('--debug-tf-samples', type=int, default=3)
    parser.add_argument('--no-teacher-forcing', action='store_true',
                        help='Disable teacher-forcing evaluation.')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template

    base_model_path = args.base_model or os.environ.get('SLAM_LLM_BASE_MODEL', 'model/vicuna-7b-v1.5')

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA requested but not available; falling back to CPU.', file=sys.stderr)
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    model, tokenizer = get_model_tokenizer(
        base_model_path,
        model_type='slam_llm_asr',
        torch_dtype=torch.bfloat16,
        device_map=None,
    )

    load_projector_checkpoint(model, args.checkpoint)

    model = model.to(device)
    model.eval()

    template = get_template('slam_llm_asr', processor=tokenizer)

    with open(args.dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[:args.num_samples]

    results: List[Dict[str, Any]] = []

    exact_match_count = 0
    readable_count = 0

    # Simple generated-text token match metric. This is not the same as teacher-forcing token acc.
    gen_total_response_tokens = 0
    gen_correct_tokens = 0

    tf_correct_total = 0
    tf_total_total = 0
    tf_exact_count = 0

    for idx, row in enumerate(rows):
        gen_batch, wav, gt_text = build_generation_batch(template, row, device)
        input_len = gen_batch['input_ids'].shape[1]

        with torch.no_grad():
            generated = model.generate(
                input_ids=gen_batch['input_ids'],
                attention_mask=gen_batch['attention_mask'],
                audio=gen_batch['audio'],
                audio_mask=gen_batch['audio_mask'],
                modality_mask=gen_batch['modality_mask'],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = split_generated_new_tokens(generated, input_len)

        if idx < args.debug_gen_samples:
            print('[GEN] idx:', idx, file=sys.stderr)
            print('[GEN] input_len:', input_len, file=sys.stderr)
            print('[GEN] generated shape:', tuple(generated.shape), file=sys.stderr)
            print('[GEN] decoded full:', tokenizer.decode(generated[0], skip_special_tokens=False)[:500],
                  file=sys.stderr)
            print('[GEN] decoded new:', tokenizer.decode(new_tokens[0], skip_special_tokens=False)[:500],
                  file=sys.stderr)

        pred_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        gt_text = gt_text.strip()

        gt_ids = tokenizer.encode(gt_text, add_special_tokens=False)
        pred_ids = tokenizer.encode(pred_text, add_special_tokens=False)
        min_len = min(len(gt_ids), len(pred_ids))
        if len(gt_ids) > 0:
            gen_correct_tokens += sum(a == b for a, b in zip(gt_ids[:min_len], pred_ids[:min_len]))
            gen_total_response_tokens += len(gt_ids)

        is_exact = pred_text == gt_text
        is_readable = len(pred_text) > 0 and not all(c in {'<', '|', '>', ' ', '\n'} for c in pred_text)
        if is_exact:
            exact_match_count += 1
        if is_readable:
            readable_count += 1

        item = {
            'audio': wav,
            'ground_truth': gt_text,
            'prediction': pred_text,
            'exact_match': is_exact,
            'readable': is_readable,
        }

        if not args.no_teacher_forcing:
            tf = teacher_forcing_check(model, tokenizer, template, row, device)
            tf_correct_total += tf['tf_correct']
            tf_total_total += tf['tf_total']

            tf_exact = tf['tf_pred_text'] == tf['tf_gold_text']
            if tf_exact:
                tf_exact_count += 1

            item.update({
                'teacher_forcing_token_acc': tf['tf_token_acc'],
                'teacher_forcing_exact': tf_exact,
                'teacher_forcing_prediction': tf['tf_pred_text'],
                'teacher_forcing_gold': tf['tf_gold_text'],
            })

            if idx < args.debug_tf_samples:
                print('[TF] idx:', idx, file=sys.stderr)
                print('[TF] acc:', tf['tf_token_acc'], file=sys.stderr)
                print('[TF] gold:', tf['tf_gold_text'], file=sys.stderr)
                print('[TF] pred:', tf['tf_pred_text'], file=sys.stderr)

        results.append(item)

    total = len(results)
    summary = {
        'num_samples': total,

        'generation_total_response_tokens': gen_total_response_tokens,
        'generation_correct_tokens_prefix_match': gen_correct_tokens,
        'generation_token_acc_prefix_match': (
            gen_correct_tokens / gen_total_response_tokens if gen_total_response_tokens else 0.0
        ),
        'generation_exact_match_count': exact_match_count,
        'generation_exact_match_rate': exact_match_count / total if total else 0.0,
        'readable_count': readable_count,
        'readable_rate': readable_count / total if total else 0.0,
    }

    if not args.no_teacher_forcing:
        summary.update({
            'teacher_forcing_total_tokens': tf_total_total,
            'teacher_forcing_correct_tokens': tf_correct_total,
            'teacher_forcing_token_acc': tf_correct_total / tf_total_total if tf_total_total else 0.0,
            'teacher_forcing_exact_count': tf_exact_count,
            'teacher_forcing_exact_rate': tf_exact_count / total if total else 0.0,
        })

    print(json.dumps(summary, ensure_ascii=False))

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    if readable_count == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()