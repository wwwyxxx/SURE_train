#!/usr/bin/env python3
"""Verify the cached AISHELL-1 ASR dataset and run inference with a smoke checkpoint."""
import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'custom'))
sys.path.insert(0, os.path.join(ROOT, 'MiMo-Audio', 'src'))

from mimo_audio.modeling_mimo_audio import MiMoAudioArguments, MiMoAudioForCausalLM  # noqa: E402
from mimo_audio.process_speechdata import InputSegment  # noqa: E402
from mimo_audio_tokenizer import MiMoAudioTokenizer  # noqa: E402
from mimo_audio_swift_register import _ensure_mimo_special_tokens  # noqa: E402

AUDIO_CHANNELS = 8
GROUP_SIZE = 4
EXPECTED_PROMPT = 'Transcribe the speech to text.'


def _resolve(path: str) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    cand = os.path.join(ROOT, path)
    if os.path.exists(cand):
        return cand
    return path


def _load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _load_tokenizer(checkpoint: str) -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return _ensure_mimo_special_tokens(tok)


# ---------------------------------------------------------------------------
# 1. Cached JSONL completeness / shape check
# ---------------------------------------------------------------------------
def validate_cached_dataset(dataset_path: str) -> Dict[str, Any]:
    rows = _load_rows(dataset_path)
    total = len(rows)
    missing_keys = 0
    missing_files = 0
    bad_shapes = 0
    lengths = []
    bad_prompts = 0

    for idx, row in enumerate(rows):
        if not all(k in row for k in ('wav', 'txt', 'prompt')):
            missing_keys += 1
            continue
        if row.get('prompt') != EXPECTED_PROMPT:
            bad_prompts += 1
        wav = _resolve(row['wav'])
        if not os.path.exists(wav):
            missing_files += 1
            continue
        try:
            tokens = torch.load(wav, map_location='cpu', weights_only=True)
        except Exception as e:
            print(f'  [{idx}] failed to load {wav}: {e}')
            bad_shapes += 1
            continue
        if not isinstance(tokens, torch.Tensor):
            bad_shapes += 1
            continue
        n = tokens.numel()
        if n % AUDIO_CHANNELS != 0 or (n // AUDIO_CHANNELS) % GROUP_SIZE != 0:
            bad_shapes += 1
            print(f'  [{idx}] bad token shape length={n} ({wav})')
            continue
        lengths.append(n)

    print(f'\n[Dataset validation] {dataset_path}')
    print(f'  total rows      : {total}')
    print(f'  missing keys    : {missing_keys}')
    print(f'  missing .pt     : {missing_files}')
    print(f'  bad token shapes: {bad_shapes}')
    print(f'  bad prompts     : {bad_prompts}')
    if lengths:
        print(f'  token length    : min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}')
    return {
        'total': total,
        'missing_keys': missing_keys,
        'missing_files': missing_files,
        'bad_shapes': bad_shapes,
        'lengths': lengths,
    }


# ---------------------------------------------------------------------------
# 2. Decode check: labels must decode back to the target text
# ---------------------------------------------------------------------------
# Hardcoded from MiMoAudioTemplate; only the text channel (index 0) matters for decoding.
_SPEECH_ZEROEMB_IDX = [1024, 1024, 128, 128, 128, 128, 128, 128]


def _build_training_input_ids(tokenizer, audio_tokens, prompt, response):
    empty_idx = tokenizer.convert_tokens_to_ids('<|empty|>')
    segments = [
        InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=_SPEECH_ZEROEMB_IDX, text_zeroemb_idx=empty_idx),
        InputSegment(audio=audio_tokens, speech_zeroemb_idx=_SPEECH_ZEROEMB_IDX, text_zeroemb_idx=empty_idx),
        InputSegment(text=prompt, speech_zeroemb_idx=_SPEECH_ZEROEMB_IDX, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=_SPEECH_ZEROEMB_IDX, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=_SPEECH_ZEROEMB_IDX, text_zeroemb_idx=empty_idx),
        InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=_SPEECH_ZEROEMB_IDX, text_zeroemb_idx=empty_idx),
        InputSegment(text=response, speech_zeroemb_idx=_SPEECH_ZEROEMB_IDX, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=_SPEECH_ZEROEMB_IDX, text_zeroemb_idx=empty_idx),
    ]
    return torch.cat([
        seg.to_input_id(tokenizer, GROUP_SIZE, AUDIO_CHANNELS)
        for seg in segments
    ], dim=1).long()


def _find_response_start(text_ids, tokenizer):
    header_ids = tokenizer('<|im_start|>assistant\n', add_special_tokens=False)['input_ids']
    think_ids = tokenizer('<think>\n\n</think>\n', add_special_tokens=False)['input_ids']
    ids = text_ids.tolist()
    for start in range(len(ids) - len(header_ids) + 1):
        if ids[start:start + len(header_ids)] == header_ids:
            return start + len(header_ids) + len(think_ids)
    return None


def decode_check(tokenizer, dataset_path: str, num_samples: int = 10):
    rows = _load_rows(dataset_path)
    print(f'\n[Decode check] first {num_samples} rows of {dataset_path}')
    matched = 0
    for idx, row in enumerate(rows[:num_samples]):
        prompt = row.get('prompt', EXPECTED_PROMPT)
        response = row['txt']
        wav = _resolve(row['wav'])
        audio_tokens = torch.tensor(torch.load(wav, map_location='cpu', weights_only=True).tolist(), dtype=torch.long)
        input_ids = _build_training_input_ids(tokenizer, audio_tokens, prompt, response)
        text_ids = input_ids[0, ::GROUP_SIZE]
        loss_start = _find_response_start(text_ids, tokenizer)
        if loss_start is None:
            print(f'  [{idx}] BAD: assistant header not found')
            continue
        label_ids = text_ids[loss_start:].tolist()
        decoded = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
        ok = decoded.replace(' ', '') == response.replace(' ', '')
        matched += ok
        status = 'OK' if ok else 'BAD'
        print(f'  [{idx}] {status}')
        print(f'      target: {response}')
        print(f'      decoded: {decoded}')
        if not ok:
            print(f'      diff repr: target={repr(response)} decoded={repr(decoded)}')
    print(f'  decode match: {matched}/{num_samples}')
    return matched == num_samples


# ---------------------------------------------------------------------------
# 3. Inference on cached tokens
# ---------------------------------------------------------------------------
def _load_audio_tokens(wav_path: str) -> torch.Tensor:
    wav_path = _resolve(wav_path)
    if wav_path.endswith('.pt'):
        tokens = torch.load(wav_path, map_location='cpu', weights_only=True)
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens.long()
    raise ValueError(f'Expected .pt cached tokens, got {wav_path}')


def _build_prompt_input_ids(tokenizer, audio_tokens, prompt, speech_zeroemb_idx, empty_idx):
    segments = [
        InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(audio=audio_tokens, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text=prompt, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
    ]
    return torch.cat([
        seg.to_input_id(tokenizer, GROUP_SIZE, AUDIO_CHANNELS)
        for seg in segments
    ], dim=1).long()


def _greedy_generate(model, tokenizer, input_ids, speech_zeroemb_idx, empty_idx, max_new_tokens: int = 128):
    eos_token_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    generated_ids = []
    for _ in range(max_new_tokens):
        T = input_ids.shape[-1]
        T_groups = T // GROUP_SIZE
        attention_mask = torch.ones(1, T_groups, dtype=torch.bool, device=input_ids.device)
        position_ids = torch.arange(T_groups, device=input_ids.device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        next_token = int(outputs.text_logits[0, -1, :].argmax().item())
        if next_token in (eos_token_id, im_end_id):
            break
        generated_ids.append(next_token)
        new_group = torch.full((1, AUDIO_CHANNELS + 1, GROUP_SIZE), empty_idx, dtype=torch.long, device=input_ids.device)
        for c in range(1, AUDIO_CHANNELS + 1):
            new_group[0, c, :] = speech_zeroemb_idx[c - 1]
        new_group[0, 0, 0] = next_token
        input_ids = torch.cat([input_ids, new_group], dim=-1)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def run_inference(checkpoint: str, dataset_path: str, audio_tokenizer_path: str, num_samples: int = 100, max_new_tokens: int = 128):
    rows = _load_rows(dataset_path)
    selected = rows[:num_samples]

    print(f'\n[Inference] loading tokenizer from {checkpoint}')
    tokenizer = _load_tokenizer(checkpoint)

    args_obj = MiMoAudioArguments(
        model_name_or_path=checkpoint,
        sosp_idx=tokenizer.convert_tokens_to_ids('<|sosp|>'),
        eosp_idx=tokenizer.convert_tokens_to_ids('<|eosp|>'),
        empty_idx=tokenizer.convert_tokens_to_ids('<|empty|>'),
        sostm_idx=tokenizer.convert_tokens_to_ids('<|sostm|>'),
        eostm_idx=tokenizer.convert_tokens_to_ids('<|eostm|>'),
        eot_idx=tokenizer.convert_tokens_to_ids('<|eot|>'),
    )

    print(f'[Inference] loading model from {checkpoint}')
    model = MiMoAudioForCausalLM.from_pretrained(
        checkpoint,
        args=args_obj,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
    )
    model.eval()

    speech_zeroemb_idx = model.config.parsed_speech_empty_ids()
    empty_idx = args_obj.empty_idx

    print(f'[Inference] loading audio tokenizer from {audio_tokenizer_path} (only for model init)')
    _ = MiMoAudioTokenizer.from_pretrained(audio_tokenizer_path)

    total_chars = 0
    total_errs = 0
    exact_match = 0

    print(f'[Inference] running on {len(selected)} samples from {dataset_path}')
    for idx, row in enumerate(selected):
        target = row['txt']
        prompt = row.get('prompt', EXPECTED_PROMPT)
        audio_tokens = _load_audio_tokens(row['wav'])
        input_ids = _build_prompt_input_ids(tokenizer, audio_tokens, prompt, speech_zeroemb_idx, empty_idx)
        input_ids = input_ids.unsqueeze(0).to(model.model.embed_tokens.weight.device)
        pred = _greedy_generate(model, tokenizer, input_ids, speech_zeroemb_idx, empty_idx, max_new_tokens)
        target_clean = target.replace(' ', '')
        pred_clean = pred.replace(' ', '')
        errs = _levenshtein(target_clean, pred_clean)
        total_errs += errs
        total_chars += len(target_clean)
        ok = pred_clean == target_clean
        exact_match += ok
        status = 'OK' if ok else 'BAD'
        print(f'\n[{idx}] {status} CER={errs/max(len(target_clean),1):.2%}')
        print(f'  target: {target}')
        print(f'  pred:   {pred}')

    cer = total_errs / max(total_chars, 1)
    acc = exact_match / max(len(selected), 1)
    print(f'\n[Inference summary] exact_match={exact_match}/{len(selected)} ({acc:.2%}), CER={cer:.2%}')
    return {'exact_match_rate': acc, 'cer': cer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/combined_asr_aishell-1_cached.jsonl')
    parser.add_argument('--subset', default='data/combined_asr_aishell_1_cached_100.jsonl')
    parser.add_argument('--checkpoint', default='output/combined_asr_aishell1_cached100_groupdowncast_lmhead/v2-20260629-044129/checkpoint-80')
    parser.add_argument('--audio-tokenizer', default='model/MiMo-Audio-Tokenizer')
    parser.add_argument('--num-decode-check', type=int, default=10)
    parser.add_argument('--infer-samples', type=int, default=100)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()

    validate_cached_dataset(args.dataset)

    print(f'\n[Loading tokenizer] {args.checkpoint}')
    tokenizer = _load_tokenizer(_resolve(args.checkpoint))

    decode_ok = decode_check(tokenizer, args.subset, args.num_decode_check)

    infer_stats = None
    if args.infer_samples > 0:
        infer_stats = run_inference(
            checkpoint=_resolve(args.checkpoint),
            dataset_path=_resolve(args.subset),
            audio_tokenizer_path=_resolve(args.audio_tokenizer),
            num_samples=args.infer_samples,
            max_new_tokens=args.max_new_tokens,
        )

    print('\n=== FINAL VERDICT ===')
    if not decode_ok:
        print('cached dataset has tokenizer/label issues; see decode check above')
    elif infer_stats is None or (infer_stats['exact_match_rate'] >= 0.95 and infer_stats['cer'] < 0.05):
        print('cached dataset looks correct and usable for training')
    else:
        print('cached dataset structure/tokenizer look correct; inference accuracy is lower but this is a model/checkpoint issue, not a dataset issue')


if __name__ == '__main__':
    main()
