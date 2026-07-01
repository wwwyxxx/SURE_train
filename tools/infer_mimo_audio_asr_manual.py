#!/usr/bin/env python3
"""Manual greedy ASR inference for MiMo-Audio (avoids broken generate() in newer transformers)."""
import argparse
import json
import os
import sys

import torch
import torchaudio
from transformers import AutoTokenizer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MIMO_AUDIO_ROOT = os.path.join(_ROOT, 'MiMo-Audio')
if _MIMO_AUDIO_ROOT not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_ROOT)

from src.mimo_audio.modeling_mimo_audio import MiMoAudioArguments, MiMoAudioForCausalLM  # noqa: E402
from src.mimo_audio.process_speechdata import InputSegment  # noqa: E402
from src.mimo_audio_tokenizer import MiMoAudioTokenizer  # noqa: E402

_SPECIAL_TOKENS = [
    '<|sosp|>', '<|eosp|>', '<|empty|>', '<|Human|>',
    '<|SpeechLM|>', '<|sostm|>', '<|eostm|>', '<|eot|>',
]


def _ensure_special_tokens(tokenizer):
    for token in _SPECIAL_TOKENS:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
    return tokenizer


def _get_mel_transform(cfg):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sampling_rate,
        n_fft=cfg.nfft,
        hop_length=cfg.hop_length,
        win_length=cfg.window_size,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        n_mels=cfg.n_mels,
        power=1.0,
        center=True,
    ).to('cuda')


def encode_audio(wav_path: str, audio_tokenizer: MiMoAudioTokenizer, audio_channels: int = 8, group_size: int = 4):
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    target_sr = audio_tokenizer.config.sampling_rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    device = next(audio_tokenizer.parameters()).device
    wav = wav.to(device)
    mel = torch.log(torch.clamp(_get_mel_transform(audio_tokenizer.config)(wav[None, :]), min=1e-7)).squeeze().transpose(0, 1)

    input_len = mel.size(0)
    segment_size = 6000
    input_len_seg = [segment_size] * (input_len // segment_size)
    if input_len % segment_size > 0:
        input_len_seg.append(input_len % segment_size)

    codes_list = []
    for features, lengths in zip(torch.split(mel, input_len_seg), [torch.tensor(x, device=device) for x in input_len_seg]):
        with torch.no_grad():
            codes, _ = audio_tokenizer.encoder.encode(
                input_features=features,
                input_lens=lengths[None],
                return_codes_only=True,
            )
        codes_list.append(codes)

    codes = torch.cat(codes_list, dim=-1)
    audio_codes = codes[:audio_channels].transpose(0, 1).detach().cpu()

    num_timesteps = audio_codes.shape[0]
    if num_timesteps % group_size != 0:
        padding_needed = group_size - (num_timesteps % group_size)
        last_tokens = audio_codes[-1:, :]
        padding_tokens = last_tokens.repeat(padding_needed, 1)
        audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)
    return audio_codes.reshape(-1)


def build_prompt_input_ids(tokenizer, audio_tokens, prompt, speech_zeroemb_idx, empty_idx, group_size, audio_channels):
    segments = [
        InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(audio=audio_tokens, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text=prompt, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
    ]
    return torch.cat([
        seg.to_input_id(tokenizer, group_size, audio_channels)
        for seg in segments
    ], dim=1).long()


def greedy_generate(model, tokenizer, input_ids, speech_zeroemb_idx, empty_idx, max_new_tokens=128):
    group_size = model.config.group_size
    audio_channels = model.config.audio_channels
    eos_token_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    generated_ids = []

    for _ in range(max_new_tokens):
        T = input_ids.shape[-1]
        T_groups = T // group_size
        attention_mask = torch.ones(1, T_groups, dtype=torch.bool, device=input_ids.device)
        position_ids = torch.arange(T_groups, device=input_ids.device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        next_token = int(outputs.text_logits[0, -1, :].argmax().item())
        if next_token in (eos_token_id, im_end_id):
            break
        generated_ids.append(next_token)

        # append a new group: text token at group start, empty elsewhere
        new_group = torch.full((1, audio_channels + 1, group_size), empty_idx, dtype=torch.long, device=input_ids.device)
        for c in range(1, audio_channels + 1):
            new_group[0, c, :] = speech_zeroemb_idx[c - 1]
        new_group[0, 0, 0] = next_token
        input_ids = torch.cat([input_ids, new_group], dim=-1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', default='data/combined_asr_aishell-1.jsonl')
    parser.add_argument('--audio-tokenizer', default='model/MiMo-Audio-Tokenizer')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()

    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_ROOT, args.checkpoint)
    dataset = args.dataset if os.path.isabs(args.dataset) else os.path.join(_ROOT, args.dataset)
    audio_tokenizer_path = args.audio_tokenizer if os.path.isabs(args.audio_tokenizer) else os.path.join(_ROOT, args.audio_tokenizer)

    print(f'[infer] Loading tokenizer from {checkpoint}')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    tokenizer = _ensure_special_tokens(tokenizer)

    args_obj = MiMoAudioArguments(
        model_name_or_path=checkpoint,
        sosp_idx=tokenizer.convert_tokens_to_ids('<|sosp|>'),
        eosp_idx=tokenizer.convert_tokens_to_ids('<|eosp|>'),
        empty_idx=tokenizer.convert_tokens_to_ids('<|empty|>'),
        sostm_idx=tokenizer.convert_tokens_to_ids('<|sostm|>'),
        eostm_idx=tokenizer.convert_tokens_to_ids('<|eostm|>'),
        eot_idx=tokenizer.convert_tokens_to_ids('<|eot|>'),
    )

    print(f'[infer] Loading model from {checkpoint}')
    model = MiMoAudioForCausalLM.from_pretrained(
        checkpoint,
        args=args_obj,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
    )
    model.eval()

    group_size = model.config.group_size
    audio_channels = model.config.audio_channels
    speech_zeroemb_idx = model.config.parsed_speech_empty_ids()
    empty_idx = args_obj.empty_idx

    print(f'[infer] Loading audio tokenizer from {audio_tokenizer_path}')
    audio_tokenizer = MiMoAudioTokenizer.from_pretrained(audio_tokenizer_path)
    audio_tokenizer.eval().bfloat16().to('cuda')

    print(f'[infer] Reading dataset {dataset}')
    with open(dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    selected = rows[args.start:args.start + args.num_samples]
    for idx, row in enumerate(selected, start=args.start):
        wav = row['wav'] if os.path.isabs(row['wav']) else os.path.join(_ROOT, row['wav'])
        prompt = row.get('prompt') or 'Transcribe the speech to text.'
        target = row.get('txt') or row.get('text') or row.get('response') or ''

        audio_tokens = encode_audio(wav, audio_tokenizer, audio_channels, group_size)
        input_ids = build_prompt_input_ids(tokenizer, audio_tokens, prompt, speech_zeroemb_idx, empty_idx, group_size, audio_channels)
        input_ids = input_ids.unsqueeze(0).to(model.model.embed_tokens.weight.device)

        pred = greedy_generate(model, tokenizer, input_ids, speech_zeroemb_idx, empty_idx, args.max_new_tokens)
        ok = 'OK' if pred.replace(' ', '') == target.replace(' ', '') else 'BAD'
        print(f'\n[{idx}] {ok}')
        print(f'target: {target}')
        print(f'pred:   {pred}')


if __name__ == '__main__':
    main()
