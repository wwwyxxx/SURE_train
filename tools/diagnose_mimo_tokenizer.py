#!/usr/bin/env python3
"""Diagnose why MiMo audio tokenizer batch encoding fails on non-aishell1 data."""
import json
import os
import sys

import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MiMo-Audio', 'src'))
from mimo_audio_tokenizer import MiMoAudioTokenizer

SAMPLING_RATE = 24000
AUDIO_CHANNELS = 8


def load_wav(wav_path, target_sr=SAMPLING_RATE):
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def compute_mel(wav, device):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=960,
        hop_length=240,
        win_length=960,
        n_mels=128,
        f_min=0,
        f_max=None,
        power=1.0,
        center=True,
    ).to(device)
    mel = torch.log(torch.clamp(transform(wav[None, :]), min=1e-7)).squeeze().transpose(0, 1)
    return mel


def try_encode(audio_tokenizer, mel, label):
    try:
        with torch.no_grad():
            codes, _ = audio_tokenizer.encoder.encode(
                input_features=mel,
                input_lens=torch.tensor(mel.size(0), device=mel.device)[None],
                return_codes_only=True,
            )
        print(f'  {label} codes shape: {codes.shape}')
        return True
    except Exception as e:
        print(f'  {label} FAILED: {e}')
        return False


def main():
    device = 'cuda:0'
    audio_tokenizer = MiMoAudioTokenizer.from_pretrained('model/MiMo-Audio-Tokenizer')
    audio_tokenizer.eval().bfloat16().to(device)

    jsonl_path = 'data/combined_asr_local_non_aishell1.jsonl'
    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f]

    paths = ['data/aishell_audio_full/BAC009S0004W0359.wav'] + [row['wav'] for row in rows[:5]]
    for path in paths:
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), '..', path)
        print(f'\n=== {path} ===')
        try:
            wav = load_wav(path)
            print(f'wav samples: {wav.shape[-1]}, duration: {wav.shape[-1]/SAMPLING_RATE:.2f}s')
            mel = compute_mel(wav.to(device), device)
            print(f'mel shape: {mel.shape}')
            try_encode(audio_tokenizer, mel, '2D')
            try_encode(audio_tokenizer, mel[None, ...], '3D')
        except Exception as e:
            print(f'load/mel FAILED: {e}')


if __name__ == '__main__':
    main()
