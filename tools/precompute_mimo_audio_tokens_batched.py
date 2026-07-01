#!/usr/bin/env python3
"""Pre-compute MiMo Audio tokenizer RVQ codes with batching for efficiency.

Uses a single GPU but processes audio in sorted batches to minimize padding
and maximize GPU utilization. Much faster than one-by-one processing.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MiMo-Audio', 'src'))
from mimo_audio_tokenizer import MiMoAudioTokenizer  # noqa: E402

SAMPLING_RATE = 24000
AUDIO_CHANNELS = 8
GROUP_SIZE = 4


def _cache_key(wav_path: str) -> str:
    return hashlib.sha256(wav_path.encode('utf-8')).hexdigest()[:32]


def _audio_length(wav_path: str) -> int:
    """Return number of samples without loading the full waveform."""
    try:
        info = torchaudio.info(wav_path)
        return info.num_frames
    except Exception:
        # fallback
        wav, sr = torchaudio.load(wav_path)
        return wav.shape[-1]


def _encode_one(wav_path: str, audio_tokenizer, device: str, audio_channels: int):
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
    wav = wav.to(device)

    mel_transform = torchaudio.transforms.MelSpectrogram(
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
    mel = torch.log(torch.clamp(mel_transform(wav[None, :]), min=1e-7)).squeeze().transpose(0, 1)

    # Encode in chunks to avoid OOM on long audio
    input_len = mel.size(0)
    segment_size = 6000
    input_len_seg = [segment_size] * (input_len // segment_size)
    if input_len % segment_size > 0:
        input_len_seg.append(input_len % segment_size)

    codes_list = []
    for features, lengths in zip(
        torch.split(mel, input_len_seg),
        [torch.tensor(x, device=device) for x in input_len_seg],
    ):
        with torch.no_grad():
            codes, _ = audio_tokenizer.encoder.encode(
                input_features=features,
                input_lens=lengths[None],
                return_codes_only=True,
            )
        codes_list.append(codes)

    codes = torch.cat(codes_list, dim=-1)  # [num_quantizers, T]
    audio_codes = codes[:audio_channels].transpose(0, 1).detach().cpu()  # [T, audio_channels]

    # Pad to multiple of group_size
    num_timesteps = audio_codes.shape[0]
    if num_timesteps % GROUP_SIZE != 0:
        padding_needed = GROUP_SIZE - (num_timesteps % GROUP_SIZE)
        last_tokens = audio_codes[-1:, :]
        padding_tokens = last_tokens.repeat(padding_needed, 1)
        audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

    return audio_codes.reshape(-1)  # [T * audio_channels]


class MelComputer:
    def __init__(self, device: str):
        self.device = device
        self.transform = torchaudio.transforms.MelSpectrogram(
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

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        mel = torch.log(torch.clamp(self.transform(wav[None, :]), min=1e-7)).squeeze().transpose(0, 1)
        return mel


def _load_wav(wav_path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


def _encode_batch(audio_tokenizer, mel_computer, wav_paths: list, device: str, audio_channels: int):
    """Encode a batch of wav files and return list of per-sample token tensors."""
    # Load and compute mel for each sample
    mels = []
    mel_lens = []
    for wp in wav_paths:
        wav = _load_wav(wp, SAMPLING_RATE).to(device)
        mel = mel_computer(wav)
        mels.append(mel)
        mel_lens.append(mel.shape[0])

    max_len = max(mel_lens)
    batch_size = len(mels)
    padded = torch.full((batch_size, max_len, 128), 0.0, dtype=torch.float32, device=device)
    for i, mel in enumerate(mels):
        padded[i, :mel.shape[0]] = mel

    input_lens = torch.tensor(mel_lens, dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, output_lengths, codes = audio_tokenizer.encoder.encode(
            input_features=padded,
            input_lens=input_lens,
            return_codes_only=True,
        )

    # codes: [n_q, total_T]; output_lengths: [B]
    output_lengths = output_lengths.cpu()
    codes = codes.cpu()

    results = []
    start = 0
    for out_len in output_lengths:
        end = start + out_len.item()
        sample_codes = codes[:, start:end]  # [n_q, T]
        audio_codes = sample_codes[:audio_channels].transpose(0, 1)  # [T, audio_channels]

        # Pad to multiple of group_size
        num_timesteps = audio_codes.shape[0]
        if num_timesteps % GROUP_SIZE != 0:
            padding_needed = GROUP_SIZE - (num_timesteps % GROUP_SIZE)
            last_tokens = audio_codes[-1:, :]
            padding_tokens = last_tokens.repeat(padding_needed, 1)
            audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

        results.append(audio_codes.reshape(-1))
        start = end
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--audio-channels', type=int, default=AUDIO_CHANNELS)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip audio files whose cache file already exists')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    print(f'Loading {len(rows)} rows...', flush=True)
    items = []
    for idx, row in enumerate(rows):
        wav = row['wav']
        wav_path = wav if os.path.isabs(wav) else os.path.join(os.path.dirname(__file__), '..', wav)
        items.append((idx, wav_path, row))

    results = [None] * len(rows)
    if args.skip_existing:
        filtered = []
        skipped = 0
        for idx, wav_path, row in items:
            key = _cache_key(wav_path)
            cache_path = cache_dir / f'{key}.pt'
            if cache_path.exists():
                results[idx] = str(cache_path)
                skipped += 1
            else:
                filtered.append((idx, wav_path, row))
        items = filtered
        print(f'Skipped {skipped} already-cached files', flush=True)

    print(f'Will encode {len(items)} valid audio files on cuda:{args.gpu} with batch size {args.batch_size}', flush=True)

    device = f'cuda:{args.gpu}'
    audio_tokenizer = MiMoAudioTokenizer.from_pretrained('model/MiMo-Audio-Tokenizer')
    audio_tokenizer.eval().bfloat16().to(device)
    mel_computer = MelComputer(device)

    cache_paths = [None] * len(rows)
    errors = []

    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start:batch_start + args.batch_size]
        idxs = [it[0] for it in batch]
        wav_paths = [it[1] for it in batch]
        try:
            tokens_list = _encode_batch(audio_tokenizer, mel_computer, wav_paths, device, args.audio_channels)
            for idx, tokens, wp in zip(idxs, tokens_list, wav_paths):
                key = _cache_key(wp)
                cache_path = cache_dir / f'{key}.pt'
                torch.save(tokens, cache_path)
                results[idx] = str(cache_path)
        except Exception as e:
            print(f'[WARN] batch {batch_start}-{batch_start+len(batch)} failed ({e}), falling back to single-sample encoding.', flush=True)
            # Fallback: encode one by one to isolate bad samples
            for idx, wp in zip(idxs, wav_paths):
                try:
                    tokens = _encode_one(wp, audio_tokenizer, device, args.audio_channels)
                    key = _cache_key(wp)
                    cache_path = cache_dir / f'{key}.pt'
                    torch.save(tokens, cache_path)
                    results[idx] = str(cache_path)
                except Exception as e2:
                    errors.append((idx, str(e2)))
                    print(f'[ERROR] idx={idx} wav={wp}: {e2}', flush=True)

        if (batch_start + len(batch)) % 1000 == 0 or batch_start + len(batch) >= len(items):
            print(f'Progress: {batch_start + len(batch)}/{len(items)}', flush=True)

    print(f'Finished. Errors: {len(errors)}/{len(items)}', flush=True)

    # Write cached jsonl
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for row, cache_path in zip(rows, results):
            new_row = dict(row)
            if cache_path is not None:
                new_row['wav'] = cache_path
            f.write(json.dumps(new_row, ensure_ascii=False) + '\n')

    print(f'Cached jsonl written to {args.output_jsonl}')


if __name__ == '__main__':
    main()
