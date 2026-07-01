#!/usr/bin/env python3
"""Pre-compute MiMo Audio tokenizer RVQ codes and cache them to disk.

This script uses multiple GPUs in parallel to encode all audio files in a
jsonl dataset. The cached tokens are saved as torch .pt files and a companion
cached jsonl is written with the same structure but pointing to the cache files.
"""
import argparse
import hashlib
import json
import os
import sys
import multiprocessing as mp
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'MiMo-Audio', 'src'))
from mimo_audio_tokenizer import MiMoAudioTokenizer  # noqa: E402


def _init_spawn():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass


# MiMo-Audio config (from tokenizer config)
SAMPLING_RATE = 24000
AUDIO_CHANNELS = 8
GROUP_SIZE = 4


def _cache_key(wav_path: str) -> str:
    """Stable cache file name from wav path."""
    return hashlib.sha256(wav_path.encode('utf-8')).hexdigest()[:32]


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


def _worker(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, cache_dir: Path, audio_channels: int):
    device = f'cuda:{gpu_id}'
    audio_tokenizer = MiMoAudioTokenizer.from_pretrained('model/MiMo-Audio-Tokenizer')
    audio_tokenizer.eval().bfloat16().to(device)

    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, wav_path, row = item
        try:
            tokens = _encode_one(wav_path, audio_tokenizer, device, audio_channels)
            key = _cache_key(wav_path)
            cache_path = cache_dir / f'{key}.pt'
            torch.save(tokens, cache_path)
            result_queue.put((idx, wav_path, str(cache_path), None))
        except Exception as e:
            result_queue.put((idx, wav_path, None, str(e)))


def _load_tokenizer(device: str):
    audio_tokenizer = MiMoAudioTokenizer.from_pretrained('model/MiMo-Audio-Tokenizer')
    audio_tokenizer.eval().bfloat16().to(device)
    return audio_tokenizer


def _process_single_gpu(gpu_id: int, rows: list, cache_dir: Path, output_jsonl: str,
                        audio_channels: int, skip_existing: bool):
    device = f'cuda:{gpu_id}'
    audio_tokenizer = _load_tokenizer(device)

    results = [None] * len(rows)
    errors = []
    skipped = 0

    for idx, row in enumerate(rows):
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        wav_path = wav if os.path.isabs(wav) else os.path.join(os.path.dirname(__file__), '..', wav)
        key = _cache_key(wav_path)
        cache_path = cache_dir / f'{key}.pt'

        if skip_existing and cache_path.exists():
            results[idx] = str(cache_path)
            skipped += 1
            continue

        success = False
        for attempt in range(2):
            try:
                tokens = _encode_one(wav_path, audio_tokenizer, device, audio_channels)
                torch.save(tokens, cache_path)
                results[idx] = str(cache_path)
                success = True
                break
            except Exception as e:
                print(f'[WARN] idx={idx} wav={wav_path} attempt={attempt} failed: {e}', flush=True)
                # CUDA assert corrupts the context; reload the model and retry once
                del audio_tokenizer
                torch.cuda.empty_cache()
                audio_tokenizer = _load_tokenizer(device)

        if not success:
            errors.append((idx, wav_path))
            print(f'[ERROR] idx={idx} wav={wav_path} skipped after retry', flush=True)

        if (idx + 1) % 1000 == 0 or idx == len(rows) - 1:
            print(f'Progress: {idx + 1}/{len(rows)} (skipped {skipped}, errors {len(errors)})', flush=True)

    print(f'Finished. Errors: {len(errors)}/{len(rows)}', flush=True)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for row, cache_path in zip(rows, results):
            new_row = dict(row)
            if cache_path is not None:
                new_row['wav'] = cache_path
            f.write(json.dumps(new_row, ensure_ascii=False) + '\n')
    print(f'Cached jsonl written to {output_jsonl}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to input jsonl')
    parser.add_argument('--output-jsonl', required=True, help='Path to output cached jsonl')
    parser.add_argument('--cache-dir', required=True, help='Directory to store .pt token files')
    parser.add_argument('--gpu', type=int, default=None, help='Single GPU id to use (single-process mode)')
    parser.add_argument('--gpus', type=int, nargs='+', default=None, help='GPU ids to use (multiprocessing mode)')
    parser.add_argument('--audio-channels', type=int, default=AUDIO_CHANNELS)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    if args.gpu is not None:
        print(f'Encoding {len(rows)} audio files on cuda:{args.gpu}...', flush=True)
        _process_single_gpu(args.gpu, rows, cache_dir, args.output_jsonl,
                            args.audio_channels, args.skip_existing)
        return

    # Multi-GPU multiprocessing mode (deprecated, kept for compatibility)
    _init_spawn()
    gpus = args.gpus if args.gpus else list(range(7))
    print(f'Encoding {len(rows)} audio files using GPUs {gpus}...')

    task_queue = mp.Queue()
    result_queue = mp.Queue()

    for idx, row in enumerate(rows):
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        wav_path = wav if os.path.isabs(wav) else os.path.join(os.path.dirname(__file__), '..', wav)
        task_queue.put((idx, wav_path, row))

    for _ in gpus:
        task_queue.put(None)

    workers = []
    for gpu_id in gpus:
        p = mp.Process(target=_worker, args=(gpu_id, task_queue, result_queue, cache_dir, args.audio_channels))
        p.start()
        workers.append(p)

    results = [None] * len(rows)
    errors = []
    for _ in range(len(rows)):
        idx, wav_path, cache_path, err = result_queue.get()
        if err:
            errors.append((idx, wav_path, err))
            print(f'[ERROR] idx={idx} wav={wav_path}: {err}')
        else:
            results[idx] = cache_path
            if (idx + 1) % 1000 == 0 or idx == len(rows) - 1:
                print(f'Progress: {idx + 1}/{len(rows)}')

    for p in workers:
        p.join()

    print(f'Finished. Errors: {len(errors)}/{len(rows)}')

    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for row, cache_path in zip(rows, results):
            new_row = dict(row)
            new_row['wav'] = cache_path
            f.write(json.dumps(new_row, ensure_ascii=False) + '\n')
    print(f'Cached jsonl written to {args.output_jsonl}')


if __name__ == '__main__':
    main()
