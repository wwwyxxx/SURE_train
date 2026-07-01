#!/usr/bin/env python3
"""Generate a 30s audio clip and a jsonl with 100 entries pointing to it."""

import json
import os
import random

import librosa
import numpy as np
import soundfile as sf

random.seed(42)

# Config
SRC_JSONL = 'data/combined_asr_aishell-1.jsonl'
TARGET_DURATION_SEC = 30
TARGET_SAMPLES = TARGET_DURATION_SEC * 16000
NUM_JSONL_ENTRIES = 1000
OUTPUT_WAV = 'data/test_audio_30s.wav'
OUTPUT_JSONL = 'data/test_audio_30s_x100.jsonl'
NUM_CLIPS = 5


def load_texts(jsonl_path, k):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    return random.sample(rows, k)


def main():
    rows = load_texts(SRC_JSONL, NUM_CLIPS)

    # Build 30s segment by concatenating several clips.
    segment = []
    segment_texts = []
    total = 0
    for row in rows:
        wav, sr = librosa.load(row['wav'], sr=16000)
        remaining = TARGET_SAMPLES - total
        if len(wav) > remaining:
            wav = wav[:remaining]
        segment.append(wav)
        segment_texts.append(row['txt'])
        total += len(wav)
        if total >= TARGET_SAMPLES:
            break

    # If not enough, loop the clips.
    while total < TARGET_SAMPLES:
        for row in rows:
            wav, sr = librosa.load(row['wav'], sr=16000)
            remaining = TARGET_SAMPLES - total
            if len(wav) > remaining:
                wav = wav[:remaining]
            segment.append(wav)
            segment_texts.append(row['txt'])
            total += len(wav)
            if total >= TARGET_SAMPLES:
                break

    segment = np.concatenate(segment)
    if len(segment) > TARGET_SAMPLES:
        segment = segment[:TARGET_SAMPLES]
    if len(segment) < TARGET_SAMPLES:
        segment = np.pad(segment, (0, TARGET_SAMPLES - len(segment)), mode='constant')

    sf.write(OUTPUT_WAV, segment, 16000)

    # Write 100 jsonl entries pointing to the same 30s wav.
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for _ in range(NUM_JSONL_ENTRIES):
            f.write(
                json.dumps(
                    {
                        'wav': OUTPUT_WAV,
                        'txt': ' '.join(segment_texts),
                        'prompt': 'Transcribe the speech to text.',
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )

    print(f'Generated {OUTPUT_WAV}')
    print(f'Duration: {len(segment) / 16000:.1f}s')
    print(f'Size: {os.path.getsize(OUTPUT_WAV) / 1024 / 1024:.1f}MB')
    print(f'JSONL: {OUTPUT_JSONL}')
    print(f'JSONL entries: {NUM_JSONL_ENTRIES}')


if __name__ == '__main__':
    main()
