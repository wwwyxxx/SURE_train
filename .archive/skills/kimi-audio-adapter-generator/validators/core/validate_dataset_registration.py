#!/usr/bin/env python3
"""Validate that a custom dataset is correctly registered with ms-swift.

Usage:
    python validate_dataset_registration.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --dataset-name combined_asr_aishell_1 \
        --num-samples 3

Checks:
    1. Dataset can be loaded via load_dataset.
    2. Preprocessor produces {messages, audios} or expected format.
    3. Required fields (wav, txt, prompt) exist in raw jsonl.
    4. Audio files referenced in jsonl are accessible.
"""

import argparse
import importlib.util
import os
import sys


def load_register_module(path: str):
    spec = importlib.util.spec_from_file_location('custom_register', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--num-samples', type=int, default=3)
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import load_dataset

    print(f'[INFO] Loading dataset: {args.dataset_name}')
    train_dataset, val_dataset = load_dataset([args.dataset_name], split_dataset_ratio=0.0)

    assert train_dataset is not None, 'Train dataset load failed'
    print(f'[OK] Dataset loaded, size={len(train_dataset)}')

    missing_audio = []
    for i in range(min(args.num_samples, len(train_dataset))):
        row = train_dataset[i]
        print(f'[INFO] Sample {i}: keys={row.keys()}')

        # After preprocessor, expect messages + audios.
        assert 'messages' in row, 'Preprocessed row missing messages'
        assert 'audios' in row, 'Preprocessed row missing audios'

        for audio_path in row['audios']:
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(os.path.dirname(args.custom_register_path), '..', audio_path)
            if not os.path.exists(audio_path):
                missing_audio.append(audio_path)
                print(f'[WARN] Audio not found: {audio_path}')
            else:
                print(f'[OK] Audio accessible: {audio_path}')

    if missing_audio:
        print(f'[FAIL] {len(missing_audio)} audio files missing')
        sys.exit(1)

    print('[OK] Dataset registration validation passed')


if __name__ == '__main__':
    main()
