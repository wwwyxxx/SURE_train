#!/usr/bin/env python3
"""Validate Kimi-Audio can handle different audio sequence lengths.

Usage:
    python validate_sequence_length.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1 \
        --durations 5 10 20 30 60 120

Checks:
    1. Template can encode audio of given duration.
    2. Sequence length is within max_length.
    3. Forward pass does not OOM for each duration.
"""

import argparse
import importlib.util
import sys

import librosa
import numpy as np
import torch


def load_register_module(path: str):
    spec = importlib.util.spec_from_file_location('custom_register', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def make_dummy_audio(duration_sec: int, sr: int = 16000):
    t = np.linspace(0, duration_sec, int(duration_sec * sr))
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--durations', nargs='+', type=int, default=[5, 10, 20, 30, 60, 120])
    parser.add_argument('--max-length', type=int, default=8192)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template, load_dataset

    model, _ = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )
    model = model.to(args.device)
    model.eval()

    template = get_template(args.model_type, load_model_tokenizer=None)
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)
    base_row = train_dataset[0]

    print(f'{"Duration(s)":>12} | {"Seq Len":>10} | {"Status":>10} | {"Peak Mem(GiB)":>16}')
    print('-' * 60)

    for duration in args.durations:
        wav = make_dummy_audio(duration)
        tmp_path = f'/tmp/dummy_{duration}s.wav'
        import soundfile as sf
        sf.write(tmp_path, wav, 16000)

        row = {
            'messages': base_row['messages'],
            'audios': [tmp_path],
        }

        try:
            encoded = template.encode(row, return_length=True)
            seq_len = encoded['input_ids'].shape[0]

            if seq_len > args.max_length:
                status = 'TOO_LONG'
                peak = 0.0
            else:
                batch = template.data_collator([encoded])
                batch = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in batch.items()}

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(args.device)
                with torch.no_grad():
                    model(**batch)
                peak = torch.cuda.max_memory_allocated(args.device) / 1024**3
                status = 'OK'

            print(f'{duration:>12} | {seq_len:>10} | {status:>10} | {peak:>16.2f}')

        except torch.OutOfMemoryError:
            print(f'{duration:>12} | {"?":>10} | {"OOM":>10} | {0.0:>16.2f}')
        except Exception as e:
            print(f'{duration:>12} | {"?":>10} | {"ERROR":>10} | {str(e)[:20]:>16}')

    print('[OK] Sequence length sweep completed')


if __name__ == '__main__':
    main()
