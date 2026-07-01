#!/usr/bin/env python3
"""Launch 8 independent single-GPU batched preprocessors in parallel.

This avoids multiprocessing CUDA issues by running each GPU in its own docker
container. After all finish, it merges the per-split cached jsonls into one.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _split_jsonl(input_path: str, output_dir: Path, n_splits: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as f:
        rows = [line for line in f]

    split_size = (len(rows) + n_splits - 1) // n_splits
    split_paths = []
    for i in range(n_splits):
        start = i * split_size
        end = min((i + 1) * split_size, len(rows))
        split_path = output_dir / f'split_{i}.jsonl'
        with open(split_path, 'w', encoding='utf-8') as f:
            f.writelines(rows[start:end])
        split_paths.append((i, str(split_path)))
    return split_paths


def _build_command(gpu_id: int, split_path: str, output_jsonl: str, cache_dir: str,
                   batch_size: int, skip_existing: bool) -> str:
    skip_flag = ' --skip-existing' if skip_existing else ''
    cmd = (
        f'docker run --rm --gpus \"\\"device={gpu_id}\\"\" --ipc=host --shm-size=64g '
        f'-v /aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train:/workspace '
        f'-w /workspace '
        f'docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-mimoaudio:v0 '
        f'bash -c "cd /workspace && /opt/conda/bin/python tools/precompute_mimo_audio_tokens_batched.py '
        f'--dataset {split_path} '
        f'--output-jsonl {output_jsonl} '
        f'--cache-dir {cache_dir} '
        f'--gpu 0 '
        f'--batch-size {batch_size}'
        f'{skip_flag} \"'
    )
    return cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-gpus', type=int, default=8)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--split-dir', default='data/audio_tokens_cache/splits')
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    print(f'Splitting {args.dataset} into {args.n_gpus} parts...')
    split_paths = _split_jsonl(args.dataset, split_dir, args.n_gpus)

    processes = []
    output_paths = []
    for gpu_id, split_path in split_paths:
        output_split = str(Path(args.cache_dir) / f'cached_split_{gpu_id}.jsonl')
        output_paths.append(output_split)
        cmd = _build_command(gpu_id, split_path, output_split, args.cache_dir,
                             args.batch_size, args.skip_existing)
        print(f'[GPU {gpu_id}] {cmd}')
        p = subprocess.Popen(cmd, shell=True)
        processes.append(p)

    print(f'Waiting for {len(processes)} GPU processes to finish...')
    for p in processes:
        p.wait()

    # Merge outputs
    print(f'Merging outputs into {args.output_jsonl}...')
    with open(args.output_jsonl, 'w', encoding='utf-8') as out_f:
        for output_split in output_paths:
            with open(output_split, 'r', encoding='utf-8') as f:
                out_f.writelines(f)

    print('Done.')


if __name__ == '__main__':
    main()
