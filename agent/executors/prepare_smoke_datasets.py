#!/usr/bin/env python3
"""Prepare smoke-test datasets from a source jsonl.

Usage:
    python prepare_smoke_datasets.py \
        --dataset data/combined_asr_aishell-1.jsonl \
        --output-dir outputs/{run_id}/smoke_test \
        --overfit-index 0 \
        --overfit-copies 100 \
        --mini-size 100

Outputs:
    {output_dir}/overfit1.jsonl
    {output_dir}/mini100.jsonl
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True, help='Path to source jsonl dataset')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--overfit-index', type=int, default=0)
    parser.add_argument('--overfit-copies', type=int, default=100)
    parser.add_argument('--mini-size', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    if len(rows) == 0:
        raise ValueError(f'Empty dataset: {args.dataset_path}')

    overfit_row = rows[args.overfit_index]
    overfit_path = os.path.join(args.output_dir, 'overfit1.jsonl')
    with open(overfit_path, 'w', encoding='utf-8') as f:
        for _ in range(args.overfit_copies):
            f.write(json.dumps(overfit_row, ensure_ascii=False) + '\n')
    print(f'[OK] Wrote {args.overfit_copies} copies to {overfit_path}')

    mini_size = min(args.mini_size, len(rows))
    mini_rows = rows[:mini_size]
    mini_path = os.path.join(args.output_dir, 'mini100.jsonl')
    with open(mini_path, 'w', encoding='utf-8') as f:
        for row in mini_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'[OK] Wrote {mini_size} samples to {mini_path}')


if __name__ == '__main__':
    main()
