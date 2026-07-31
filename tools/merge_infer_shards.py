#!/usr/bin/env python3
"""Merge per-shard inference outputs into one jsonl per dataset (sorted by idx)."""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', default='output/run1/predictions')
    ap.add_argument('--names', nargs='*', default=[
        'aishell1-test_ASR_infer',
        'librispeech_test-clean_ASR',
        'librispeech_test-other_ASR',
    ])
    ap.add_argument('--keep-shards', action='store_true')
    args = ap.parse_args()

    for name in args.names:
        shards = sorted(glob.glob(os.path.join(args.pred_dir, f'{name}.shard*.jsonl')))
        if not shards:
            print(f'SKIP {name}: no shard files found')
            continue
        seen = {}
        for shard in shards:
            with open(shard, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    seen[rec['idx']] = rec
        merged = [seen[i] for i in sorted(seen)]
        out_path = os.path.join(args.pred_dir, f'{name}.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'{name}: merged {len(merged)} samples from {len(shards)} shards -> {out_path}')
        if not args.keep_shards:
            for shard in shards:
                os.remove(shard)


if __name__ == '__main__':
    main()
