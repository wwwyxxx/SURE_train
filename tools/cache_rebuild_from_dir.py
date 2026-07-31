#!/usr/bin/env python3
"""Rebuild a cached jsonl from an original jsonl and a directory of .pt token caches.

The cache file name is derived from the absolute path of the original wav using the
same sha256 prefix scheme as precompute_mimo_audio_tokens.py.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _cache_key(wav_path: str) -> str:
    return hashlib.sha256(wav_path.encode('utf-8')).hexdigest()[:32]


def rebuild(input_jsonl: str, cache_dir: str, output_jsonl: str):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    missing = []
    total = 0
    with open(input_jsonl, 'r', encoding='utf-8') as in_f, \
         open(output_jsonl, 'w', encoding='utf-8') as out_f:
        for line in in_f:
            total += 1
            row = json.loads(line)
            wav = row.get('wav') or row.get('audio') or row.get('audio_path')
            if wav is None:
                print(f'ERROR line {total}: no wav/audio field', file=sys.stderr)
                missing.append((total, None))
                out_f.write(json.dumps(row, ensure_ascii=False) + '\n')
                continue

            wav_abs = wav if os.path.isabs(wav) else str(Path(input_jsonl).resolve().parent.parent / wav)
            key = _cache_key(wav_abs)
            pt_file = cache_path / f'{key}.pt'
            if not pt_file.exists():
                missing.append((total, str(pt_file)))
            new_row = dict(row)
            new_row['wav'] = str(pt_file)
            out_f.write(json.dumps(new_row, ensure_ascii=False) + '\n')

    print(f'Rebuilt {total} rows -> {output_jsonl}')
    print(f'Missing cache files: {len(missing)}')
    if missing:
        print('First 10 missing:')
        for idx, pt in missing[:10]:
            print(f'  line {idx}: {pt}')
        return 1
    print('OK: all cache files present')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-jsonl', required=True, help='Original jsonl with wav paths')
    parser.add_argument('--cache-dir', required=True, help='Directory containing .pt cache files')
    parser.add_argument('--output-jsonl', required=True, help='Output cached jsonl')
    args = parser.parse_args()
    sys.exit(rebuild(args.input_jsonl, args.cache_dir, args.output_jsonl))
