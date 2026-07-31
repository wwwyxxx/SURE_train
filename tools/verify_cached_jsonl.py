#!/usr/bin/env python3
"""Verify a cached jsonl: line count and that every wav points to an existing .pt file."""
import argparse
import json
import sys
from pathlib import Path


def verify(jsonl_path: str, expected_lines: int | None = None):
    path = Path(jsonl_path)
    if not path.exists():
        print(f'ERROR: {jsonl_path} does not exist', file=sys.stderr)
        return 1

    missing = []
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            row = json.loads(line)
            wav = row.get('wav') or row.get('audio') or row.get('audio_path')
            if not wav:
                print(f'ERROR line {total}: no wav/audio field', file=sys.stderr)
                missing.append((total, wav))
                continue
            if not Path(wav).exists():
                missing.append((total, wav))
    print(f'Total lines: {total}')
    if expected_lines is not None:
        print(f'Expected lines: {expected_lines}')
        if total != expected_lines:
            print(f'ERROR: line count mismatch ({total} != {expected_lines})', file=sys.stderr)
            return 1
    print(f'Missing .pt files: {len(missing)}')
    if missing:
        print('First 10 missing:')
        for idx, wav in missing[:10]:
            print(f'  line {idx}: {wav}')
        return 1
    print('OK: all wav paths exist')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--expected-lines', type=int, default=None)
    args = parser.parse_args()
    sys.exit(verify(args.jsonl, args.expected_lines))
