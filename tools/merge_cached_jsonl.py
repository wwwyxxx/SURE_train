#!/usr/bin/env python3
"""Merge multiple cached jsonl files into one and optionally validate .pt existence."""
import argparse
import json
import sys
from pathlib import Path


def merge(input_jsonls: list[str], output_jsonl: str, validate: bool = True):
    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    missing = 0
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for inp in input_jsonls:
            inp_path = Path(inp)
            if not inp_path.exists():
                print(f'ERROR: input {inp} does not exist', file=sys.stderr)
                return 1
            with open(inp_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    row = json.loads(line)
                    wav = row.get('wav') or row.get('audio') or row.get('audio_path')
                    if validate and wav and not Path(wav).exists():
                        missing += 1
                        if missing <= 10:
                            print(f'WARN: missing .pt at line {total + 1}: {wav}', file=sys.stderr)
                    out_f.write(json.dumps(row, ensure_ascii=False) + '\n')
                    total += 1
    print(f'Merged {len(input_jsonls)} files into {output_jsonl}')
    print(f'Total lines: {total}')
    if validate:
        print(f'Missing .pt files: {missing}')
    return 0 if missing == 0 else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True, help='Input cached jsonls in desired order')
    parser.add_argument('--output', required=True, help='Output merged cached jsonl')
    parser.add_argument('--no-validate', action='store_true', help='Skip .pt existence check')
    args = parser.parse_args()
    sys.exit(merge(args.inputs, args.output, validate=not args.no_validate))
