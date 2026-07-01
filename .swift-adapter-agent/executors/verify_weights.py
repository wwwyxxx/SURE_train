#!/usr/bin/env python3
"""Verify required model weights exist locally.

Usage:
    python executors/verify_weights.py \
        --input outputs/{run_id}/download_plan.json \
        --output outputs/{run_id}/verify_weights_report.json
"""

import argparse
import json
import os


def check_local_dir(path: str) -> bool:
    """Check if a local model dir looks valid (has config.json or pytorch bin)."""
    if not os.path.isdir(path):
        return False
    markers = ['config.json', 'pytorch_model.bin', 'model.safetensors']
    return any(os.path.exists(os.path.join(path, m)) for m in markers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to download_plan.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        plan = json.load(f)

    results = []
    missing = []
    for item in plan['downloads']:
        local_dir = item['local_dir']
        exists = check_local_dir(local_dir)
        results.append({
            'component': item['component'],
            'model_id': item.get('model_id'),
            'local_dir': local_dir,
            'exists': exists,
        })
        if not exists:
            missing.append(item)

    report = {
        'all_exist': len(missing) == 0,
        'results': results,
        'missing': missing,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
