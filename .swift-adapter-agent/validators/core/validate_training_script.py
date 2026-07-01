#!/usr/bin/env python3
"""Validate generated training script.

Usage:
    python validate_training_script.py --script run_xxx.sh
"""

import argparse
import os
import re
import sys


def run(script_path: str):
    assert os.path.exists(script_path), f'Script not found: {script_path}'

    with open(script_path, 'r') as f:
        content = f.read()

    required = [
        'swift sft',
        '--custom_register_path',
        '--model',
        '--model_type',
        '--dataset',
        '--output_dir',
    ]
    missing = [r for r in required if r not in content]
    if missing:
        print(f'[FAIL] Missing required arguments: {missing}')
        sys.exit(1)

    # Basic shell syntax check.
    import subprocess
    result = subprocess.run(['bash', '-n', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[FAIL] Shell syntax error:\n{result.stderr}')
        sys.exit(1)

    print('[OK] Training script validation passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', required=True)
    args = parser.parse_args()
    run(args.script)
