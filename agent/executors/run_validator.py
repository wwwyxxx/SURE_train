#!/usr/bin/env python3
"""Run a validator script and output structured JSON report.

Usage:
    python executors/run_validator.py \
        --validator validators/core/validate_dataset_registration.py \
        --custom-register-path outputs/{run_id}/custom/xxx.py \
        --dataset-name xxx \
        --output validation_result.json

Output JSON:
    {
        "passed": true,
        "validator": "...",
        "report": "...",
        "metrics": {}
    }
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validator', required=True, help='Path to validator script')
    parser.add_argument('--output', required=True, help='Output JSON report path')
    parser.add_argument('--timeout', type=int, default=600)
    # Forward remaining args to validator.
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.validator):
        result = {
            'passed': False,
            'validator': args.validator,
            'report': f'Validator not found: {args.validator}',
            'metrics': {}
        }
        save_and_exit(args.output, result, 1)

    cmd = ['python', args.validator] + unknown
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        result = {
            'passed': False,
            'validator': args.validator,
            'report': f'Validator timed out after {args.timeout}s',
            'stdout': '',
            'stderr': '',
            'metrics': {}
        }
        save_and_exit(args.output, result, 1)

    passed = proc.returncode == 0
    result = {
        'passed': passed,
        'validator': args.validator,
        'report': proc.stdout[-2000:] if proc.stdout else '',
        'stdout': proc.stdout,
        'stderr': proc.stderr,
        'metrics': {}
    }
    save_and_exit(args.output, result, 0 if passed else 1)


def save_and_exit(path: str, result: dict, exit_code: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
