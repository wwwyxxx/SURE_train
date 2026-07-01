#!/usr/bin/env python3
"""Batch download model weights according to a download plan.

Before downloading, checks SURE_train/model/ for existing local weights.

Usage:
    python executors/batch_download.py \
        --plan outputs/{run_id}/download_plan.json \
        --sure-train-dir SURE_train \
        --output outputs/{run_id}/download_report.json
"""

import argparse
import json
import os
import subprocess
import sys


def download_one(model_id: str, local_dir: str, use_hf: bool = False) -> tuple:
    """Return (success, message)."""
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'download_modelscope.py'),
        '--model-id', model_id,
        '--local-dir', local_dir,
        '--output', '/tmp/single_download_report.json',
    ]
    if use_hf:
        cmd.append('--use-hf')

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 and not use_hf:
        # Retry with HF
        cmd.append('--use-hf')
        result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        with open('/tmp/single_download_report.json', 'r') as f:
            report = json.load(f)
        return report['passed'], report.get('report', '')
    except Exception:
        return False, result.stderr or result.stdout


def check_local_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    markers = ['config.json', 'pytorch_model.bin', 'model.safetensors']
    return any(os.path.exists(os.path.join(path, m)) for m in markers)


def find_local_model(model_id: str, sure_train_dir: str) -> str:
    """Use find_local_model.py to check SURE_train/model/."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'find_local_model.py'),
        '--model-id', model_id,
        '--sure-train-dir', sure_train_dir,
        '--output', '/tmp/find_local_model_report.json',
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    try:
        with open('/tmp/find_local_model_report.json', 'r') as f:
            report = json.load(f)
        if report.get('found'):
            return report['local_dir']
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--sure-train-dir', default='SURE_train')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    with open(args.plan, 'r') as f:
        plan = json.load(f)

    results = []
    for item in plan['downloads']:
        component = item['component']
        model_id = item.get('model_id')
        local_dir = item.get('local_dir')

        # Skip random init or copy_from sources.
        if not model_id or model_id.lower() in ('random', 'copy_from_lm_head'):
            results.append({
                'component': component,
                'model_id': model_id,
                'local_dir': local_dir,
                'passed': True,
                'report': 'No download needed',
            })
            continue

        # 1. Check if already exists at requested local_dir.
        if local_dir and check_local_dir(local_dir):
            results.append({
                'component': component,
                'model_id': model_id,
                'local_dir': local_dir,
                'passed': True,
                'report': 'Already exists at requested local_dir',
            })
            continue

        # 2. Check SURE_train/model/ for existing weights.
        found_local = find_local_model(model_id, args.sure_train_dir)
        if found_local:
            results.append({
                'component': component,
                'model_id': model_id,
                'local_dir': found_local,
                'passed': True,
                'report': f'Found existing weights at {found_local}',
            })
            continue

        # 3. Download from modelscope/huggingface.
        if not local_dir:
            # Default to /workspace/model/<last-part>
            local_dir = os.path.join('/workspace/model', model_id.split('/')[-1])
        success, message = download_one(model_id, local_dir)
        results.append({
            'component': component,
            'model_id': model_id,
            'local_dir': local_dir,
            'passed': success,
            'report': message,
        })

    missing = [r for r in results if not r['passed']]
    report = {
        'passed': len(missing) == 0,
        'downloads': results,
        'missing': missing,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
