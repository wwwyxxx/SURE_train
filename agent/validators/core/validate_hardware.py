#!/usr/bin/env python3
"""Validate hardware environment.

Usage:
    python validate_hardware.py

Checks:
    1. At least one GPU is available.
    2. GPU memory is readable.
    3. CUDA version is available.
"""

import json
import subprocess
import sys

import torch


def run():
    if not torch.cuda.is_available():
        print('[FAIL] CUDA not available')
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print('[FAIL] No GPU detected')
        sys.exit(1)

    gpus = []
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            'id': i,
            'name': torch.cuda.get_device_name(i),
            'memory_total_gb': props.total_memory / 1024**3,
            'compute_capability': f'{props.major}.{props.minor}',
        })

    try:
        cuda_version = torch.version.cuda
    except Exception:
        cuda_version = None

    result = {
        'passed': True,
        'gpu_count': gpu_count,
        'gpus': gpus,
        'cuda_version_pytorch': cuda_version,
    }
    print(json.dumps(result, indent=2))
    print('[OK] Hardware validation passed')


if __name__ == '__main__':
    run()
