#!/usr/bin/env python3
"""Probe hardware and output JSON, respecting max_gpus limit.

Usage:
    python executors/hardware_probe.py --max-gpus 7 --output outputs/{run_id}/hardware.json
"""

import argparse
import json
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-gpus', type=int, default=7, help='Maximum GPUs to use')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': round(props.total_memory / 1024**3, 2),
                'compute_capability': f'{props.major}.{props.minor}',
            })

    # Respect max_gpus limit.
    usable_gpus = gpus[:args.max_gpus]

    try:
        cuda_version = torch.version.cuda
    except Exception:
        cuda_version = None

    result = {
        'passed': len(usable_gpus) > 0,
        'gpu_count': len(gpus),
        'max_gpus': args.max_gpus,
        'usable_gpu_count': len(usable_gpus),
        'usable_gpus': usable_gpus,
        'cuda_version_pytorch': cuda_version,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
