#!/usr/bin/env python3
"""Download model weights from ModelScope.

Usage:
    python executors/download_modelscope.py \
        --model-id qwen/Qwen2.5-7B \
        --local-dir /workspace/model/Qwen2.5-7B \
        --output outputs/{run_id}/download_report.json

Also supports HuggingFace hub if modelscope fails:
    --use-hf
"""

import argparse
import json
import os
import sys


def download_from_modelscope(model_id: str, local_dir: str):
    try:
        from modelscope import snapshot_download
        snapshot_download(model_id, cache_dir=os.path.dirname(local_dir), local_dir=local_dir)
        return True, f"Downloaded {model_id} to {local_dir}"
    except Exception as e:
        return False, str(e)


def download_from_hf(model_id: str, local_dir: str):
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=local_dir)
        return True, f"Downloaded {model_id} to {local_dir}"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--local-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--use-hf', action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.local_dir), exist_ok=True)

    if args.use_hf:
        success, message = download_from_hf(args.model_id, args.local_dir)
    else:
        success, message = download_from_modelscope(args.model_id, args.local_dir)
        if not success:
            print(f'[WARN] ModelScope failed: {message}. Trying HF...')
            success, message = download_from_hf(args.model_id, args.local_dir)

    result = {
        'passed': success,
        'model_id': args.model_id,
        'local_dir': args.local_dir,
        'report': message,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
