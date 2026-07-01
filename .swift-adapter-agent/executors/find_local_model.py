#!/usr/bin/env python3
"""Find matching local model directory under SURE_train/model/.

Usage:
    python executors/find_local_model.py \
        --model-id qwen/Qwen2.5-7B \
        --sure-train-dir /abs/path/to/SURE_train \
        --output outputs/{run_id}/find_local_model.json
"""

import argparse
import json
import os


def is_valid_model_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    markers = ['config.json', 'pytorch_model.bin', 'model.safetensors']
    return any(os.path.exists(os.path.join(path, m)) for m in markers)


def resolve_sure_train_dir(sure_train_dir: str) -> str:
    """Resolve SURE_train directory, allowing relative paths from project root."""
    # Try direct path first.
    if os.path.isdir(os.path.join(sure_train_dir, 'model')):
        return sure_train_dir

    # Try parent directory (if running from .swift-adapter-agent/).
    parent = os.path.join('..', sure_train_dir)
    if os.path.isdir(os.path.join(parent, 'model')):
        return os.path.abspath(parent)

    # Search upwards for a directory matching sure_train_dir with model/ subdir.
    current = os.path.abspath('.')
    for _ in range(5):
        candidate = os.path.join(current, sure_train_dir)
        if os.path.isdir(os.path.join(candidate, 'model')):
            return candidate
        parent_dir = os.path.dirname(current)
        if parent_dir == current:
            break
        current = parent_dir

    return sure_train_dir


def find_local_model(model_id: str, sure_train_dir: str) -> str:
    sure_train_dir = resolve_sure_train_dir(sure_train_dir)
    model_root = os.path.join(sure_train_dir, 'model')
    if not os.path.isdir(model_root):
        return None

    # Try exact model_id last part.
    candidates = []
    if '/' in model_id:
        candidates.append(model_id.split('/')[-1])
    candidates.append(model_id)

    # Also try normalized names.
    normalized = model_id.replace('_', '-').replace('/', '-').lower()
    candidates.append(normalized)

    # Search subdirectories.
    for entry in os.listdir(model_root):
        entry_path = os.path.join(model_root, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry in candidates or entry.lower() in [c.lower() for c in candidates]:
            if is_valid_model_dir(entry_path):
                return entry_path

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--sure-train-dir', default='SURE_train')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    local_dir = find_local_model(args.model_id, args.sure_train_dir)
    result = {
        'found': local_dir is not None,
        'model_id': args.model_id,
        'local_dir': local_dir,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
