#!/usr/bin/env python3
"""Resolve docker image for a model family using three-tier lookup.

Naming convention:
  Image: docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-<model>:<version>
  Dockerfile dir: SURE_train/Dockerfile/<model-family-with-hyphens>_dockerfile/

Usage:
    python executors/resolve_docker_image.py \
        --model-family kimi_audio \
        --version v0 \
        --sure-train-dir SURE_train \
        --output outputs/{run_id}/docker_resolution.json

Output JSON:
    {
      "action": "use_existing" | "build_existing" | "build_new",
      "image_name": "docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0",
      "dockerfile_path": "SURE_train/Dockerfile/kimi-audio_dockerfile/Dockerfile",
      "exists_locally": true,
      "dockerfile_exists": true
    }
"""

import argparse
import json
import os
import subprocess
import sys


def normalize_image_model_name(model_family: str) -> str:
    """kimi_audio -> kimiaudio"""
    return model_family.replace('_', '').replace('-', '').lower()


def normalize_dockerfile_dir_name(model_family: str) -> str:
    """kimi_audio -> kimi-audio_dockerfile"""
    return model_family.replace('_', '-') + '_dockerfile'


def list_local_images() -> list:
    try:
        result = subprocess.run(
            ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
            capture_output=True, text=True, check=True
        )
        return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    except Exception as e:
        print(f'[WARN] Failed to list docker images: {e}', file=sys.stderr)
        return []


def find_existing_image(model_family: str, version: str) -> str:
    """Return matching image name or None."""
    model = normalize_image_model_name(model_family)
    prefix = f'docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-{model}:'
    candidates = [img for img in list_local_images() if img.startswith(prefix)]
    if not candidates:
        return None
    # Prefer exact version match.
    exact = f'{prefix}{version}'
    if exact in candidates:
        return exact
    # Otherwise return first available.
    return candidates[0]


def find_existing_dockerfile(model_family: str, sure_train_dir: str) -> str:
    """Return Dockerfile path or None."""
    dir_name = normalize_dockerfile_dir_name(model_family)
    path = os.path.join(sure_train_dir, 'Dockerfile', dir_name, 'Dockerfile')
    if os.path.isfile(path):
        return path
    return None


def resolve_sure_train_dir(sure_train_dir: str) -> str:
    """Resolve SURE_train directory, allowing relative paths from project root."""
    # Try direct path first.
    if os.path.isdir(os.path.join(sure_train_dir, 'model')):
        return sure_train_dir

    # Try parent directory (if running from .swift-adapter-agent/).
    parent = os.path.join('..', sure_train_dir)
    if os.path.isdir(os.path.join(parent, 'model')):
        return os.path.abspath(parent)

    # Search upwards for a directory with model/ subdir.
    current = os.path.abspath('.')
    for _ in range(5):
        if os.path.isdir(os.path.join(current, 'model')):
            return current
        candidate = os.path.join(current, sure_train_dir)
        if os.path.isdir(os.path.join(candidate, 'model')):
            return candidate
        parent_dir = os.path.dirname(current)
        if parent_dir == current:
            break
        current = parent_dir

    return sure_train_dir


def generate_dockerfile(model_family: str, sure_train_dir: str) -> str:
    """Generate a minimal Dockerfile following project conventions.

    NOTE: This creates files on disk. Callers must explicitly opt-in via
    --create-if-missing.
    """
    dir_name = normalize_dockerfile_dir_name(model_family)
    dockerfile_dir = os.path.join(sure_train_dir, 'Dockerfile', dir_name)
    os.makedirs(dockerfile_dir, exist_ok=True)
    path = os.path.join(dockerfile_dir, 'Dockerfile')

    # Derive model package name from model_family.
    package_name = model_family.replace('_', '-')

    dockerfile = f'''FROM docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl build-essential ninja-build cmake pkg-config \\
    ffmpeg sox libsndfile1 libgl1-mesa-glx openssh-server ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY {package_name} /workspace/{package_name}
COPY ms-swift /workspace/ms-swift

WORKDIR /workspace/{package_name}
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace
CMD ["/bin/bash"]
'''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(dockerfile)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-family', required=True)
    parser.add_argument('--version', default='v0')
    parser.add_argument('--sure-train-dir', default='SURE_train')
    parser.add_argument('--output', required=True)
    parser.add_argument('--create-if-missing', action='store_true',
                        help='Generate a Dockerfile when no image or Dockerfile exists')
    args = parser.parse_args()

    sure_train_dir = resolve_sure_train_dir(args.sure_train_dir)
    image_name = find_existing_image(args.model_family, args.version)
    dockerfile_path = find_existing_dockerfile(args.model_family, sure_train_dir)

    created = False
    if image_name:
        action = 'use_existing'
    elif dockerfile_path:
        action = 'build_existing'
        model = normalize_image_model_name(args.model_family)
        image_name = f'docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-{model}:{args.version}'
    elif args.create_if_missing:
        action = 'build_new'
        dockerfile_path = generate_dockerfile(args.model_family, sure_train_dir)
        created = True
        model = normalize_image_model_name(args.model_family)
        image_name = f'docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-{model}:{args.version}'
    else:
        action = 'build_new'
        model = normalize_image_model_name(args.model_family)
        image_name = f'docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-{model}:{args.version}'

    result = {
        'action': action,
        'image_name': image_name,
        'dockerfile_path': dockerfile_path,
        'exists_locally': action == 'use_existing',
        'dockerfile_exists': dockerfile_path is not None and os.path.isfile(dockerfile_path),
        'created': created,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
