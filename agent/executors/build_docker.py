#!/usr/bin/env python3
"""Build docker image from resolved Dockerfile.

Usage:
    python executors/build_docker.py \
        --dockerfile SURE_train/Dockerfile/kimi-audio_dockerfile/Dockerfile \
        --image-name docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0 \
        --output outputs/{run_id}/docker_build_report.json
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dockerfile', required=True)
    parser.add_argument('--image-name', required=True)
    parser.add_argument('--build-context', default='SURE_train')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.dockerfile):
        result = {
            'passed': False,
            'image_name': args.image_name,
            'report': f'Dockerfile not found: {args.dockerfile}',
        }
        save_and_exit(args.output, result, 1)

    build_cmd = [
        'docker', 'build',
        '-t', args.image_name,
        '-f', args.dockerfile,
        args.build_context,
    ]

    print(f'[INFO] Building image {args.image_name} ...')
    print(f'[INFO] Dockerfile: {args.dockerfile}')
    print(f'[INFO] Context: {args.build_context}')

    proc = subprocess.run(build_cmd)

    if proc.returncode != 0:
        result = {
            'passed': False,
            'image_name': args.image_name,
            'report': 'docker build failed',
        }
        save_and_exit(args.output, result, 1)

    result = {
        'passed': True,
        'image_name': args.image_name,
        'report': f'Successfully built {args.image_name}',
    }
    save_and_exit(args.output, result, 0)


def save_and_exit(path: str, result: dict, exit_code: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
