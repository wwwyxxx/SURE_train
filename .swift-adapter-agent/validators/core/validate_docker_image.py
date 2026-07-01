#!/usr/bin/env python3
"""Validate docker image can run and import required packages.

Usage:
    python validate_docker_image.py --image swift-adapter:latest
"""

import argparse
import subprocess
import sys


def run(image: str):
    cmd = [
        'docker', 'run', '--rm', '--gpus', 'all',
        image,
        'python', '-c',
        'import torch; import swift; import transformers; print("OK")'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'[FAIL] Docker import test failed:\n{result.stderr}')
        sys.exit(1)
    print(f'[OK] Docker image {image} validated')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    run(args.image)
