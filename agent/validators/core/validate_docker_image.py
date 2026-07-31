#!/usr/bin/env python3
"""Validate docker image can run and import required packages.

Usage:
    python validate_docker_image.py --image swift-adapter:latest
"""

import argparse
import subprocess
import sys


def run(image: str, gpus: str = 'all'):
    cmd = [
        'docker', 'run', '--rm', '--gpus', gpus,
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
    parser.add_argument('--gpus', default='all',
                        help='GPU devices to pass to docker run --gpus (default: all)')
    args = parser.parse_args()
    run(args.image, args.gpus)
