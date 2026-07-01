#!/usr/bin/env python3
"""Run integration test validators in order.

Core validators are model-agnostic. Model-family-specific validators live under
``validators/model_specific/{model_family}/`` and are discovered dynamically.

Usage:
    python executors/run_integration_tests.py \
        --custom-register-path outputs/{run_id}/custom/xxx.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --model-family kimi_audio \
        --dataset-name combined_asr_aishell_1 \
        --output outputs/{run_id}/integration_test_report.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys


# Core validators run for every model family, in order.
CORE_VALIDATORS = [
    ('forward_pass', 'validators/core/validate_forward_pass.py'),
    ('loss_computation', 'validators/core/validate_loss_computation.py'),
    ('single_step_training', 'validators/core/validate_single_step_training.py'),
    ('checkpoint_save_load', 'validators/core/validate_checkpoint_save_load.py'),
    ('inference', 'validators/core/validate_inference.py'),
    ('freeze_unfreeze', 'validators/core/validate_freeze_unfreeze.py'),
]


def discover_model_specific_validators(model_family: str):
    """Discover extra validators for a model family, sorted alphabetically."""
    pattern = os.path.join(
        'validators', 'model_specific', model_family, 'validate_*.py'
    )
    scripts = sorted(glob.glob(pattern))
    validators = []
    for script in scripts:
        name = os.path.splitext(os.path.basename(script))[0]
        validators.append((name, script))
    return validators


def run_validator(name: str, script: str, common_args: list, extra_args: list,
                  timeout: int = 300):
    """Run a single validator and return a result dict."""
    cmd = ['python', script] + common_args + extra_args
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    passed = proc.returncode == 0
    return {
        'name': name,
        'script': script,
        'passed': passed,
        'stdout': proc.stdout,
        'stderr': proc.stderr,
    }


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--model-family', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    common = [
        '--custom-register-path', args.custom_register_path,
        '--model', args.model,
        '--model-type', args.model_type,
        '--dataset-name', args.dataset_name,
    ]

    extra_by_validator = {
        'inference': ['--max-new-tokens', str(args.max_new_tokens)],
        'single_step_training': ['--lr', str(args.lr)],
    }

    results = []
    all_passed = True

    # 1. Run core validators.
    for name, script in CORE_VALIDATORS:
        if not os.path.exists(script):
            results.append({
                'name': name,
                'script': script,
                'passed': False,
                'stdout': '',
                'stderr': f'Validator script not found: {script}',
            })
            all_passed = False
            break

        extra = extra_by_validator.get(name, [])
        result = run_validator(name, script, common, extra)
        results.append(result)
        if not result['passed']:
            all_passed = False
            break

    # 2. Run model-family-specific validators if core passed.
    if all_passed:
        specific = discover_model_specific_validators(args.model_family)
        for name, script in specific:
            result = run_validator(name, script, common, [])
            results.append(result)
            if not result['passed']:
                all_passed = False
                break

    report = {
        'passed': all_passed,
        'model_family': args.model_family,
        'results': results,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    run()
