#!/usr/bin/env python3
"""Smoke test orchestrator for model adapter harness.

By default this script only prepares datasets and generates training/inference
script files. Pass --run to actually execute the smoke tests (long running).

Usage:
    python run_smoke_test.py \
        --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
        --model /workspace/model/MiMo-Audio-7B-Base-merged \
        --model-type mimo_audio \
        --model-family mimo_audio \
        --dataset-name combined_asr_aishell_1 \
        --dataset-path data/combined_asr_aishell-1.jsonl \
        --output-dir outputs/{run_id}/smoke_test \
        --output outputs/{run_id}/smoke_test_report.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path('/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train')
IMAGE = 'docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-mimoaudio:v0'


# Map model_family to the model-specific inference script inside the harness.
# For unknown families, the agent must provide --inference-script.
INFER_SCRIPTS = {
    'mimo_audio': str(ROOT / '.swift-adapter-agent/executors/model_specific/mimo_audio/infer_mimo.py'),
}


def load_register_module(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location('custom_register', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def prepare_datasets(args):
    cmd = [
        'python', str(ROOT / '.swift-adapter-agent/executors/prepare_smoke_datasets.py'),
        '--dataset-path', str(ROOT / args.dataset_path),
        '--output-dir', args.output_dir,
        '--overfit-index', '0',
        '--overfit-copies', '100',
        '--mini-size', '100',
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        return False
    print(proc.stdout)
    return True


def write_script(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(0o755)


def generate_overfit_script(args):
    out_dir = Path(args.output_dir) / 'overfit1'
    script = f"""#!/usr/bin/env bash
set -euo pipefail
rm -rf "{out_dir}"
NPROC_PER_NODE=1 \\
CUDA_VISIBLE_DEVICES=0 \\
swift sft \\
  --custom_register_path {args.custom_register_path} \\
  --model {args.model} \\
  --model_type {args.model_type} \\
  --dataset {args.model_family}_smoke_overfit1 \\
  --train_type full \\
  --split_dataset_ratio 0 \\
  --per_device_train_batch_size 4 \\
  --gradient_accumulation_steps 1 \\
  --num_train_epochs 2 \\
  --learning_rate 1e-4 \\
  --lr_scheduler_type constant \\
  --warmup_ratio 0 \\
  --max_grad_norm 1.0 \\
  --bf16 true \\
  --gradient_checkpointing true \\
  --max_length 1024 \\
  --logging_steps 1 \\
  --save_steps 1000 \\
  --save_total_limit 1 \\
  --save_only_model true \\
  --output_dir {out_dir} \\
  --report_to none
"""
    write_script(Path(args.output_dir) / 'run_overfit1.sh', script)
    return out_dir


def generate_mini_script(args, bs: int = 8):
    out_dir = Path(args.output_dir) / 'mini100'
    script = f"""#!/usr/bin/env bash
set -euo pipefail
rm -rf "{out_dir}"
NPROC_PER_NODE=7 \\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \\
swift sft \\
  --custom_register_path {args.custom_register_path} \\
  --model {args.model} \\
  --model_type {args.model_type} \\
  --dataset {args.model_family}_smoke_mini100 \\
  --train_type full \\
  --split_dataset_ratio 0 \\
  --per_device_train_batch_size {bs} \\
  --gradient_accumulation_steps 4 \\
  --num_train_epochs 20 \\
  --learning_rate 1e-4 \\
  --lr_scheduler_type constant \\
  --warmup_ratio 0 \\
  --max_grad_norm 1.0 \\
  --bf16 true \\
  --gradient_checkpointing true \\
  --max_length 1024 \\
  --logging_steps 10 \\
  --save_steps 1000 \\
  --save_total_limit 1 \\
  --save_only_model true \\
  --output_dir {out_dir} \\
  --report_to none
"""
    write_script(Path(args.output_dir) / 'run_mini100.sh', script)
    return out_dir


def generate_bs_test_script(args, bs: int):
    out_dir = Path(args.output_dir) / f'bs_test_{bs}'
    script = f"""#!/usr/bin/env bash
set -euo pipefail
rm -rf "{out_dir}"
NPROC_PER_NODE=7 \\
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \\
swift sft \\
  --custom_register_path {args.custom_register_path} \\
  --model {args.model} \\
  --model_type {args.model_type} \\
  --dataset {args.batch_size_test_dataset} \\
  --train_type full \\
  --split_dataset_ratio 0 \\
  --per_device_train_batch_size {bs} \\
  --gradient_accumulation_steps 4 \\
  --num_train_epochs 1 \\
  --learning_rate 1e-4 \\
  --lr_scheduler_type constant \\
  --warmup_ratio 0 \\
  --max_grad_norm 1.0 \\
  --bf16 true \\
  --gradient_checkpointing true \\
  --max_length 2048 \\
  --logging_steps 10 \\
  --save_steps 1000 \\
  --save_total_limit 1 \\
  --save_only_model true \\
  --output_dir {out_dir} \\
  --report_to none
"""
    write_script(Path(args.output_dir) / f'run_bs_test_{bs}.sh', script)
    return out_dir


def generate_inference_script(args, checkpoint: str, dataset: str, output: Path, num_samples: int, max_new_tokens: int):
    infer_script = args.inference_script or INFER_SCRIPTS.get(args.model_family)
    if infer_script is None:
        raise ValueError(
            f'No inference script known for model_family={args.model_family}. '
            'Pass --inference-script explicitly.'
        )
    script = f"""#!/usr/bin/env bash
set -euo pipefail
python {infer_script} \\
  --checkpoint {checkpoint} \\
  --dataset {dataset} \\
  --num-samples {num_samples} \\
  --max-new-tokens {max_new_tokens}
"""
    write_script(output, script)


def run_docker_command(cmd: list, log_path: Path, gpu: str = 'all'):
    """Run a command inside the MiMo-Audio docker container."""
    docker_cmd = [
        'docker', 'run', '--rm', '--gpus', f'device={gpu}',
        '--ipc=host', '--shm-size=64g',
        '-v', f'{ROOT}:/workspace',
        '-w', '/workspace',
        IMAGE,
        'bash', '-c', 'cd /workspace && ' + ' '.join(cmd)
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        proc = subprocess.run(docker_cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def parse_log_for_metrics(log_path: Path):
    """Extract final token_acc and loss from ms-swift training log."""
    token_acc = None
    loss = None
    if not log_path.exists():
        return token_acc, loss
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'token_acc' in line:
                try:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if p == 'token_acc' and i + 2 < len(parts):
                            token_acc = float(parts[i + 2])
                except Exception:
                    pass
            if 'loss' in line and 'eval_loss' not in line:
                try:
                    parts = line.strip().split()
                    for i, p in enumerate(parts):
                        if p == 'loss' and i + 2 < len(parts):
                            loss = float(parts[i + 2])
                except Exception:
                    pass
    return token_acc, loss


def find_latest_checkpoint(output_dir: Path):
    """Find the latest checkpoint directory under output_dir."""
    versions = sorted(output_dir.glob('v*'))
    if not versions:
        return None
    latest_version = versions[-1]
    checkpoints = sorted(latest_version.glob('checkpoint-*'))
    if not checkpoints:
        return None
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--model-family', required=True)
    parser.add_argument('--dataset-name', required=True, help='Registered dataset name for normal training')
    parser.add_argument('--dataset-path', required=True, help='Path to source jsonl for smoke datasets')
    parser.add_argument('--batch-size-test-dataset', default='test_audio_30s_x100',
                        help='Registered dataset name for batch size OOM test')
    parser.add_argument('--inference-script', default=None,
                        help='Path to model-specific inference script (optional, defaults by model_family)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', action='store_true', help='Actually execute smoke tests')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepare_datasets(args)

    report = {
        'passed': False,
        'tests': {},
        'notes': [
            'Smoke test datasets prepared.',
            'Training scripts generated.',
            'If --run was not passed, execute the generated scripts manually.',
        ],
    }

    # Test 1: overfit1
    overfit_out = generate_overfit_script(args)
    overfit_infer = out_dir / 'infer_overfit1.sh'
    report['tests']['overfit1'] = {
        'script': str(out_dir / 'run_overfit1.sh'),
        'inference_script': str(overfit_infer),
        'expected_token_acc': 0.90,
        'expected_exact_match': True,
        'output_dir': str(overfit_out),
    }

    # Test 2: mini100
    mini_out = generate_mini_script(args)
    mini_infer = out_dir / 'infer_mini100.sh'
    report['tests']['mini100'] = {
        'script': str(out_dir / 'run_mini100.sh'),
        'inference_script': str(mini_infer),
        'expected_token_acc': 0.65,
        'expected_readable': True,
        'output_dir': str(mini_out),
    }

    # Test 3: batch size
    bs_scripts = {}
    for bs in [8, 6, 4, 2, 1]:
        bs_out = generate_bs_test_script(args, bs)
        bs_scripts[bs] = {
            'script': str(out_dir / f'run_bs_test_{bs}.sh'),
            'output_dir': str(bs_out),
        }
    report['tests']['batch_size'] = {
        'candidates': bs_scripts,
        'max_batch_size': None,
    }

    if args.run:
        # Execute overfit1.
        overfit_log = out_dir / 'overfit1.log'
        rc = run_docker_command(['bash', str(out_dir / 'run_overfit1.sh')], overfit_log, gpu='0')
        token_acc, loss = parse_log_for_metrics(overfit_log)
        report['tests']['overfit1']['returncode'] = rc
        report['tests']['overfit1']['token_acc'] = token_acc
        report['tests']['overfit1']['final_loss'] = loss

        ckpt = find_latest_checkpoint(overfit_out)
        if ckpt:
            generate_inference_script(
                args,
                checkpoint=str(ckpt),
                dataset=str(out_dir / 'overfit1.jsonl'),
                output=overfit_infer,
                num_samples=1,
                max_new_tokens=128,
            )

        # Execute mini100.
        mini_log = out_dir / 'mini100.log'
        rc = run_docker_command(['bash', str(out_dir / 'run_mini100.sh')], mini_log, gpu='0,1,2,3,4,5,6')
        token_acc, loss = parse_log_for_metrics(mini_log)
        report['tests']['mini100']['returncode'] = rc
        report['tests']['mini100']['token_acc'] = token_acc
        report['tests']['mini100']['final_loss'] = loss

        ckpt = find_latest_checkpoint(mini_out)
        if ckpt:
            generate_inference_script(
                args,
                checkpoint=str(ckpt),
                dataset=str(out_dir / 'mini100.jsonl'),
                output=mini_infer,
                num_samples=100,
                max_new_tokens=128,
            )

        # Execute batch size test.
        max_bs = None
        for bs in [8, 6, 4, 2, 1]:
            bs_log = out_dir / f'bs_test_{bs}.log'
            rc = run_docker_command(['bash', str(out_dir / f'run_bs_test_{bs}.sh')], bs_log, gpu='0,1,2,3,4,5,6')
            oom = False
            if bs_log.exists():
                text = bs_log.read_text(errors='ignore')
                oom = 'OutOfMemoryError' in text or 'CUDA out of memory' in text
            report['tests']['batch_size']['candidates'][str(bs)]['returncode'] = rc
            report['tests']['batch_size']['candidates'][str(bs)]['oom'] = oom
            if rc == 0 and not oom and max_bs is None:
                max_bs = bs
        report['tests']['batch_size']['max_batch_size'] = max_bs

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
