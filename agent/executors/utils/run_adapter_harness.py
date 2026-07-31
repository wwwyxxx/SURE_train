#!/usr/bin/env python
import argparse
import shlex
import subprocess
import sys
from typing import List


def run(cmd: List[str], *, dry_run: bool = False) -> None:
    print('\n$ ' + ' '.join(shlex.quote(x) for x in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def add_validation_args(cmd: List[str], args, stage: str) -> List[str]:
    cmd += [
        sys.executable,
        args.validator,
        '--custom-register-path',
        args.custom_register_path,
        '--model',
        args.model,
        '--model-type',
        args.model_type,
        '--template',
        args.template,
        '--dataset',
        args.dataset,
        '--stage',
        stage,
    ]
    if args.required_keys:
        cmd += ['--required-keys'] + args.required_keys
    if args.required_batch_keys:
        cmd += ['--required-batch-keys'] + args.required_batch_keys
    if args.assert_text_audio_shapes:
        cmd.append('--assert-text-audio-shapes')
    if args.must_train_prefix:
        cmd += ['--must-train-prefix'] + args.must_train_prefix
    if args.must_freeze_prefix:
        cmd += ['--must-freeze-prefix'] + args.must_freeze_prefix
    if args.no_backward:
        cmd.append('--no-backward')
    return cmd


def overfit_cmd(args) -> List[str]:
    cmd = [
        'swift',
        'sft',
        '--custom_register_path',
        args.custom_register_path,
        '--model',
        args.model,
        '--model_type',
        args.model_type,
        '--dataset',
        args.dataset,
        '--train_type',
        args.train_type,
        '--split_dataset_ratio',
        '0',
        '--per_device_train_batch_size',
        str(args.batch_size),
        '--gradient_accumulation_steps',
        str(args.gradient_accumulation_steps),
        '--num_train_epochs',
        str(args.num_train_epochs),
        '--learning_rate',
        str(args.learning_rate),
        '--max_length',
        str(args.max_length),
        '--logging_steps',
        str(args.logging_steps),
        '--save_steps',
        str(args.save_steps),
        '--save_total_limit',
        str(args.save_total_limit),
        '--output_dir',
        args.output_dir,
        '--report_to',
        'none',
    ]
    if args.bf16:
        cmd += ['--bf16', 'true']
    if args.gradient_checkpointing:
        cmd += ['--gradient_checkpointing', 'true']
    if args.freeze_llm is not None:
        cmd += ['--freeze_llm', str(args.freeze_llm).lower()]
    if args.freeze_vit is not None:
        cmd += ['--freeze_vit', str(args.freeze_vit).lower()]
    if args.freeze_aligner is not None:
        cmd += ['--freeze_aligner', str(args.freeze_aligner).lower()]
    if args.save_only_model:
        cmd += ['--save_only_model', 'true']
    return cmd


def parse_bool(value: str):
    if value.lower() in {'true', '1', 'yes'}:
        return True
    if value.lower() in {'false', '0', 'no'}:
        return False
    raise argparse.ArgumentTypeError(f'expected bool, got {value!r}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run staged ms-swift custom adapter harness checks.')
    parser.add_argument('--validator', default='skills/swift-model-adapter/scripts/validate_swift_adapter.py')
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--template', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=['cpu', 'forward', 'overfit', 'infer'],
        default=['cpu'])
    parser.add_argument('--required-keys', nargs='*', default=['input_ids', 'labels'])
    parser.add_argument('--required-batch-keys', nargs='*', default=['input_ids', 'labels'])
    parser.add_argument('--assert-text-audio-shapes', action='store_true')
    parser.add_argument('--must-train-prefix', nargs='*', default=[])
    parser.add_argument('--must-freeze-prefix', nargs='*', default=[])
    parser.add_argument('--no-backward', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    parser.add_argument('--train-type', default='full')
    parser.add_argument('--freeze-llm', type=parse_bool, default=None)
    parser.add_argument('--freeze-vit', type=parse_bool, default=None)
    parser.add_argument('--freeze-aligner', type=parse_bool, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--num-train-epochs', type=float, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--logging-steps', type=int, default=1)
    parser.add_argument('--save-steps', type=int, default=1000000)
    parser.add_argument('--save-total-limit', type=int, default=1)
    parser.add_argument('--save-only-model', action='store_true')
    parser.add_argument('--output-dir', default='output/adapter_harness_overfit')

    parser.add_argument('--infer-script')
    parser.add_argument('--checkpoint')
    parser.add_argument('--audio')
    parser.add_argument('--prompt')
    parser.add_argument('--max-new-tokens', type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for stage in args.stages:
        if stage == 'cpu':
            run(add_validation_args([], args, 'cpu'), dry_run=args.dry_run)
        elif stage == 'forward':
            run(add_validation_args([], args, 'forward'), dry_run=args.dry_run)
        elif stage == 'overfit':
            run(overfit_cmd(args), dry_run=args.dry_run)
        elif stage == 'infer':
            if not args.infer_script:
                raise SystemExit('--infer-script is required for infer stage')
            cmd = [sys.executable, args.infer_script]
            if args.checkpoint:
                cmd += ['--checkpoint', args.checkpoint]
            if args.audio:
                cmd += ['--audio', args.audio]
            if args.prompt:
                cmd += ['--prompt', args.prompt]
            if args.max_new_tokens is not None:
                cmd += ['--max-new-tokens', str(args.max_new_tokens)]
            run(cmd, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
