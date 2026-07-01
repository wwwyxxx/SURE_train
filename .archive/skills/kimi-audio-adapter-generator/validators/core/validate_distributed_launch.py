#!/usr/bin/env python3
"""Validate that DDP training can launch and initialize.

Usage:
    NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python validate_distributed_launch.py \
        --custom-register-path custom/kimi_audio_swift_register.py \
        --model /workspace/model/Qwen2.5-7B \
        --model-type kimi_audio_text \
        --dataset-name combined_asr_aishell_1

Checks:
    1. All ranks initialize process group.
    2. Model is wrapped in DDP.
    3. One forward/backward step runs on all ranks.
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def load_register_module(path: str):
    spec = importlib.util.spec_from_file_location('custom_register', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['custom_register'] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--dataset-name', required=True)
    args = parser.parse_args()

    if not dist.is_initialized():
        dist.init_process_group('nccl')

    rank = int(os.environ.get('RANK', dist.get_rank()))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', dist.get_world_size()))
    torch.cuda.set_device(local_rank)

    load_register_module(args.custom_register_path)

    from swift.llm import get_model_tokenizer, get_template, load_dataset

    model, _ = get_model_tokenizer(
        args.model, model_type=args.model_type, torch_dtype=torch.bfloat16, device_map=None
    )
    model = model.to(f'cuda:{local_rank}')
    model = DDP(model, device_ids=[local_rank])

    template = get_template(args.model_type, load_model_tokenizer=None)
    train_dataset, _ = load_dataset([args.dataset_name], split_dataset_ratio=0.0)

    encoded = template.encode(train_dataset[0], return_length=True)
    batch = template.data_collator([encoded])
    batch = {k: v.to(f'cuda:{local_rank}') if hasattr(v, 'to') else v for k, v in batch.items()}

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(**batch)
    outputs.loss.backward()

    dist.barrier()

    print(f'[OK] Rank {rank}/{world_size} DDP launch validation passed')
    print(f'[OK] loss={outputs.loss.item():.4f}')


if __name__ == '__main__':
    main()
