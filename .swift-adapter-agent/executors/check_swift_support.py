#!/usr/bin/env python3
"""Check whether ms-swift already supports a given model family.

This script parses the ms-swift source code (it does NOT import swift,
because the base environment may lack dependencies like transformers).

Usage:
    python executors/check_swift_support.py \
        --model-family qwen2_audio \
        --sure-train-dir SURE_train \
        --output outputs/{run_id}/swift_support_report.json

Output JSON:
    {
      "supported": true,
      "model_type": "qwen2_audio",
      "support_level": "native",
      "category": "MLLMModelType",
      "inference_engine": "PtEngine",
      "has_training_examples": false,
      "has_inference_examples": false,
      "notes": "Native ms-swift model type. Use --model_type qwen2_audio."
    }
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# Map harness model_family values to candidate ms-swift model_type values.
# A model_family may match several candidates; we pick the first one found.
FAMILY_TO_CANDIDATES = {
    'kimi_audio': ['kimi_audio'],
    'mimo_audio': ['mimo_audio'],
    'qwen2_audio': ['qwen2_audio', 'qwen_audio'],
    'qwen_audio': ['qwen_audio', 'qwen2_audio'],
    'qwen_omni': ['qwen2_5_omni', 'qwen3_omni'],
    'qwen2_5_omni': ['qwen2_5_omni'],
    'qwen3_omni': ['qwen3_omni'],
    'step_audio': ['step_audio'],
    'step_audio2_mini': ['step_audio2_mini', 'step_audio'],
    'qwen2': ['qwen2', 'qwen2_5'],
    'llama3': ['llama3'],
    'other': [],
}


def resolve_sure_train_dir(sure_train_dir: str) -> str:
    """Resolve SURE_train directory, allowing relative paths from project root."""
    if os.path.isdir(os.path.join(sure_train_dir, 'model')):
        return sure_train_dir
    parent = os.path.join('..', sure_train_dir)
    if os.path.isdir(os.path.join(parent, 'model')):
        return os.path.abspath(parent)
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


def parse_constant_file(constant_path: str) -> dict:
    """Parse model type constants from swift/llm/model/constant.py.

    Returns:
        dict: {model_type: category}
    """
    categories = [
        ('LLMModelType', 'LLM'),
        ('MLLMModelType', 'MLLM'),
        ('BertModelType', 'Bert'),
        ('RMModelType', 'Reward'),
        ('RerankerModelType', 'Reranker'),
    ]
    result = {}
    with open(constant_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for class_name, category in categories:
        # Find the class body
        pattern = rf'class {class_name}:\n(.*?)\n\n?class '
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue
        body = match.group(1)
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Match: xxx = 'yyy'
            m = re.match(r"([a-zA-Z0-9_]+)\s*=\s*['\"]([a-zA-Z0-9_]+)['\"]", line)
            if m:
                model_type = m.group(2)
                result[model_type] = category
    return result


def check_examples(sure_train_dir: str, model_type: str) -> dict:
    """Check whether ms-swift has training/inference examples for this model_type."""
    ms_swift = os.path.join(sure_train_dir, 'ms-swift')
    examples_dir = os.path.join(ms_swift, 'examples')
    if not os.path.isdir(examples_dir):
        return {'training': False, 'inference': False}

    has_training = False
    has_inference = False
    model_type_pattern = model_type.replace('_', '[_-]?')

    for root, _, files in os.walk(examples_dir):
        for fname in files:
            if not fname.endswith(('.sh', '.py', '.md')):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception:
                continue
            if re.search(model_type_pattern, text, re.IGNORECASE):
                lower_text = text.lower()
                if 'sft' in lower_text or 'train' in lower_text or 'finetune' in lower_text:
                    has_training = True
                if 'infer' in lower_text:
                    has_inference = True

    return {'training': has_training, 'inference': has_inference}


def infer_category_notes(category: str, model_type: str) -> str:
    if category == 'MLLM':
        return (
            f'{model_type} is a native multimodal model type in ms-swift. '
            'Training uses --model_type and --model. Inference typically uses PtEngine.'
        )
    if category == 'LLM':
        return (
            f'{model_type} is a native LLM type in ms-swift. '
            'Training and inference are fully supported.'
        )
    return f'{model_type} is registered in ms-swift category {category}.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-family', required=True)
    parser.add_argument('--sure-train-dir', default='SURE_train')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    sure_train_dir = resolve_sure_train_dir(args.sure_train_dir)
    constant_path = os.path.join(
        sure_train_dir, 'ms-swift', 'swift', 'llm', 'model', 'constant.py'
    )

    if not os.path.isfile(constant_path):
        result = {
            'supported': False,
            'model_type': None,
            'support_level': 'unknown',
            'category': None,
            'inference_engine': None,
            'has_training_examples': False,
            'has_inference_examples': False,
            'notes': f'ms-swift constant.py not found at {constant_path}',
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    supported_types = parse_constant_file(constant_path)
    candidates = FAMILY_TO_CANDIDATES.get(args.model_family, [args.model_family])

    matched_type = None
    for cand in candidates:
        if cand in supported_types:
            matched_type = cand
            break

    if matched_type is None:
        result = {
            'supported': False,
            'model_type': None,
            'support_level': 'none',
            'category': None,
            'inference_engine': None,
            'has_training_examples': False,
            'has_inference_examples': False,
            'notes': (
                f'{args.model_family} is not a native ms-swift model_type. '
                'Custom registration (register_model + register_template) is required.'
            ),
        }
    else:
        category = supported_types[matched_type]
        examples = check_examples(sure_train_dir, matched_type)
        inference_engine = 'PtEngine'
        if category == 'LLM':
            inference_engine = 'PtEngine / vLLM / lmdeploy / sglang'
        result = {
            'supported': True,
            'model_type': matched_type,
            'support_level': 'native',
            'category': category,
            'inference_engine': inference_engine,
            'has_training_examples': examples['training'],
            'has_inference_examples': examples['inference'],
            'notes': infer_category_notes(category, matched_type),
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
