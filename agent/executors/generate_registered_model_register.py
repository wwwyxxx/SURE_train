#!/usr/bin/env python3
"""Generate a custom model-registration script for registered ms-swift models.

When a registered model is initialized from per-component sources (e.g. Qwen2.5-7B
for the LLM, Whisper for the audio encoder), this script produces a Python file
that registers a derived model_type and performs the component replacement at
runtime. The pipeline no longer needs to assemble a static checkpoint.

Usage:
    python executors/generate_registered_model_register.py \
        --model-family qwen2_5_omni \
        --model-type qwen2_5_omni \
        --base-model-path /workspace/model/Qwen2.5-Omni-7B \
        --component-paths-json outputs/{run_id}/component_paths.json \
        --output outputs/{run_id}/custom/qwen2_5_omni_registered_model_register.py

component_paths.json example:
    {
      "language_model": "/workspace/model/Qwen2.5-7B",
      "vision_tower": "/workspace/model/whisper-large-v3"
    }
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = SCRIPT_DIR.parent / 'templates' / 'model_register'
COMPONENT_MAP_PATH = SCRIPT_DIR / 'data' / 'registered_model_arch_components.json'


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_text(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def load_adapter(model_type: str):
    """Load per-model adapter if available, otherwise fall back to default."""
    adapter_path = ADAPTER_DIR / f'{model_type}.adapter.py'
    if not adapter_path.exists():
        adapter_path = ADAPTER_DIR / 'default.adapter.py'

    module_name = f'model_register_adapter_{model_type}'
    spec = importlib.util.spec_from_file_location(module_name, str(adapter_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def generate(config: Dict[str, Any]) -> str:
    """Generate the registration script string."""
    model_type = config['model_type']
    adapter = load_adapter(model_type)
    if not hasattr(adapter, 'render'):
        raise ValueError(f'Adapter for {model_type} does not define render(config)')
    return adapter.render(config)


def main():
    parser = argparse.ArgumentParser(
        description='Generate a custom model-registration script for registered models.')
    parser.add_argument('--model-family', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--base-model-path', required=True,
                        help='Official/base checkpoint used as --model at runtime.')
    parser.add_argument('--component-paths-json', required=True,
                        help='JSON mapping component names to source checkpoint paths or "random".')
    parser.add_argument('--output', required=True,
                        help='Path for the generated Python register script.')
    parser.add_argument('--derived-model-type', default=None,
                        help='Derived model_type to register. Defaults to {model_type}_custom.')
    args = parser.parse_args()

    component_paths = load_json(args.component_paths_json)

    derived_model_type = args.derived_model_type or f'{args.model_type}_custom'

    arch_components = load_json(str(COMPONENT_MAP_PATH))
    if args.model_type not in arch_components:
        print(f'[WARN] No architecture component mapping for {args.model_type}; '
              'default adapter may fail.', file=sys.stderr)
        arch_components = {}

    config = {
        'model_family': args.model_family,
        'model_type': args.model_type,
        'derived_model_type': derived_model_type,
        'base_model_path': args.base_model_path,
        'component_paths': component_paths,
        'arch_components': arch_components.get(args.model_type, {}),
    }

    script = generate(config)
    save_text(args.output, script)
    print(f'[OK] Generated model register script: {args.output}')


if __name__ == '__main__':
    main()
