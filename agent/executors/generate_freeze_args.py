#!/usr/bin/env python3
"""Generate ms-swift freeze/train CLI arguments from component_trainable strategy."""
import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_text_head_param(model_family: str, model_type: str, component_trainable: dict,
                            override: str | None, mapping_path: Path) -> str | None:
    """Resolve the actual parameter name for the text head pseudo-component."""
    if not component_trainable.get('text_head'):
        return None
    if override:
        return override

    text_head_map = load_json(str(mapping_path))
    for key in (model_type, model_family):
        if key in text_head_map:
            return text_head_map[key]

    # Fallback: try common patterns based on language_model prefix
    components_path = mapping_path.parent / 'registered_model_arch_components.json'
    components = load_json(str(components_path)) if components_path.exists() else {}
    for key in (model_type, model_family):
        if key not in components:
            continue
        lm = components[key].get('language_model')
        if isinstance(lm, str):
            return f"{lm}.lm_head"
        if isinstance(lm, list) and lm:
            for prefix in lm:
                if prefix.endswith('.model'):
                    return prefix[: -len('.model')] + '.lm_head'
            return f"{lm[0]}.lm_head"
    return None


def _single_prefix(prefixes):
    if isinstance(prefixes, list):
        if len(prefixes) != 1:
            raise ValueError(f'Expected a single prefix, got {prefixes}')
        return prefixes[0]
    return prefixes


def resolve_component_param(component_name: str, model_type: str, arch_components: dict) -> str | None:
    """Resolve a component name to its parameter prefix."""
    if model_type not in arch_components:
        return None
    prefixes = arch_components[model_type].get(component_name)
    if not prefixes:
        return None
    try:
        return _single_prefix(prefixes)
    except ValueError:
        # For nested aligners (e.g. qwen2_5_omni's thinker.audio_tower.proj) the
        # mapping may already be a concrete parameter prefix.
        if isinstance(prefixes, list):
            return prefixes[0]
        return prefixes


def generate_freeze_args(component_trainable: dict, text_head_param: str | None,
                         additional_trainable: list[str] | None,
                         use_model_register: bool,
                         arch_components: dict,
                         model_type: str) -> list[str]:
    args = []
    mapping = {
        'language_model': '--freeze_llm',
        'vision_tower': '--freeze_vit',
        'aligner': '--freeze_aligner',
    }

    trainable = []

    if use_model_register:
        # Custom register script freezes all parameters by default.
        # Use regex freeze to ensure ms-swift does not re-enable any parameters,
        # then selectively unfreeze via --trainable_parameters.
        args.extend(['--freeze_parameters_regex', '.*'])

        for component, flag in mapping.items():
            if component in component_trainable:
                # Flag is mostly informational; trainable_parameters does the real unfreezing.
                value = 'false' if component_trainable[component] else 'true'
                args.extend([flag, value])
                if component_trainable[component]:
                    prefix = resolve_component_param(component, model_type, arch_components)
                    if prefix:
                        trainable.append(prefix)
    else:
        for component, flag in mapping.items():
            if component in component_trainable:
                value = 'false' if component_trainable[component] else 'true'
                args.extend([flag, value])

    # generator is usually frozen for ASR; warn if user wants to train it
    if component_trainable.get('generator'):
        print('[WARN] generator=true is unusual for ASR-only tasks; consider setting it to false',
              file=sys.stderr)
        if not use_model_register:
            args.extend(['--freeze_aligner', 'false'])  # placeholder; ms-swift has no --freeze_generator
        else:
            prefix = resolve_component_param('generator', model_type, arch_components)
            if prefix:
                trainable.append(prefix)

    if text_head_param:
        trainable.append(text_head_param)
    if additional_trainable:
        trainable.extend(additional_trainable)

    if trainable:
        # ms-swift accepts --trainable_parameters multiple times or as a list
        for param in trainable:
            args.extend(['--trainable_parameters', param])

    return args


def main():
    parser = argparse.ArgumentParser(
        description='Generate ms-swift freeze/train arguments from component_trainable strategy.')
    parser.add_argument('--model-family', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--component-trainable-json', required=True,
                        help='Path to JSON file containing component_trainable object.')
    parser.add_argument('--text-head-param', default=None,
                        help='Override the resolved text head parameter name.')
    parser.add_argument('--additional-trainable-parameters-json', default=None,
                        help='Optional JSON file with list of additional trainable parameter prefixes.')
    parser.add_argument('--use-model-register', action='store_true',
                        help='Generate args for a run that uses a custom model-register script. '
                             'This adds --freeze_parameters_regex and emits --trainable_parameters '
                             'for trainable components.')
    parser.add_argument('--output', required=True,
                        help='Path to write the output JSON report.')
    parser.add_argument('--output-format', choices=['json', 'args'], default='json',
                        help='Output a JSON report or a plain argument string.')
    args = parser.parse_args()

    component_trainable = load_json(args.component_trainable_json)
    additional = None
    if args.additional_trainable_parameters_json:
        additional = load_json(args.additional_trainable_parameters_json)

    script_dir = Path(__file__).resolve().parent
    mapping_path = script_dir / 'data' / 'registered_model_text_head.json'
    arch_components_path = script_dir / 'data' / 'registered_model_arch_components.json'
    arch_components = load_json(str(arch_components_path)) if arch_components_path.exists() else {}

    text_head_param = resolve_text_head_param(
        args.model_family, args.model_type, component_trainable,
        args.text_head_param, mapping_path)

    if component_trainable.get('text_head') and not text_head_param:
        print('[ERROR] Could not resolve text_head parameter name automatically. '
              'Provide --text-head-param explicitly.', file=sys.stderr)
        sys.exit(1)

    freeze_args = generate_freeze_args(
        component_trainable, text_head_param, additional,
        args.use_model_register, arch_components, args.model_type)

    def _fmt(args_list):
        return ' \\\n  '.join(args_list) if args_list else ''

    if args.output_format == 'args':
        Path(args.output).write_text(_fmt(freeze_args), encoding='utf-8')
    else:
        report = {
            'model_family': args.model_family,
            'model_type': args.model_type,
            'text_head_param': text_head_param,
            'freeze_args': freeze_args,
            'freeze_args_string': _fmt(freeze_args),
        }
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'[OK] Freeze args written to {args.output}')


if __name__ == '__main__':
    main()
