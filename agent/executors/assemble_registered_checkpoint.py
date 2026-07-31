#!/usr/bin/env python3
"""Assemble a registered ms-swift model checkpoint from per-component sources.

This allows users to initialize each component (language_model, aligner,
vision_tower, generator, ...) from a separate checkpoint path or leave it
randomly initialized, instead of requiring a single official pretrained
checkpoint.

Usage:
    python executors/assemble_registered_checkpoint.py \
        --model-family qwen2_audio \
        --model-type qwen2_audio \
        --base-model-path /workspace/model/Qwen2-Audio-7B \
        --component-paths-json outputs/{run_id}/component_paths.json \
        --output-dir outputs/{run_id}/assembled_model \
        --run-id {run_id}

component_paths.json example:
    {
      "language_model": "/workspace/model/Qwen2.5-7B",
      "aligner": "/workspace/model/Qwen2-Audio-7B",
      "vision_tower": "/workspace/model/whisper-large-v3",
      "generator": "random"
    }

The script uses the component target prefixes defined in
executors/data/registered_model_arch_components.json.  By default each source
key is prepended with the target prefix, e.g. a Qwen2 source key
``model.layers.0.*`` becomes ``language_model.model.layers.0.*`` when the
target prefix is ``language_model``.

For components whose target prefix is a list, or for cross-architecture
components (e.g. Whisper encoder -> audio_tower), provide an optional
component_config JSON with explicit key mappings:

    {
      "vision_tower": {
        "target_submodule": "audio_tower",
        "key_mapping": {"model.encoder.conv1": "conv1", ...}
      }
    }
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
COMPONENT_MAP_PATH = SCRIPT_DIR / 'data' / 'registered_model_arch_components.json'
AUTO_MAPPING_PATH = SCRIPT_DIR / 'data' / 'registered_component_key_mappings.json'


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def detect_source_model_type(source_path: str) -> Optional[str]:
    """Detect source model type by reading its config.json if available."""
    p = Path(source_path)
    if p.is_file():
        p = p.parent
    config_path = p / 'config.json'
    if not config_path.exists():
        return None
    try:
        config = load_json(str(config_path))
        model_type = config.get('model_type')
        if model_type:
            return model_type
        architectures = config.get('architectures') or []
        if architectures:
            # e.g. WhisperForConditionalGeneration -> whisper
            arch = architectures[0]
            # Strip suffixes like ForConditionalGeneration, ForCausalLM, Model
            for suffix in ['ForConditionalGeneration', 'ForCausalLM', 'ForMaskedLM', 'ForSequenceClassification', 'Model']:
                if arch.endswith(suffix):
                    return arch[: -len(suffix)].lower()
            return arch.lower()
    except Exception:
        return None
    return None


def get_auto_key_mapping(model_type: str, component_name: str,
                         source_model_type: str) -> Optional[Dict[str, Any]]:
    """Look up a default key mapping for known cross-architecture components."""
    if not AUTO_MAPPING_PATH.exists():
        return None
    try:
        mappings = load_json(str(AUTO_MAPPING_PATH))
    except Exception:
        return None
    model_mappings = mappings.get(model_type, {})
    component_mappings = model_mappings.get(component_name, {})
    return component_mappings.get(source_model_type)


def _load_safetensors_state_dict(path: str) -> Dict[str, Any]:
    from safetensors import safe_open
    state = {}
    with safe_open(path, framework='pt', device='cpu') as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def _load_torch_state_dict(path: str) -> Dict[str, Any]:
    import torch
    return torch.load(path, map_location='cpu', weights_only=True)


def discover_checkpoint_files(path: str) -> Tuple[List[str], Optional[str]]:
    """Return (list of weight files, index file or None) for a checkpoint dir."""
    p = Path(path)
    candidates = []
    index_file = None

    safetensors_index = p / 'model.safetensors.index.json'
    pytorch_index = p / 'pytorch_model.bin.index.json'
    single_safetensors = p / 'model.safetensors'
    single_pytorch = p / 'pytorch_model.bin'

    if safetensors_index.exists():
        index_file = str(safetensors_index)
        index = load_json(index_file)
        weight_map = index.get('weight_map', {})
        seen = set()
        for filename in weight_map.values():
            filepath = p / filename
            if str(filepath) not in seen and filepath.exists():
                candidates.append(str(filepath))
                seen.add(str(filepath))
    elif pytorch_index.exists():
        index_file = str(pytorch_index)
        index = load_json(index_file)
        weight_map = index.get('weight_map', {})
        seen = set()
        for filename in weight_map.values():
            filepath = p / filename
            if str(filepath) not in seen and filepath.exists():
                candidates.append(str(filepath))
                seen.add(str(filepath))
    elif single_safetensors.exists():
        candidates = [str(single_safetensors)]
    elif single_pytorch.exists():
        candidates = [str(single_pytorch)]

    return candidates, index_file


def load_state_dict_from_path(path: str) -> Dict[str, Any]:
    """Load a full state dict from a checkpoint directory or single file."""
    if os.path.isfile(path):
        if path.endswith('.safetensors'):
            return _load_safetensors_state_dict(path)
        if path.endswith('.bin') or path.endswith('.pt') or path.endswith('.pth'):
            return _load_torch_state_dict(path)
        raise ValueError(f'Unsupported checkpoint file: {path}')

    files, _ = discover_checkpoint_files(path)
    if not files:
        raise FileNotFoundError(f'No checkpoint weights found in {path}')

    state = {}
    for filepath in files:
        if filepath.endswith('.safetensors'):
            piece = _load_safetensors_state_dict(filepath)
        else:
            piece = _load_torch_state_dict(filepath)
        overlap = set(state.keys()) & set(piece.keys())
        if overlap:
            raise ValueError(f'Duplicate keys loading {filepath}: {list(overlap)[:5]}')
        state.update(piece)
    return state


def save_state_dict(output_dir: str, state: Dict[str, Any]) -> str:
    """Save state dict, preferring safetensors if available."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file
        output_file = output_path / 'model.safetensors'
        save_file(state, str(output_file))
        return str(output_file)
    except Exception as e:
        print(f'[WARN] safetensors save failed ({e}); falling back to torch.save')
        import torch
        output_file = output_path / 'pytorch_model.bin'
        torch.save(state, output_file)
        return str(output_file)


def get_target_prefixes(component_config: Dict[str, Any],
                        component_name: str,
                        default_prefixes: List[str]) -> Tuple[List[str], Optional[Dict[str, str]], bool]:
    """Resolve target prefixes and optional explicit key mapping.

    Returns:
        (target_prefixes, key_mapping or None, drop_unmapped)
    """
    cfg = component_config.get(component_name, {})
    explicit_submodule = cfg.get('target_submodule')
    if explicit_submodule is not None:
        prefixes = [explicit_submodule] if isinstance(explicit_submodule, str) else explicit_submodule
    else:
        prefixes = default_prefixes

    key_mapping = cfg.get('key_mapping')
    if key_mapping is not None:
        key_mapping = {str(k): str(v) for k, v in key_mapping.items()}

    drop_unmapped = cfg.get('drop_unmapped', False)
    return prefixes, key_mapping, bool(drop_unmapped)


def remap_keys(source_state: Dict[str, Any],
               target_prefixes: List[str],
               key_mapping: Optional[Dict[str, str]] = None,
               drop_unmapped: bool = False) -> Dict[str, Any]:
    """Remap source keys to target keys.

    If key_mapping is provided, use it directly (prefixes are ignored).
    Keys not present in key_mapping are kept as-is unless drop_unmapped=True.
    Otherwise prepend each source key with the first target prefix.
    """
    if key_mapping is not None:
        # Sort mapping keys by length descending to avoid partial replacements
        sorted_mapping = sorted(key_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        result = {}
        for src_key, tensor in source_state.items():
            matched = False
            for old, new in sorted_mapping:
                if src_key.startswith(old):
                    result[src_key.replace(old, new, 1)] = tensor
                    matched = True
                    break
            if not matched and not drop_unmapped:
                result[src_key] = tensor
        return result

    if len(target_prefixes) > 1:
        raise ValueError(
            f'Multiple target prefixes {target_prefixes} require an explicit '
            'target_submodule or key_mapping in component_config'
        )
    prefix = target_prefixes[0]
    return {f'{prefix}.{k}': v for k, v in source_state.items()}


def patch_omni_audio_tower(assembled_state: Dict[str, Any],
                           base_model_path: str,
                           component_paths: Dict[str, str],
                           component_config: Dict[str, Any],
                           arch_components: Dict[str, List[str]]) -> bool:
    """If the audio tower was initialized from Whisper, copy Omni-specific layers from the official base model.

    Qwen2.5-Omni's thinker.audio_tower is based on Whisper encoder but adds:
      - audio_bos_eos_token
      - ln_post
      - proj
    and removes Whisper's final encoder layer_norm / embed_positions.
    When Whisper is used as the source, those extra layers are missing, so we copy them from the official base model.
    """
    if not base_model_path or not os.path.exists(base_model_path):
        return False

    model_cfg = load_json(os.path.join(base_model_path, 'config.json'))
    if model_cfg.get('model_type') != 'qwen2_5_omni':
        return False

    vision_source = component_paths.get('vision_tower')
    if vision_source is None or vision_source == 'random':
        return False
    if detect_source_model_type(vision_source) not in ('whisper', 'whisperforconditionalgeneration'):
        return False

    base_state = load_state_dict_from_path(base_model_path)
    omni_audio_keys = [k for k in base_state.keys() if k.startswith('thinker.audio_tower.')]
    patched = 0
    for k in omni_audio_keys:
        if k.endswith('audio_bos_eos_token.weight') or '.ln_post.' in k or '.proj.' in k:
            if k not in assembled_state:
                assembled_state[k] = base_state[k]
                patched += 1
            elif assembled_state[k].shape != base_state[k].shape:
                assembled_state[k] = base_state[k]
                patched += 1

    # Also remove Whisper-only keys that Omni does not use
    whisper_only = [k for k in assembled_state.keys()
                    if k.startswith('thinker.audio_tower.embed_positions.')
                    or k == 'thinker.audio_tower.layer_norm.weight'
                    or k == 'thinker.audio_tower.layer_norm.bias']
    for k in whisper_only:
        del assembled_state[k]

    print(f'[PATCH] Copied {patched} Omni-specific audio_tower tensors; removed {len(whisper_only)} Whisper-only tensors')
    return patched > 0 or len(whisper_only) > 0


def copy_base_model_files(base_model_path: str, output_dir: str, report: Dict[str, Any]):

    """Copy config, tokenizer, and processor files from base checkpoint."""
    base = Path(base_model_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    copied = []
    skipped = []

    # Always try to copy config.json
    config_src = base / 'config.json'
    if config_src.exists():
        shutil.copy2(config_src, out / 'config.json')
        copied.append('config.json')
    else:
        skipped.append('config.json')

    # Tokenizer / preprocessor files
    patterns = [
        'tokenizer*.json', 'tokenizer*.model', 'vocab*.json', 'vocab*.txt',
        'preprocessor_config.json', 'processor_config.json', 'chat_template.json',
        'added_tokens.json', 'special_tokens_map.json', 'merges.txt',
        '*.txt', '*.json',
    ]
    seen = set()
    for pattern in patterns:
        for src_file in base.glob(pattern):
            if src_file.name in seen:
                continue
            if src_file.name == 'config.json':
                continue
            # Avoid copying weight files / index files from base path
            if src_file.suffix in ('.bin', '.safetensors', '.pt', '.pth'):
                continue
            if src_file.name.endswith('.safetensors.index.json') or src_file.name.endswith('.bin.index.json'):
                continue
            try:
                shutil.copy2(src_file, out / src_file.name)
                copied.append(src_file.name)
                seen.add(src_file.name)
            except Exception as exc:
                skipped.append(f'{src_file.name}: {exc}')

    report['base_model_files_copied'] = copied
    report['base_model_files_skipped'] = skipped


def assemble_checkpoint(args) -> Dict[str, Any]:
    if not COMPONENT_MAP_PATH.exists():
        raise FileNotFoundError(f'Component mapping not found: {COMPONENT_MAP_PATH}')

    component_map = load_json(str(COMPONENT_MAP_PATH))
    model_type = args.model_type or args.model_family
    if model_type not in component_map:
        raise ValueError(
            f'No component mapping found for model_type {model_type}. '
            f'Supported: {list(component_map.keys())}'
        )

    arch_components = component_map[model_type]
    component_paths = {}
    if args.component_paths_json:
        component_paths = load_json(args.component_paths_json)

    component_config = {}
    if args.component_config_json:
        component_config = load_json(args.component_config_json)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assembled_state: Dict[str, Any] = {}
    report = {
        'model_family': args.model_family,
        'model_type': model_type,
        'assembled_model_path': str(output_dir),
        'components': {},
        'random_components': [],
        'warnings': [],
    }

    for component_name, default_prefixes in arch_components.items():
        source = component_paths.get(component_name)
        if source is None or source == 'random':
            report['random_components'].append(component_name)
            continue

        if not os.path.exists(source):
            report['warnings'].append(f'{component_name}: source path does not exist: {source}')
            report['random_components'].append(component_name)
            continue

        source_model_type = detect_source_model_type(source)

        try:
            source_state = load_state_dict_from_path(source)
        except Exception as exc:
            report['warnings'].append(f'{component_name}: failed to load {source}: {exc}')
            report['random_components'].append(component_name)
            continue

        try:
            prefixes, key_mapping, drop_unmapped = get_target_prefixes(
                component_config, component_name, default_prefixes
            )
            drop_unmapped = bool(drop_unmapped)
            auto_mapping = None
            # If user did not provide a key mapping, try auto mapping for known cross-arch cases
            if key_mapping is None and source_model_type:
                auto_mapping = get_auto_key_mapping(model_type, component_name, source_model_type)
                if auto_mapping:
                    key_mapping = auto_mapping.get('key_mapping')
                    drop_unmapped = auto_mapping.get('drop_unmapped', False)

            remapped = remap_keys(source_state, prefixes, key_mapping, drop_unmapped=drop_unmapped)
        except Exception as exc:
            report['warnings'].append(f'{component_name}: key remapping failed: {exc}')
            report['random_components'].append(component_name)
            continue

        overlap = set(assembled_state.keys()) & set(remapped.keys())
        if overlap:
            report['warnings'].append(
                f'{component_name}: {len(overlap)} keys overlap with already assembled weights; '
                'later component overwrote earlier ones'
            )
        assembled_state.update(remapped)

        report['components'][component_name] = {
            'source_path': source,
            'source_model_type': source_model_type,
            'source_keys': len(source_state),
            'remapped_keys': len(remapped),
            'target_prefixes': prefixes,
            'key_mapping_used': key_mapping is not None,
            'auto_mapping_used': auto_mapping is not None,
        }

    # Save assembled weights
    try:
        saved_path = save_state_dict(str(output_dir), assembled_state)
        report['saved_weights'] = saved_path
    except Exception as exc:
        report['warnings'].append(f'Failed to save assembled weights: {exc}')
        raise

    # Post-process audio tower: if Whisper was used to initialize vision_tower/audio_tower,
    # fill missing Omni-specific layers (ln_post, proj, audio_bos_eos_token) from official Omni.
    try:
        patched = patch_omni_audio_tower(
            assembled_state,
            args.base_model_path,
            component_paths,
            component_config,
            arch_components,
        )
        if patched:
            saved_path = save_state_dict(str(output_dir), assembled_state)
            report['saved_weights'] = saved_path
            report['audio_tower_patched'] = True
    except Exception as exc:
        report['warnings'].append(f'Audio tower patch failed: {exc}')

    # Copy base model auxiliary files if requested
    if args.base_model_path:
        copy_base_model_files(args.base_model_path, str(output_dir), report)
    else:
        report['base_model_path'] = None
        report['base_model_files_copied'] = []
        report['base_model_files_skipped'] = []

    # Write report
    if args.output_report:
        report_path = args.output_report
    else:
        report_path = str(output_dir / 'checkpoint_assembly_report.json')
    save_json(report_path, report)
    print(f'[OK] Assembly report written to {report_path}')
    print(json.dumps(report, indent=2, ensure_ascii=False))

    return report


def main():
    parser = argparse.ArgumentParser(description='Assemble registered checkpoint from components')
    parser.add_argument('--model-family', required=True,
                        help='Harness model_family, e.g. qwen2_audio')
    parser.add_argument('--model-type', default=None,
                        help='ms-swift model_type; defaults to model_family')
    parser.add_argument('--base-model-path', default=None,
                        help='Official/base checkpoint to copy config/tokenizer/processor from')
    parser.add_argument('--component-paths-json', required=True,
                        help='JSON file mapping component names to checkpoint paths or "random"')
    parser.add_argument('--component-config-json', default=None,
                        help='Optional JSON with per-component target_submodule / key_mapping overrides')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write assembled checkpoint')
    parser.add_argument('--output-report', default=None,
                        help='Path for checkpoint_assembly_report.json (default: output_dir/checkpoint_assembly_report.json)')
    args = parser.parse_args()

    assemble_checkpoint(args)


if __name__ == '__main__':
    main()
