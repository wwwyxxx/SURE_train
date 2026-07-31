"""Default adapter for runtime model registration of registered ms-swift models.

This adapter performs whole-submodule replacement. It works when each component
in ``component_paths`` maps to a single submodule of the base model.
For models that need partial replacement or extra post-processing (e.g.
qwen2_5_omni's audio tower), use a dedicated per-model adapter instead.
"""
import json


def _resolve_target_path(component_name: str, arch_components: dict) -> str:
    prefixes = arch_components.get(component_name)
    if not prefixes:
        raise ValueError(f'No target prefix defined for component {component_name}')
    if isinstance(prefixes, list):
        if len(prefixes) != 1:
            raise ValueError(
                f'Component {component_name} has multiple target prefixes {prefixes}; '
                'default adapter cannot handle it. Use a per-model adapter.'
            )
        prefixes = prefixes[0]
    return prefixes


def render(config: dict) -> str:
    """Render a generic model-register Python script.

    config keys:
      - model_type: base ms-swift model_type (e.g. qwen2_audio)
      - derived_model_type: new model_type to register (e.g. qwen2_audio_custom)
      - base_model_path: official/base checkpoint path
      - component_paths: dict {component_name: source_path or "random"}
      - arch_components: dict from registered_model_arch_components.json
    """
    derived = config['derived_model_type']
    base = config['base_model_path']
    component_paths = config['component_paths']
    arch_components = config['arch_components']

    replacements = []
    for component_name, source_path in component_paths.items():
        if source_path == 'random':
            continue
        target_path = _resolve_target_path(component_name, arch_components)
        replacements.append((component_name, source_path, target_path))

    replacements_json = json.dumps(replacements, ensure_ascii=False)

    return f'''"""Auto-generated custom model registration for {{derived}}.

Base model: {{base}}
Component replacements: {json.dumps(component_paths, ensure_ascii=False)}
"""
import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModel, AutoProcessor
from swift.llm import ModelInfo, register_model


@register_model(
    ModelInfo(
        model_type='{{derived}}',
        model_dir='{{base}}',
        task_type='causal_lm',
    ),
    exist_ok=True,
)
def get_model_tokenizer_{{derived.replace('-', '_')}}(
    model_dir: str,
    model_info: ModelInfo,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs,
) -> Tuple[Any, Any]:
    torch_dtype = model_kwargs.get('torch_dtype', torch.bfloat16)

    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    if not load_model:
        return None, processor

    model = AutoModel.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    replacements = json.loads({repr(replacements_json)})
    for component_name, source_path, target_path in replacements:
        src_model = AutoModel.from_pretrained(
            source_path,
            torch_dtype=torch_dtype,
            device_map='cpu',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # navigate to parent and set submodule
        parts = target_path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], src_model)

    # Freeze all parameters by default; training script will selectively unfreeze.
    for p in model.parameters():
        p.requires_grad = False

    return model, processor
'''
