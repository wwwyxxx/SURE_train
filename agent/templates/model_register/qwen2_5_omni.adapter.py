"""Adapter for Qwen2.5-Omni runtime model registration."""


def render(config: dict) -> str:
    """Render a custom model-register script for qwen2_5_omni_custom.

    Expected component_paths keys:
      - language_model: path to Qwen2.5-7B (or compatible LLM)
      - vision_tower: path to Whisper-large-v3 (encoder only)
    """
    derived = config['derived_model_type']
    base = config['base_model_path']
    component_paths = config['component_paths']
    llm_path = component_paths.get('language_model', '/workspace/model/Qwen2.5-7B')
    audio_encoder_path = component_paths.get('vision_tower', '/workspace/model/whisper-large-v3')

    return f'''#!/usr/bin/env python3
"""Auto-generated custom model registration for {derived}.

Base model: {base}
LLM source: {llm_path}
Audio encoder source: {audio_encoder_path}
"""
import os
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, WhisperModel

from swift.llm import Model, ModelGroup, ModelMeta, TemplateType, register_model
from swift.utils import get_logger


logger = get_logger()


def _replace_thinker_model(model, llm_path: str, torch_dtype):
    """Replace model.thinker.model weights with Qwen2.5-7B, on CPU."""
    logger.info(f'[{derived}] Loading LLM from {{llm_path}} on CPU')
    qwen_model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch_dtype,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    llm_state = qwen_model.model.state_dict()
    del qwen_model

    missing, unexpected = model.thinker.model.load_state_dict(llm_state, strict=False)
    logger.info(f'[{derived}] LLM replaced: copied {{len(llm_state)}} tensors')
    if missing:
        logger.info(f'[{derived}] LLM missing keys: {{missing[:10]}}')
    if unexpected:
        logger.info(f'[{derived}] LLM unexpected keys: {{unexpected[:10]}}')
    del llm_state


def _replace_audio_encoder(model, audio_encoder_path: str, torch_dtype):
    """Replace thinker.audio_tower.encoder weights with Whisper, keep Omni layers."""
    logger.info(f'[{derived}] Loading audio encoder from {{audio_encoder_path}} on CPU')
    whisper = WhisperModel.from_pretrained(
        audio_encoder_path,
        torch_dtype=torch_dtype,
        device_map='cpu',
        low_cpu_mem_usage=True,
    )
    whisper_state = whisper.encoder.state_dict()
    del whisper

    omni_state = model.thinker.audio_tower.state_dict()
    mapped_state = {{}}
    for k, v in omni_state.items():
        if k.startswith('layers.'):
            mapped_state[k] = whisper_state[k]
        elif k.startswith('embed_positions.') or k in ('layer_norm.weight', 'layer_norm.bias'):
            # Whisper-only keys that Omni does not use.
            continue
        else:
            mapped_state[k] = v

    missing, unexpected = model.thinker.audio_tower.load_state_dict(mapped_state, strict=False)
    logger.info(f'[{derived}] Audio encoder replaced: copied {{len(mapped_state)}} tensors')
    if missing:
        logger.info(f'[{derived}] Missing keys (keep official Omni): {{missing[:10]}}')
    if unexpected:
        logger.info(f'[{derived}] Unexpected keys (ignored): {{unexpected[:10]}}')
    del mapped_state


def get_model_tokenizer_{derived}(
    model_dir, model_info, model_kwargs, load_model, **kwargs
):
    """Load official Qwen2.5-Omni on CPU, replace LLM and audio encoder."""
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import vision_process
    from swift.llm.model.utils import use_submodel_func
    from swift.llm.model.model.qwen import patch_qwen_vl_utils

    torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs['tokenizer'] = processor.tokenizer

    global_vars = patch_qwen_vl_utils(vision_process)
    processor.global_vars = global_vars

    if not load_model:
        return None, processor

    logger.info(f'[{derived}] Loading official Omni from {{model_dir}} on CPU')
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    _replace_thinker_model(model, '{llm_path}', torch_dtype)
    _replace_audio_encoder(model, '{audio_encoder_path}', torch_dtype)

    model_info.config = model.config

    base_model = model.model if 'AWQ' in model.__class__.__name__ else model
    use_submodel_func(base_model, 'thinker')
    if not hasattr(base_model.config, 'keys_to_ignore_at_inference'):
        base_model.config.keys_to_ignore_at_inference = []
    if 'past_key_values' not in base_model.config.keys_to_ignore_at_inference:
        base_model.config.keys_to_ignore_at_inference.append('past_key_values')
    base_model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
    base_model.config.talker_config.pad_token_id = None

    # Freeze all parameters by default; training script selectively unfreezes.
    for p in model.parameters():
        p.requires_grad = False

    import gc
    gc.collect()

    return model, processor


register_model(
    ModelMeta(
        '{derived}',
        [
            ModelGroup([
                Model('Qwen2.5-Omni-7B', '{base}'),
            ]),
        ],
        TemplateType.qwen2_5_omni,
        get_model_tokenizer_{derived},
        model_arch=None,
        architectures=['Qwen2_5OmniModel', 'Qwen2_5OmniForConditionalGeneration'],
        requires=['transformers>=4.50', 'soundfile', 'qwen_omni_utils', 'decord'],
        tags=['vision', 'video', 'audio'],
        additional_saved_files=['spk_dict.pt'],
    ),
    exist_ok=True,
)

logger.info('[{derived}] Model type {derived} registered successfully')
'''
