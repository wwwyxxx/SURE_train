"""Adapter for Qwen2-Audio runtime model registration.

Includes the audio-token-count fix for the native Qwen2AudioTemplate.
"""


def render(config: dict) -> str:
    derived = config['derived_model_type']
    base = config['base_model_path']
    component_paths = config['component_paths']
    llm_path = component_paths.get('language_model', '/workspace/model/Qwen2.5-7B')
    audio_encoder_path = component_paths.get('vision_tower', '/workspace/model/whisper-large-v3')

    return f'''"""Auto-generated custom model registration for {derived}.

Base model: {base}
LLM source: {llm_path}
Audio encoder source: {audio_encoder_path}
"""
import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Qwen2ForCausalLM,
    WhisperModel,
)
from swift.llm import ModelInfo, get_model_tokenizer, register_model, register_template
from swift.llm.template import Template
from swift.llm.template.constant import MLLMTemplateType
from swift.llm.template.template.qwen import Qwen2AudioTemplate, QwenTemplateMeta
from swift.llm.template.vision_utils import load_audio, load_batch


_ROOT = os.path.dirname(os.path.abspath(__file__))


class Qwen2AudioTemplateFixed(Qwen2AudioTemplate):
    """Fix native template's single <|AUDIO|> placeholder mismatch."""

    def _encode(self, inputs):
        encoded = Template._encode(self, inputs)
        if inputs.audios:
            audios = load_batch(
                inputs.audios,
                load_func=partial(load_audio, sampling_rate=self.sampling_rate),
            )
            audio_inputs = self.processor.feature_extractor(
                audios,
                sampling_rate=self.sampling_rate,
                return_attention_mask=True,
                return_tensors='pt',
            )
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            encoded.update(audio_inputs)

            audio_token_id = self._tokenize('<|AUDIO|>')[0]
            input_ids = encoded['input_ids']
            audio_idx = next(
                (i for i, tid in enumerate(input_ids) if tid == audio_token_id), None
            )
            if audio_idx is not None:
                num_frames = audio_inputs['feature_attention_mask'].sum(-1)
                feat_lengths = (num_frames - 1) // 2 + 1
                num_audio_tokens = ((feat_lengths - 2) // 2 + 1).tolist()
                if isinstance(num_audio_tokens, int):
                    num_audio_tokens = [num_audio_tokens]
                n_tokens = num_audio_tokens[0]
                new_input_ids = (
                    input_ids[:audio_idx]
                    + [audio_token_id] * n_tokens
                    + input_ids[audio_idx + 1:]
                )
                encoded['input_ids'] = new_input_ids
                labels = encoded.get('labels')
                if labels is not None:
                    new_labels = (
                        labels[:audio_idx]
                        + [-100] * n_tokens
                        + labels[audio_idx + 1:]
                    )
                    encoded['labels'] = new_labels
        return encoded


register_template(
    QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplateFixed),
    exist_ok=True,
)


def _replace_language_model(model, llm_path: str, torch_dtype):
    qwen25 = Qwen2ForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch_dtype,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    state_dict = qwen25.model.state_dict()
    model.language_model.load_state_dict(state_dict, strict=True)
    del qwen25


def _replace_audio_tower(model, audio_encoder_path: str, torch_dtype):
    whisper = WhisperModel.from_pretrained(
        audio_encoder_path,
        torch_dtype=torch_dtype,
        device_map='cpu',
        low_cpu_mem_usage=True,
    )
    whisper_state = whisper.encoder.state_dict()
    # Prefix Whisper encoder keys with audio_tower.
    prefixed = {{f'audio_tower.{{k}}': v for k, v in whisper_state.items()}}
    model.load_state_dict(prefixed, strict=False)
    del whisper


@register_model(
    ModelInfo(
        model_type='{derived}',
        model_dir='{base}',
        task_type='causal_lm',
    ),
    exist_ok=True,
)
def get_model_tokenizer_{derived}(
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

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    _replace_language_model(model, '{llm_path}', torch_dtype)
    _replace_audio_tower(model, '{audio_encoder_path}', torch_dtype)

    # Enable flash attention only for the LLM; audio tower keeps default impl.
    if hasattr(model.config, 'text_config'):
        model.config.text_config._attn_implementation = 'flash_attention_2'
        model.config.text_config.attn_implementation = 'flash_attention_2'

    for p in model.parameters():
        p.requires_grad = False

    return model, processor
'''
