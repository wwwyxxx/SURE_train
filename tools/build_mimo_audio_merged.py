#!/usr/bin/env python3
"""Build a correct MiMo-Audio-7B-Base-merged checkpoint.

Audio components are taken from model/MiMo-Audio-7B-Base (the official audio base),
and the text LLM backbone (model.* + lm_head.*) is overwritten from
model/MiMo-7B-Base.
"""
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MIMO_AUDIO_SRC = os.path.join(_ROOT, 'MiMo-Audio', 'src')
if _MIMO_AUDIO_SRC not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_SRC)

from mimo_audio.modeling_mimo_audio import MiMoAudioArguments, MiMoAudioForCausalLM  # noqa: E402

LLM_BACKBONE_DIR = os.path.join(_ROOT, 'model', 'MiMo-7B-Base')
AUDIO_BASE_DIR = os.path.join(_ROOT, 'model', 'MiMo-Audio-7B-Base')
OUTPUT_DIR = os.path.join(_ROOT, 'model', 'MiMo-Audio-7B-Base-merged-v2')

_SPECIAL_TOKENS = [
    '<|sosp|>', '<|eosp|>', '<|empty|>', '<|Human|>',
    '<|SpeechLM|>', '<|sostm|>', '<|eostm|>', '<|eot|>',
]


def _ensure_special_tokens(tokenizer):
    for token in _SPECIAL_TOKENS:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
    return tokenizer


def _load_llm_backbone(model, llm_backbone_dir):
    print(f'[build] Overwriting LLM backbone from {llm_backbone_dir}')
    index_path = os.path.join(llm_backbone_dir, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            weight_map = json.load(f)['weight_map']
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = sorted([
            f for f in os.listdir(llm_backbone_dir)
            if f.endswith('.safetensors')
        ])

    model_keys = set(model.state_dict().keys())
    copied = 0
    ignored = 0
    for shard in shard_files:
        shard_path = os.path.join(llm_backbone_dir, shard)
        with safe_open(shard_path, framework='pt') as f:
            for key in f.keys():
                if key in model_keys:
                    param = model.get_parameter(key)
                    tensor = f.get_tensor(key).to(device=param.device, dtype=param.dtype)
                    param.data.copy_(tensor)
                    copied += 1
                else:
                    ignored += 1
    print(f'[build] LLM overwrite done: copied={copied}, ignored={ignored}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'[build] Loading tokenizer from {AUDIO_BASE_DIR}')
    tokenizer = AutoTokenizer.from_pretrained(AUDIO_BASE_DIR, trust_remote_code=True)
    tokenizer = _ensure_special_tokens(tokenizer)

    args = MiMoAudioArguments(
        model_name_or_path=AUDIO_BASE_DIR,
        sosp_idx=tokenizer.convert_tokens_to_ids('<|sosp|>'),
        eosp_idx=tokenizer.convert_tokens_to_ids('<|eosp|>'),
        empty_idx=tokenizer.convert_tokens_to_ids('<|empty|>'),
        sostm_idx=tokenizer.convert_tokens_to_ids('<|sostm|>'),
        eostm_idx=tokenizer.convert_tokens_to_ids('<|eostm|>'),
        eot_idx=tokenizer.convert_tokens_to_ids('<|eot|>'),
    )

    print(f'[build] Loading audio base model from {AUDIO_BASE_DIR}')
    model = MiMoAudioForCausalLM.from_pretrained(
        AUDIO_BASE_DIR,
        args=args,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
    )
    model.eval()

    _load_llm_backbone(model, LLM_BACKBONE_DIR)

    print(f'[build] Saving model to {OUTPUT_DIR}')
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size='5GB')
    print(f'[build] Saving tokenizer to {OUTPUT_DIR}')
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Preserve chat template if the old merged dir had one
    old_chat = os.path.join(_ROOT, 'model', 'MiMo-Audio-7B-Base-merged', 'chat_template.jinja')
    if os.path.exists(old_chat):
        shutil.copy(old_chat, os.path.join(OUTPUT_DIR, 'chat_template.jinja'))
        print(f'[build] Copied chat_template.jinja')

    print(f'[build] Done: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
