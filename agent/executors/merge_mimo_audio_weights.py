#!/usr/bin/env python3
"""Build a full MiMoAudioForCausalLM by initializing LLM backbone from MiMo-7B-Base
and randomly initializing the audio components (patch encoder/decoder).

This implements option B of the user decision: use MiMo-7B-Base as backbone.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

# Import MiMo-Audio modeling code
SCRIPT_DIR = Path(__file__).resolve().parent
MIMO_AUDIO_SRC = SCRIPT_DIR.parents[1] / "MiMo-Audio" / "src"
sys.path.insert(0, str(MIMO_AUDIO_SRC))
from mimo_audio.modeling_mimo_audio import MiMoAudioArguments, MiMoAudioConfig, MiMoAudioForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-path", required=True, help="Path to MiMo-7B-Base (bare LLM)")
    parser.add_argument("--audio-config-path", required=True, help="Path to MiMo-Audio-7B-Base config dir")
    parser.add_argument("--output-path", required=True, help="Where to save the merged full model")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    llm_path = Path(args.llm_path).resolve()
    audio_config_path = Path(args.audio_config_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] LLM backbone: {llm_path}")
    print(f"[INFO] Audio config source: {audio_config_path}")
    print(f"[INFO] Output path: {output_path}")

    # Load tokenizer from LLM path and add special tokens required by MiMo-Audio
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(llm_path), trust_remote_code=True)
    special_tokens = [
        "<|sosp|>", "<|eosp|>", "<|empty|>", "<|Human|>",
        "<|SpeechLM|>", "<|sostm|>", "<|eostm|>", "<|eot|>",
    ]
    added = []
    for token in special_tokens:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
            added.append(token)
    if added:
        print(f"[INFO] Added special tokens: {added}")
    else:
        print("[INFO] All special tokens already present.")

    # Save tokenizer to output path
    tokenizer.save_pretrained(str(output_path))

    # Build MiMoAudioArguments from tokenizer
    mimo_args = MiMoAudioArguments(
        model_name_or_path=str(output_path),
        sosp_idx=tokenizer.convert_tokens_to_ids("<|sosp|>"),
        eosp_idx=tokenizer.convert_tokens_to_ids("<|eosp|>"),
        empty_idx=tokenizer.convert_tokens_to_ids("<|empty|>"),
        sostm_idx=tokenizer.convert_tokens_to_ids("<|sostm|>"),
        eostm_idx=tokenizer.convert_tokens_to_ids("<|eostm|>"),
        eot_idx=tokenizer.convert_tokens_to_ids("<|eot|>"),
    )

    # Load MiMo-Audio config
    config_path = audio_config_path / "config.json"
    print(f"[INFO] Loading MiMo-Audio config from {config_path}")
    config = MiMoAudioConfig.from_pretrained(str(audio_config_path))
    config.torch_dtype = args.dtype

    # Save config to output path
    config.save_pretrained(str(output_path))

    # Initialize full MiMoAudio model (random init for audio components)
    print("[INFO] Initializing MiMoAudioForCausalLM...")
    model = MiMoAudioForCausalLM(config, mimo_args)
    model = model.to(dtype=dtype)

    # Load MiMo-7B-Base LLM backbone weights
    print(f"[INFO] Loading LLM weights from {llm_path}...")
    if (llm_path / "model.safetensors.index.json").exists():
        from safetensors import safe_open
        index = json.load(open(llm_path / "model.safetensors.index.json"))
        weight_map = index["weight_map"]
        llm_state = {}
        loaded_files = set()
        for key, filename in weight_map.items():
            filepath = llm_path / filename
            if filepath not in loaded_files:
                print(f"[INFO] Loading {filepath}")
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        llm_state[k] = f.get_tensor(k)
                loaded_files.add(filepath)
    else:
        raise FileNotFoundError(f"No model.safetensors.index.json found in {llm_path}")

    # Copy LLM backbone weights into the full model
    model_state = model.state_dict()
    copied = 0
    skipped = []
    for key, tensor in llm_state.items():
        if key.startswith("model.mtp_layers"):
            skipped.append((key, "MTP layer not used in MiMo-Audio"))
            continue
        if key in model_state:
            if model_state[key].shape != tensor.shape:
                skipped.append((key, f"shape mismatch: {model_state[key].shape} vs {tensor.shape}"))
                continue
            model_state[key] = tensor.to(dtype=dtype)
            copied += 1
        else:
            skipped.append((key, "key not in MiMoAudio model"))

    print(f"[INFO] Copied {copied} tensors from LLM backbone.")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} tensors:")
        for key, reason in skipped[:20]:
            print(f"  - {key}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    # Load state dict back into model
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}) - these were randomly initialized:")
        for k in missing[:20]:
            print(f"  - {k}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}):")
        for k in unexpected[:20]:
            print(f"  - {k}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")

    # Save model in safetensors format
    print(f"[INFO] Saving merged model to {output_path}...")
    model.save_pretrained(str(output_path), safe_serialization=True, max_shard_size="5GB")

    print("[OK] Done.")
    print(f"[INFO] Output path: {output_path}")


if __name__ == "__main__":
    main()
