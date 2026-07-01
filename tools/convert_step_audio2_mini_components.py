#!/usr/bin/env python3
"""
Assemble a Step-Audio-2-mini checkpoint from three independent sources.

Usage:
    python3 tools/convert_step_audio2_mini_components.py \
        --qwen25-dir model/Qwen2.5-7B \
        --qwen2-audio-dir model/Qwen2-Audio-7B \
        --step-audio-dir model/Step-Audio-2-mini \
        --output-dir model/Step-Audio-2-mini-assembled \
        --copy-overlapping-embeddings

What it does:
    1. Copy config/tokenizer/processor files from Step-Audio-2-mini (base).
    2. Load Step-Audio-2-mini's weights as the initial state (keeps adapter, lm_head, embed_tokens).
    3. Overwrite transformer backbone (model.layers.* + model.norm.*) with Qwen2.5-7B.
    4. Overwrite audio encoder (encoder.*) with Qwen2-Audio-7B's audio_tower, applying key mapping.
    5. Optionally copy the overlapping part of embed_tokens / lm_head from Qwen2.5-7B
       and keep the newly added Step-Audio-2-mini tokens randomly initialized.
    6. Save the assembled weights to model.safetensors (or sharded files if large).

Vocab overlap:
    - Qwen2.5-7B vocab_size=152064, added tokens end at ID 151664.
    - Step-Audio-2-mini vocab_size=158720, with extra tokens starting at ID 151665.
    - IDs 0..151664 are shared, so only those rows/columns are copied from Qwen2.5-7B.
    - The overlap size is inferred from Qwen2.5-7B's tokenizer_config.json automatically.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_safetensors_state_dict(path: str) -> Dict[str, Any]:
    from safetensors import safe_open

    state = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key)
    return state


def _load_torch_state_dict(path: str) -> Dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def discover_checkpoint_files(path: str) -> Tuple[List[str], Optional[str]]:
    p = Path(path)
    candidates = []
    index_file = None

    safetensors_index = p / "model.safetensors.index.json"
    pytorch_index = p / "pytorch_model.bin.index.json"
    single_safetensors = p / "model.safetensors"
    single_pytorch = p / "pytorch_model.bin"

    if safetensors_index.exists():
        index_file = str(safetensors_index)
        index = load_json(index_file)
        weight_map = index.get("weight_map", {})
        seen = set()
        for filename in weight_map.values():
            filepath = p / filename
            if str(filepath) not in seen and filepath.exists():
                candidates.append(str(filepath))
                seen.add(str(filepath))
    elif pytorch_index.exists():
        index_file = str(pytorch_index)
        index = load_json(index_file)
        weight_map = index.get("weight_map", {})
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
    if os.path.isfile(path):
        if path.endswith(".safetensors"):
            return _load_safetensors_state_dict(path)
        if path.endswith(".bin") or path.endswith(".pt") or path.endswith(".pth"):
            return _load_torch_state_dict(path)
        raise ValueError(f"Unsupported checkpoint file: {path}")

    files, _ = discover_checkpoint_files(path)
    if not files:
        raise FileNotFoundError(f"No checkpoint weights found in {path}")

    state = {}
    for filepath in files:
        if filepath.endswith(".safetensors"):
            piece = _load_safetensors_state_dict(filepath)
        else:
            piece = _load_torch_state_dict(filepath)
        overlap = set(state.keys()) & set(piece.keys())
        if overlap:
            raise ValueError(f"Duplicate keys loading {filepath}: {list(overlap)[:5]}")
        state.update(piece)
    return state


def build_audio_tower_mapping(source_keys: List[str]) -> Dict[str, str]:
    """Map Qwen2-Audio-7B audio_tower.* keys to Step-Audio-2-mini encoder.* keys."""
    mapping = {}
    for k in source_keys:
        new = None
        if k.startswith("audio_tower.conv1."):
            new = "encoder.conv1." + k.split(".", 2)[2]
        elif k.startswith("audio_tower.conv2."):
            new = "encoder.conv2." + k.split(".", 2)[2]
        elif k.startswith("audio_tower.embed_positions."):
            new = "encoder.positional_embedding." + k.split(".", 2)[2]
        elif k.startswith("audio_tower.layer_norm."):
            new = "encoder.after_norm." + k.split(".", 2)[2]
        elif re.match(r"audio_tower\.layers\.(\d+)\.fc1\.(.*)", k):
            m = re.match(r"audio_tower\.layers\.(\d+)\.fc1\.(.*)", k)
            new = f"encoder.blocks.{m.group(1)}.mlp.0.{m.group(2)}"
        elif re.match(r"audio_tower\.layers\.(\d+)\.fc2\.(.*)", k):
            m = re.match(r"audio_tower\.layers\.(\d+)\.fc2\.(.*)", k)
            new = f"encoder.blocks.{m.group(1)}.mlp.2.{m.group(2)}"
        elif re.match(r"audio_tower\.layers\.(\d+)\.final_layer_norm\.(.*)", k):
            m = re.match(r"audio_tower\.layers\.(\d+)\.final_layer_norm\.(.*)", k)
            new = f"encoder.blocks.{m.group(1)}.mlp_ln.{m.group(2)}"
        elif re.match(r"audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight", k):
            m = re.match(r"audio_tower\.layers\.(\d+)\.self_attn\.k_proj\.weight", k)
            new = f"encoder.blocks.{m.group(1)}.attn.key.weight"
        elif re.match(r"audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.(.*)", k):
            m = re.match(r"audio_tower\.layers\.(\d+)\.self_attn\.out_proj\.(.*)", k)
            new = f"encoder.blocks.{m.group(1)}.attn.out.{m.group(2)}"
        elif re.match(r"audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.(.*)", k):
            m = re.match(r"audio_tower\.layers\.(\d+)\.self_attn\.q_proj\.(.*)", k)
            new = f"encoder.blocks.{m.group(1)}.attn.query.{m.group(2)}"
        elif re.match(r"audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.(.*)", k):
            m = re.match(r"audio_tower\.layers\.(\d+)\.self_attn\.v_proj\.(.*)", k)
            new = f"encoder.blocks.{m.group(1)}.attn.value.{m.group(2)}"
        elif re.match(r"audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.(.*)", k):
            m = re.match(r"audio_tower\.layers\.(\d+)\.self_attn_layer_norm\.(.*)", k)
            new = f"encoder.blocks.{m.group(1)}.attn_ln.{m.group(2)}"
        else:
            raise ValueError(f"Unmapped audio_tower key: {k}")
        mapping[k] = new
    return mapping


def infer_overlap_vocab_size(qwen25_dir: str) -> int:
    """Infer the number of shared token IDs between Qwen2.5-7B and Step-Audio-2-mini.

    Qwen2.5-7B added tokens end at 151664, so IDs 0..151664 are shared.
    This is read from tokenizer_config.json['added_tokens_decoder'].
    """
    config_path = Path(qwen25_dir) / "tokenizer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"tokenizer_config.json not found in {qwen25_dir}")

    cfg = load_json(str(config_path))
    added = cfg.get("added_tokens_decoder", {})
    if not added:
        # Fallback: assume full vocab overlap
        return cfg.get("vocab_size", 152064)

    max_id = max(int(k) for k in added.keys())
    return max_id + 1


def merge_embed_weights(base_tensor, source_tensor, overlap_size: int):
    """Copy the first overlap_size rows from source to base."""
    import torch

    if source_tensor.shape[0] < overlap_size:
        raise ValueError(
            f"source embed has only {source_tensor.shape[0]} rows, "
            f"but overlap_size={overlap_size}"
        )
    if base_tensor.shape[0] < overlap_size:
        raise ValueError(
            f"base embed has only {base_tensor.shape[0]} rows, "
            f"but overlap_size={overlap_size}"
        )

    base_tensor[:overlap_size, :] = source_tensor[:overlap_size, :].to(base_tensor.dtype)
    return base_tensor


def merge_lm_head_weights(base_tensor, source_tensor, overlap_size: int):
    """Copy the first overlap_size rows from source to base.

    Both embed_tokens.weight and lm_head.weight are shaped [vocab_size, hidden_size]
    in the Qwen2 / StepAudio2 checkpoints.
    """
    import torch

    if source_tensor.shape[0] < overlap_size:
        raise ValueError(
            f"source lm_head has only {source_tensor.shape[0]} rows, "
            f"but overlap_size={overlap_size}"
        )
    if base_tensor.shape[0] < overlap_size:
        raise ValueError(
            f"base lm_head has only {base_tensor.shape[0]} rows, "
            f"but overlap_size={overlap_size}"
        )

    base_tensor[:overlap_size, :] = source_tensor[:overlap_size, :].to(base_tensor.dtype)
    return base_tensor


def copy_base_files(base_dir: str, output_dir: str):
    """Copy non-weight auxiliary files from base checkpoint."""
    base = Path(base_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    skip_suffixes = (".bin", ".safetensors", ".pt", ".pth")
    skip_names = {
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    }

    copied = []
    for src_file in base.iterdir():
        if not src_file.is_file():
            continue
        if src_file.suffix in skip_suffixes:
            continue
        if src_file.name in skip_names:
            continue
        shutil.copy2(src_file, out / src_file.name)
        copied.append(src_file.name)
    return copied


def save_state_dict_sharded(
    output_dir: str,
    state: Dict[str, Any],
    max_size_bytes: int = 5 * (1024 ** 3),
) -> Tuple[List[str], Dict[str, str]]:
    """Save state dict as sharded safetensors, similar to HF format."""
    from safetensors.torch import save_file
    import torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Simple greedy bin packing by tensor size
    bins: List[List[str]] = []
    bin_sizes: List[int] = []

    for key in sorted(state.keys()):
        tensor = state[key]
        size = tensor.numel() * tensor.element_size()
        placed = False
        for i, bin_keys in enumerate(bins):
            if bin_sizes[i] + size <= max_size_bytes:
                bin_keys.append(key)
                bin_sizes[i] += size
                placed = True
                break
        if not placed:
            bins.append([key])
            bin_sizes.append(size)

    weight_map = {}
    for i, bin_keys in enumerate(bins):
        bin_state = {k: state[k] for k in bin_keys}
        filename = f"model-{i + 1:05d}-of-{len(bins):05d}.safetensors"
        save_file(bin_state, str(out / filename))
        for k in bin_keys:
            weight_map[k] = filename

    return [str(out / f"model-{i + 1:05d}-of-{len(bins):05d}.safetensors") for i in range(len(bins))], weight_map


def main():
    parser = argparse.ArgumentParser(
        description="Assemble Step-Audio-2-mini checkpoint from Qwen2.5-7B + Qwen2-Audio-7B"
    )
    parser.add_argument("--qwen25-dir", required=True, help="Path to Qwen2.5-7B checkpoint")
    parser.add_argument("--qwen2-audio-dir", required=True, help="Path to Qwen2-Audio-7B checkpoint")
    parser.add_argument("--step-audio-dir", required=True, help="Path to Step-Audio-2-mini base checkpoint")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--copy-overlapping-embeddings",
        action="store_true",
        help="Copy shared token embeddings (0..151664) from Qwen2.5-7B; keep newly added tokens random.",
    )
    parser.add_argument(
        "--overlap-vocab-size",
        type=int,
        default=None,
        help="Override the number of shared token IDs. Defaults to inferred from Qwen2.5-7B tokenizer_config.",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help="Max shard size, e.g. 5GB, 2GB",
    )
    args = parser.parse_args()

    # Parse shard size
    max_size_str = args.max_shard_size.upper().replace("GB", "").replace("MB", "").strip()
    if "GB" in args.max_shard_size.upper():
        max_size_bytes = int(float(max_size_str) * (1024 ** 3))
    else:
        max_size_bytes = int(float(max_size_str) * (1024 ** 2))

    print(f"[1/6] Copying auxiliary files from {args.step_audio_dir} ...")
    copied = copy_base_files(args.step_audio_dir, args.output_dir)
    print(f"      Copied {len(copied)} files: {copied[:5]}{'...' if len(copied) > 5 else ''}")

    print(f"[2/6] Loading base weights from {args.step_audio_dir} ...")
    base_state = load_state_dict_from_path(args.step_audio_dir)
    print(f"      Base keys: {len(base_state)}")

    print(f"[3/6] Loading Qwen2.5-7B backbone from {args.qwen25_dir} ...")
    qwen25_state = load_state_dict_from_path(args.qwen25_dir)
    qwen25_backbone_keys = [k for k in qwen25_state if k.startswith("model.layers.") or k.startswith("model.norm.")]
    print(f"      Backbone keys to copy: {len(qwen25_backbone_keys)}")

    # Sanity check: all target keys exist in base
    missing_in_base = [k for k in qwen25_backbone_keys if k not in base_state]
    if missing_in_base:
        print(f"[WARN] {len(missing_in_base)} backbone keys not in base (will be added): {missing_in_base[:5]}")

    for k in qwen25_backbone_keys:
        base_state[k] = qwen25_state[k]
    print("      Backbone overwritten.")

    print(f"[4/6] Loading Qwen2-Audio-7B encoder from {args.qwen2_audio_dir} ...")
    qwen2_audio_state = load_state_dict_from_path(args.qwen2_audio_dir)
    audio_keys = [k for k in qwen2_audio_state if k.startswith("audio_tower")]
    print(f"      audio_tower keys: {len(audio_keys)}")

    mapping = build_audio_tower_mapping(audio_keys)
    for src_key in audio_keys:
        tgt_key = mapping[src_key]
        base_state[tgt_key] = qwen2_audio_state[src_key]
    print(f"      Encoder remapped to {len(mapping)} target keys.")

    overlap_size = None
    if args.copy_overlapping_embeddings:
        print(f"[5/6] Merging overlapping embed_tokens / lm_head from {args.qwen25_dir} ...")
        overlap_size = args.overlap_vocab_size or infer_overlap_vocab_size(args.qwen25_dir)
        print(f"      Shared token IDs: 0..{overlap_size - 1} (overlap_size={overlap_size})")

        embed_key = "model.embed_tokens.weight"
        if embed_key in qwen25_state and embed_key in base_state:
            base_state[embed_key] = merge_embed_weights(
                base_state[embed_key], qwen25_state[embed_key], overlap_size
            )
            print(f"      {embed_key}: copied first {overlap_size} rows from Qwen2.5-7B")
        else:
            print(f"      [WARN] {embed_key} not found in one of the checkpoints; skipping")

        lm_head_key = "lm_head.weight"
        if lm_head_key in qwen25_state and lm_head_key in base_state:
            base_state[lm_head_key] = merge_lm_head_weights(
                base_state[lm_head_key], qwen25_state[lm_head_key], overlap_size
            )
            print(f"      {lm_head_key}: copied first {overlap_size} rows from Qwen2.5-7B")
        else:
            print(f"      [WARN] {lm_head_key} not found in one of the checkpoints; skipping")
    else:
        print("[5/6] Skipping embed_tokens / lm_head merge (use --copy-overlapping-embeddings to enable)")

    # Verify expected target keys are present
    expected_prefixes = ["model.layers.", "model.norm.", "encoder.", "adapter.", "model.embed_tokens", "lm_head"]
    for prefix in expected_prefixes:
        count = sum(1 for k in base_state if k.startswith(prefix))
        print(f"      {prefix}*: {count} keys")

    print(f"[6/6] Saving assembled checkpoint to {args.output_dir} ...")
    shard_files, weight_map = save_state_dict_sharded(args.output_dir, base_state, max_size_bytes)
    print(f"      Shards: {len(shard_files)}")

    index_path = Path(args.output_dir) / "model.safetensors.index.json"
    save_json(str(index_path), {"metadata": {"total_size": sum(os.path.getsize(f) for f in shard_files)}, "weight_map": weight_map})
    print(f"      Index saved to {index_path}")

    report_path = Path(args.output_dir) / "assembly_report.json"
    report = {
        "output_dir": os.path.abspath(args.output_dir),
        "qwen25_dir": os.path.abspath(args.qwen25_dir),
        "qwen2_audio_dir": os.path.abspath(args.qwen2_audio_dir),
        "step_audio_dir": os.path.abspath(args.step_audio_dir),
        "total_keys": len(base_state),
        "qwen25_backbone_keys": len(qwen25_backbone_keys),
        "audio_encoder_keys": len(audio_keys),
        "copy_overlapping_embeddings": args.copy_overlapping_embeddings,
        "overlap_vocab_size": overlap_size,
        "shards": [os.path.basename(f) for f in shard_files],
    }
    save_json(str(report_path), report)
    print(f"      Report saved to {report_path}")
    print("[OK] Done.")


if __name__ == "__main__":
    main()
