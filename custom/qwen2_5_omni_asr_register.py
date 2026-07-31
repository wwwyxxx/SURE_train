import json
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import (AutoModelForCausalLM, Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration,
                          Qwen2_5OmniProcessor, WhisperForConditionalGeneration)

from swift.llm import (DatasetMeta, Model, ModelGroup, ModelMeta, MultiModelKeys, ResponsePreprocessor, register_dataset,
                       register_model_arch)
from swift.llm.model.utils import use_submodel_func
from swift.llm.model.register import register_model
from swift.llm.template.constant import MLLMTemplateType
from swift.utils import get_logger

logger = get_logger()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths to the component models used for from-scratch initialization.
_QWEN25_OMNI_CONFIG_DIR = os.path.join(_ROOT, 'model', 'Qwen2.5-Omni-7B')
_QWEN25_LLM_DIR = os.path.join(_ROOT, 'model', 'Qwen2.5-7B')
_WHISPER_ENCODER_DIR = os.path.join(_ROOT, 'model', 'whisper-large-v3')


def _resolve_project_path(path: str) -> str:
    if path is None:
        return path
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    candidates = [
        os.path.join(_ROOT, path),
        os.path.join(_ROOT, 'example', os.path.basename(path)),
        os.path.join('/workspace', path),
        os.path.join('/workspace', 'example', os.path.basename(path)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


class Qwen2_5OmniASRPreprocessor(ResponsePreprocessor):
    """Convert reprodata ASR rows into the chat format expected by Qwen2.5-Omni."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        text = row.get('txt') or row.get('text') or row.get('response')
        prompt = row.get('prompt') or 'Transcribe the speech to text.'
        if wav is None or text is None:
            return None
        wav = _resolve_project_path(wav)
        return {
            'messages': [
                {'role': 'user', 'content': f'{prompt} <audio>'},
                {'role': 'assistant', 'content': text},
            ],
            'audios': [wav],
        }


def _init_thinker_from_qwen25(thinker, qwen25_model) -> None:
    """Initialize the thinker LLM and lm_head from Qwen2.5-7B."""
    src_state = qwen25_model.state_dict()
    dst_state = thinker.state_dict()
    copied = 0
    skipped = 0

    mapping = {
        'model.embed_tokens.': 'model.embed_tokens.',
        'model.layers.': 'model.layers.',
        'model.norm.': 'model.norm.',
        'lm_head.': 'lm_head.',
    }

    with torch.no_grad():
        for src_name, src_tensor in src_state.items():
            dst_name = None
            for src_prefix, dst_prefix in mapping.items():
                if src_name.startswith(src_prefix):
                    dst_name = dst_prefix + src_name[len(src_prefix):]
                    break
            if dst_name is None or dst_name not in dst_state:
                skipped += 1
                continue
            dst_tensor = dst_state[dst_name]
            if dst_tensor.shape != src_tensor.shape:
                logger.warning(f'Shape mismatch for {src_name}: {src_tensor.shape} vs {dst_tensor.shape}')
                skipped += 1
                continue
            dst_tensor.copy_(src_tensor.to(dtype=dst_tensor.dtype))
            copied += 1

    logger.info(f'Initialized thinker from Qwen2.5-7B: copied={copied}, skipped={skipped}')


def _init_audio_tower_from_whisper(audio_tower, whisper_model) -> None:
    """Initialize the audio encoder body from Whisper-large-v3; keep adapter/proj random."""
    whisper_encoder = whisper_model.model.encoder
    src_state = whisper_encoder.state_dict()
    dst_state = audio_tower.state_dict()
    copied = 0
    skipped = 0

    mapping = {
        'conv1.': 'conv1.',
        'conv2.': 'conv2.',
        'embed_positions.weight': 'positional_embedding.positional_embedding',
        'layers.': 'layers.',
        'layer_norm.': 'ln_post.',
    }

    with torch.no_grad():
        for src_name, src_tensor in src_state.items():
            dst_name = None
            for src_prefix, dst_prefix in mapping.items():
                if src_name.startswith(src_prefix) or src_name == src_prefix:
                    if src_name == src_prefix:
                        dst_name = dst_prefix
                    else:
                        dst_name = dst_prefix + src_name[len(src_prefix):]
                    break
            if dst_name is None or dst_name not in dst_state:
                skipped += 1
                continue
            dst_tensor = dst_state[dst_name]
            if dst_tensor.shape != src_tensor.shape:
                logger.warning(f'Shape mismatch for {src_name}: {src_tensor.shape} vs {dst_tensor.shape}')
                skipped += 1
                continue
            dst_tensor.copy_(src_tensor.to(dtype=dst_tensor.dtype))
            copied += 1

    logger.info(f'Initialized audio_tower encoder from Whisper: copied={copied}, skipped={skipped}')


def _freeze_for_asr(model: Qwen2_5OmniForConditionalGeneration) -> None:
    """Freeze the LLM and the Whisper-derived audio encoder body; train adapter + lm_head."""
    for _, p in model.named_parameters():
        p.requires_grad = False

    # Trainable parts:
    #   - lm_head (text prediction head)
    #   - audio_tower adapter: proj, avg_pooler, audio_bos_eos_token
    trainable_prefixes = [
        'thinker.lm_head.',
        'thinker.audio_tower.proj.',
        'thinker.audio_tower.avg_pooler.',
        'thinker.audio_tower.audio_bos_eos_token.',
    ]
    trainable_prefixes = tuple(trainable_prefixes)
    for name, p in model.named_parameters():
        if name.startswith(trainable_prefixes):
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f'Frozen LLM and audio encoder body. Trainable params: {trainable:,} / {total:,}')


def _is_official_base_checkpoint(model_dir: str) -> bool:
    """Return True if ``model_dir`` is an official/base checkpoint, not a training output."""
    if not os.path.isdir(model_dir):
        return False
    # Training checkpoints contain swift-specific files.
    for marker in ('training_args.bin', 'trainer_state.json', 'args.json'):
        if os.path.exists(os.path.join(model_dir, marker)):
            return False
    # A checkpoint inside a versioned output directory is a training checkpoint.
    parent = os.path.basename(os.path.dirname(model_dir))
    if parent.startswith('v') and '-20' in parent:
        return False
    return True


def _resolve_checkpoint_dir(model_dir: str) -> Optional[str]:
    """Return ``model_dir`` itself if it holds saved weights, else the latest checkpoint-* subdir.

    Searches one level of versioned output directories (e.g. ``v5-.../checkpoint-200``) so
    that inference scripts can point to the parent output folder.

    Returns ``None`` when ``model_dir`` is the original model configuration directory and
    contains no trained weights.
    """
    if not os.path.isdir(model_dir):
        return None
    weight_names = ('model.safetensors', 'pytorch_model.bin',
                    'model.safetensors.index.json', 'pytorch_model.bin.index.json')

    # If model_dir is an official/base checkpoint (not a training output), do not treat it
    # as a trained checkpoint even though it contains weight files.
    if any(os.path.exists(os.path.join(model_dir, name)) for name in weight_names):
        if not _is_official_base_checkpoint(model_dir):
            return model_dir

    checkpoint_dirs = []
    for root, dirs, _ in os.walk(model_dir):
        # Only descend into immediate subdirectories (versioned run folders).
        depth = root.count(os.sep) - model_dir.count(os.sep)
        if depth > 1:
            dirs[:] = []
            continue
        for d in dirs:
            if d.startswith('checkpoint-'):
                sub = os.path.join(root, d)
                if any(os.path.exists(os.path.join(sub, n)) for n in weight_names):
                    checkpoint_dirs.append(sub)
    if not checkpoint_dirs:
        return None
    # Prefer larger checkpoint step number, fallback to newest mtime.
    def _sort_key(path: str):
        name = os.path.basename(path)
        try:
            step = int(name.split('-', 1)[1])
        except Exception:
            step = 0
        return (step, os.path.getmtime(path))
    return sorted(checkpoint_dirs, key=_sort_key)[-1]


def _ensure_spk_dict(model_dir: str) -> None:
    """Copy spk_dict.pt from the original model config dir if the checkpoint is missing it.

    Qwen2.5-Omni's ``from_pretrained`` always looks for ``spk_dict.pt`` even when audio
    output is disabled.  Checkpoints produced by swift only contain model weights, so we
    provide the auxiliary file from the original release configuration.
    """
    ckpt_dir = _resolve_checkpoint_dir(model_dir)
    if ckpt_dir is None:
        return
    dst = os.path.join(ckpt_dir, 'spk_dict.pt')
    if os.path.exists(dst):
        return
    src = os.path.join(_QWEN25_OMNI_CONFIG_DIR, 'spk_dict.pt')
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        logger.info(f'Copied spk_dict.pt to checkpoint: {dst}')


def get_model_tokenizer_qwen2_5_omni_asr(
    model_dir: str,
    model_info: Any,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs,
):
    """Load Qwen2.5-Omni with thinker initialized from Qwen2.5-7B and audio encoder from Whisper.

    During training the model is initialized from scratch.  During inference ``model_dir`` points
    to a training checkpoint and the saved weights are loaded directly.
    """
    config = Qwen2_5OmniConfig.from_pretrained(_QWEN25_OMNI_CONFIG_DIR, trust_remote_code=True)
    config.enable_audio_output = False

    processor = Qwen2_5OmniProcessor.from_pretrained(_QWEN25_OMNI_CONFIG_DIR, trust_remote_code=True)

    if not load_model:
        return None, processor

    torch_dtype = getattr(model_info, 'torch_dtype', None) or torch.bfloat16

    # Apply attention implementation requested by swift CLI (e.g. flash_attn).
    attn_impl = kwargs.get('attn_impl') or getattr(model_info, 'attn_impl', None)
    if attn_impl:
        from swift.llm.model.utils import AttnImpl
        AttnImpl.update_attn_impl(config, attn_impl)

    resolved_ckpt = _resolve_checkpoint_dir(model_dir)

    if resolved_ckpt is not None:
        logger.info(f'Loading trained checkpoint weights from {resolved_ckpt}')
        _ensure_spk_dict(model_dir)
        # Respect swift's device_map for inference; training from-scratch keeps CPU init.
        device_map = model_kwargs.get('device_map', 'cpu')
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            resolved_ckpt,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2' if AttnImpl.to_use_flash_attn(attn_impl) else None,
        )
    else:
        # Load with the modified config so only the thinker is instantiated; talker/token2wav weights
        # in the checkpoint are ignored automatically.
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            _QWEN25_OMNI_CONFIG_DIR,
            config=config,
            torch_dtype=torch_dtype,
            device_map='cpu',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2' if AttnImpl.to_use_flash_attn(attn_impl) else None,
        )

        # Initialize thinker from Qwen2.5-7B.
        logger.info('Loading Qwen2.5-7B for thinker initialization...')
        qwen25_model = AutoModelForCausalLM.from_pretrained(
            _QWEN25_LLM_DIR,
            torch_dtype=torch_dtype,
            device_map='cpu',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _init_thinker_from_qwen25(model.thinker, qwen25_model)
        del qwen25_model
        torch.cuda.empty_cache()

        # Initialize audio encoder from Whisper-large-v3.
        logger.info('Loading Whisper-large-v3 for audio encoder initialization...')
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            _WHISPER_ENCODER_DIR,
            torch_dtype=torch_dtype,
            device_map='cpu',
            low_cpu_mem_usage=True,
        )
        _init_audio_tower_from_whisper(model.thinker.audio_tower, whisper_model)
        del whisper_model
        torch.cuda.empty_cache()

    _freeze_for_asr(model)
    use_submodel_func(model, 'thinker')
    model_info.config = model.config
    return model, processor


# Register a model arch that marks this as an omni model for swift's bookkeeping.
register_model_arch(
    MultiModelKeys(
        'qwen2_5_omni_asr',
        language_model=['thinker.model', 'thinker.lm_head'],
        vision_tower=['thinker.audio_tower', 'thinker.visual'],
        aligner=['thinker.audio_tower.proj', 'thinker.visual.merger'],
        generator=[],
    ))

register_model(
    ModelMeta(
        model_type='qwen2_5_omni_asr',
        model_groups=[ModelGroup([Model(model_path=_QWEN25_OMNI_CONFIG_DIR)])],
        template=MLLMTemplateType.qwen2_5_omni,
        get_function=get_model_tokenizer_qwen2_5_omni_asr,
        model_arch='qwen2_5_omni_asr',
        architectures=['Qwen2_5OmniForConditionalGeneration'],
        is_multimodal=True,
        tags=['audio', 'asr'],
    ))


register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'outputs', 'qwen2.5_omni', 'reprodata_asr_single_overfit1_local.jsonl'),
        dataset_name='reprodata_asr_single_overfit1_local',
        preprocess_func=Qwen2_5OmniASRPreprocessor(),
    ),
    exist_ok=True)


register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data', 'combined_asr_local.jsonl'),
        dataset_name='combined_asr_local',
        preprocess_func=Qwen2_5OmniASRPreprocessor(),
    ),
    exist_ok=True)


# Local-path versions of the reprodata ASR datasets used when the original audio paths
# under /hpc_stor03 are not readable inside the training container.
register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'outputs', 'qwen2.5_omni', 'reprodata_asr_single_overfit100_local.jsonl'),
        dataset_name='reprodata_asr_single_overfit100_local',
        preprocess_func=Qwen2_5OmniASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'outputs', 'qwen2.5_omni', 'reprodata_asr_overfit100_local.jsonl'),
        dataset_name='reprodata_asr_overfit100_local',
        preprocess_func=Qwen2_5OmniASRPreprocessor(),
    ),
    exist_ok=True)

# Register the reprodata ASR datasets with original paths if available.
register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data', 'reprodata_asr_overfit100.jsonl'),
        dataset_name='reprodata_asr_overfit100',
        preprocess_func=Qwen2_5OmniASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'example', 'ASR_BAC009S0002W0263_overfit100.jsonl'),
        dataset_name='asr_one_overfit100',
        preprocess_func=Qwen2_5OmniASRPreprocessor(),
    ),
    exist_ok=True)


# Aishell-1 test set with local audio paths for inference inside the container.
register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'outputs', 'qwen2.5_omni', 'aishell1-test_ASR_infer_local.jsonl'),
        dataset_name='aishell1_test_asr_infer_local',
        preprocess_func=Qwen2_5OmniASRPreprocessor(),
    ),
    exist_ok=True)

# Single-sample overfit dataset used for the first verification stage.
_SINGLE_OVERFIT_PATH = os.path.join(_ROOT, 'outputs', 'qwen2.5_omni', 'reprodata_asr_single_overfit100.jsonl')
if os.path.exists(_SINGLE_OVERFIT_PATH):
    register_dataset(
        DatasetMeta(
            dataset_path=_SINGLE_OVERFIT_PATH,
            dataset_name='reprodata_asr_single_overfit100',
            preprocess_func=Qwen2_5OmniASRPreprocessor(),
        ),
        exist_ok=True)
else:
    # Fallback path that may be created manually.
    _SINGLE_OVERFIT_PATH_FALLBACK = os.path.join(_ROOT, 'data', 'reprodata_asr_single_overfit100.jsonl')
    if os.path.exists(_SINGLE_OVERFIT_PATH_FALLBACK):
        register_dataset(
            DatasetMeta(
                dataset_path=_SINGLE_OVERFIT_PATH_FALLBACK,
                dataset_name='reprodata_asr_single_overfit100',
                preprocess_func=Qwen2_5OmniASRPreprocessor(),
            ),
            exist_ok=True)
