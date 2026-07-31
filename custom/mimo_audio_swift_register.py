import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from swift.llm import (DatasetMeta, Model, ModelGroup, ModelMeta, MultiModelKeys, Template,
                       TemplateMeta, register_dataset, register_model, register_model_arch,
                       register_template)
from swift.utils import get_logger


logger = get_logger()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MIMO_AUDIO_SRC = os.path.join(_ROOT, 'MiMo-Audio', 'src')
if _MIMO_AUDIO_SRC not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_SRC)

from mimo_audio.modeling_mimo_audio import MiMoAudioArguments, MiMoAudioForCausalLM  # noqa: E402
from mimo_audio.process_speechdata import InputSegment  # noqa: E402
from mimo_audio_tokenizer import MiMoAudioTokenizer  # noqa: E402


def _resolve_project_path(path: str) -> str:
    if path is None:
        return path
    if os.path.exists(path):
        return path
    basename = os.path.basename(path)
    candidates = [
        os.path.join(_ROOT, path),
        os.path.join(_ROOT, 'example', basename),
        os.path.join('/workspace', path),
        os.path.join('/workspace', 'example', basename),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


# ---------------------------------------------------------------------------
# Special tokens used by MiMo-Audio
# ---------------------------------------------------------------------------
_MIMO_AUDIO_SPECIAL_TOKENS = [
    '<|sosp|>', '<|eosp|>', '<|empty|>', '<|Human|>',
    '<|SpeechLM|>', '<|sostm|>', '<|eostm|>', '<|eot|>',
]


def _ensure_mimo_special_tokens(tokenizer):
    added = False
    for token in _MIMO_AUDIO_SPECIAL_TOKENS:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
            added = True
    if added:
        logger.info('[MiMo-Audio] Added special tokens to tokenizer.')
    return tokenizer


def _build_mimo_args(tokenizer, model_name_or_path: str) -> MiMoAudioArguments:
    return MiMoAudioArguments(
        model_name_or_path=model_name_or_path,
        sosp_idx=tokenizer.convert_tokens_to_ids('<|sosp|>'),
        eosp_idx=tokenizer.convert_tokens_to_ids('<|eosp|>'),
        empty_idx=tokenizer.convert_tokens_to_ids('<|empty|>'),
        sostm_idx=tokenizer.convert_tokens_to_ids('<|sostm|>'),
        eostm_idx=tokenizer.convert_tokens_to_ids('<|eostm|>'),
        eot_idx=tokenizer.convert_tokens_to_ids('<|eot|>'),
    )


# ---------------------------------------------------------------------------
# Audio tokenizer helper (lazy singleton per process)
# ---------------------------------------------------------------------------
class _AudioTokenizerHelper:
    _instance = None

    @classmethod
    def get(cls, model_path: Optional[str] = None):
        if cls._instance is None:
            path = model_path or os.environ.get('MIMO_AUDIO_TOKENIZER', os.path.join(_ROOT, 'model/MiMo-Audio-Tokenizer'))
            logger.info(f'[MiMo-Audio] Loading audio tokenizer from {path}')
            cls._instance = MiMoAudioTokenizer.from_pretrained(path)
            cls._instance.eval().bfloat16().to('cuda:0')
        return cls._instance


def _get_mel_transform(cfg):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sampling_rate,
        n_fft=cfg.nfft,
        hop_length=cfg.hop_length,
        win_length=cfg.window_size,
        f_min=cfg.fmin,
        f_max=cfg.fmax,
        n_mels=cfg.n_mels,
        power=1.0,
        center=True,
    ).to('cuda:0')


def _encode_audio_to_list(wav_path: str, audio_channels: int = 8, group_size: int = 4) -> List[int]:
    """Encode a single audio file into RVQ tokens (list of ints) on GPU.

    This is called during dataset preprocessing in the main process, so it is
    safe to use CUDA. The resulting list is stored in the dataset and can be
    loaded by CPU dataloader workers without touching CUDA.
    """
    wav_path = _resolve_project_path(wav_path)
    audio_tokenizer = _AudioTokenizerHelper.get()
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    target_sr = audio_tokenizer.config.sampling_rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    device = next(audio_tokenizer.parameters()).device
    wav = wav.to(device)
    mel = torch.log(torch.clamp(_get_mel_transform(audio_tokenizer.config)(wav[None, :]), min=1e-7)).squeeze().transpose(0, 1)

    input_len = mel.size(0)
    segment_size = 6000
    input_len_seg = [segment_size] * (input_len // segment_size)
    if input_len % segment_size > 0:
        input_len_seg.append(input_len % segment_size)

    codes_list = []
    for features, lengths in zip(torch.split(mel, input_len_seg), [torch.tensor(x, device=device) for x in input_len_seg]):
        with torch.no_grad():
            codes, _ = audio_tokenizer.encoder.encode(
                input_features=features,
                input_lens=lengths[None],
                return_codes_only=True,
            )
        codes_list.append(codes)

    codes = torch.cat(codes_list, dim=-1)  # [num_quantizers, T]
    audio_codes = codes[:audio_channels].transpose(0, 1).detach().cpu()  # [T, audio_channels]

    # Pad to multiple of group_size
    num_timesteps = audio_codes.shape[0]
    if num_timesteps % group_size != 0:
        padding_needed = group_size - (num_timesteps % group_size)
        last_tokens = audio_codes[-1:, :]
        padding_tokens = last_tokens.repeat(padding_needed, 1)
        audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)
    return audio_codes.reshape(-1).tolist()  # [T * audio_channels]


# ---------------------------------------------------------------------------
# Dataset preprocessor
# ---------------------------------------------------------------------------
class MiMoAudioASRPreprocessor:
    """Convert raw jsonl rows into {messages, audio_tokens} for ms-swift."""

    def _process_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        text = row.get('txt') or row.get('text') or row.get('response')
        prompt = row.get('prompt') or 'Transcribe the speech to text.'
        if wav is None or text is None:
            return None

        # If the dataset has been pre-cached, wav points to a .pt file containing
        # flattened RVQ tokens. Load directly to avoid re-running the audio tokenizer.
        resolved = _resolve_project_path(wav)
        if isinstance(resolved, str) and resolved.endswith('.pt') and os.path.exists(resolved):
            audio_tokens = torch.load(resolved, map_location='cpu', weights_only=True).tolist()
        else:
            audio_tokens = _encode_audio_to_list(wav)

        return {
            'messages': [
                {'role': 'user', 'content': f'{prompt} <audio>'},
                {'role': 'assistant', 'content': text},
            ],
            'objects': {'audio_tokens': audio_tokens},
        }

    def __call__(self, dataset, *, num_proc: int = 1, load_from_cache_file: bool = False, strict: bool = False, **kwargs):
        return dataset.map(
            self._process_row,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            desc='MiMoAudioASRPreprocessor',
        )


# ---------------------------------------------------------------------------
# Model wrapper for ms-swift
# ---------------------------------------------------------------------------
class MiMoAudioSFTModel(MiMoAudioForCausalLM):
    """MiMo-Audio wrapper that exposes a training-compatible forward().

    The original MiMoAudioForCausalLM.forward() returns only the last-position
    logits for autoregressive sampling. This wrapper computes logits for every
    text-group position and applies cross-entropy only on text tokens.
    """

    def forward(
        self,
        input_ids: torch.LongTensor,          # [B, audio_channels+1, T]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T_groups]
        position_ids: Optional[torch.LongTensor] = None,  # [B, T_groups]
        labels: Optional[torch.LongTensor] = None,        # [B, T_groups]
        text_loss_mask: Optional[torch.Tensor] = None,    # [B, T_groups]
        **kwargs,
    ):
        embed_device = self.model.embed_tokens.weight.device
        if input_ids.device != embed_device:
            input_ids = input_ids.to(embed_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(embed_device)
            if position_ids is not None:
                position_ids = position_ids.to(embed_device)
            if labels is not None:
                labels = labels.to(embed_device)
            if text_loss_mask is not None:
                text_loss_mask = text_loss_mask.to(embed_device)
        inputs_embeds = self._prepare_input_embeds(input_ids)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state  # [B, T_groups, H]
        text_logits = self.lm_head(hidden_states)  # [B, T_groups, vocab_size]

        loss = None
        if labels is not None:
            # Ensure labels/mask live on the same device as logits (lm_head may be on a different GPU).
            labels = labels.to(text_logits.device)
            if text_loss_mask is not None:
                text_loss_mask = text_loss_mask.to(text_logits.device)
            # Standard causal shift: logits at position i predict labels at i+1.
            shift_logits = text_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = text_loss_mask[..., 1:].contiguous() if text_loss_mask is not None else None

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            if shift_mask is not None:
                losses = losses * shift_mask.view(-1).to(losses.dtype)
                loss = losses.sum() / (shift_mask.sum() + 1e-6)
            else:
                loss = losses.mean()

        return CausalLMOutputWithPast(
            loss=loss,
            logits=text_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _freeze_for_asr(model: MiMoAudioSFTModel) -> None:
    """Freeze the whole model except selected audio/text output components.

    By default only unfreeze speech_embeddings, input_local_transformer,
    speech_group_downcast and lm_head.  Override with the comma-separated env
    variable MIMO_AUDIO_TRAINABLE_COMPONENTS, e.g.
        MIMO_AUDIO_TRAINABLE_COMPONENTS=speech_group_downcast,lm_head
    """
    for _, p in model.named_parameters():
        p.requires_grad = False

    env = os.environ.get('MIMO_AUDIO_TRAINABLE_COMPONENTS')
    if env:
        train_prefixes = tuple(p.strip() + '.' for p in env.split(',') if p.strip())
    else:
        train_prefixes = (
            'speech_embeddings.',
            'input_local_transformer.',
            'speech_group_downcast.',
            'lm_head.',
        )

    trainable_count = 0
    trainable_names = []
    for name, p in model.named_parameters():
        if name.startswith(train_prefixes):
            p.requires_grad = True
            trainable_count += p.numel()
            trainable_names.append(name)
    logger.info(f'[MiMo-Audio] Frozen all but {train_prefixes}. Trainable params: {trainable_count / 1e6:.1f}M')
    logger.info(f'[MiMo-Audio] Trainable modules: {sorted(set(n.split(".")[0] for n in trainable_names))}')


def _load_llm_backbone(model: MiMoAudioSFTModel, llm_backbone_dir: str) -> None:
    """Overwrite the text LLM backbone weights from a plain MiMo-7B-Base checkpoint.

    Only keys that exist in both the base checkpoint and the audio model are copied.
    Audio-specific modules (speech embeddings, local transformers, speech heads) are
    left untouched, so their initialization comes from the audio base model.
    """
    logger.info(f'[MiMo-Audio] Overwriting LLM backbone from {llm_backbone_dir}')
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
    loaded_keys = set()
    for shard in shard_files:
        shard_path = os.path.join(llm_backbone_dir, shard)
        with safe_open(shard_path, framework='pt') as f:
            for key in f.keys():
                if key in model_keys:
                    param = model.get_parameter(key)
                    tensor = f.get_tensor(key).to(device=param.device, dtype=param.dtype)
                    param.data.copy_(tensor)
                    copied += 1
                    loaded_keys.add(key)
                else:
                    ignored += 1
    missing = len(model_keys - loaded_keys)
    logger.info(
        f'[MiMo-Audio] LLM backbone overwrite done: copied={copied}, '
        f'ignored={ignored}, missing={missing}'
    )


def get_model_tokenizer_mimo_audio(
    model_dir: str,
    model_info,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs,
):
    torch_dtype = model_info.torch_dtype or torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = _ensure_mimo_special_tokens(tokenizer)

    if not load_model:
        return None, tokenizer

    mimo_args = _build_mimo_args(tokenizer, model_dir)
    model = MiMoAudioSFTModel.from_pretrained(
        model_dir,
        args=mimo_args,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        **model_kwargs,
    )
    model.to(dtype=torch_dtype)

    llm_backbone_dir = os.environ.get('MIMO_AUDIO_LLM_BACKBONE')
    if llm_backbone_dir:
        _load_llm_backbone(model, llm_backbone_dir)

    if os.environ.get('MIMO_AUDIO_FREEZE_LLM', '1') != '0':
        _freeze_for_asr(model)

    # Optional: re-initialize specific trainable components from scratch.
    # This is useful for ablations that want to verify the audio base's
    # pre-trained lm_head / speech_group_downcast are not necessary.
    if os.environ.get('MIMO_AUDIO_RANDOM_INIT_LM_HEAD', '0') == '1':
        logger.info('[MiMo-Audio] Re-initializing lm_head with random weights.')
        model.lm_head.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    if os.environ.get('MIMO_AUDIO_RANDOM_INIT_SPEECH_GROUP_DOWNCAST', '0') == '1':
        logger.info('[MiMo-Audio] Re-initializing speech_group_downcast with random weights.')
        model.speech_group_downcast.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    _log_component_status(model, model_dir, llm_backbone_dir)
    return model, tokenizer


def _log_component_status(model: MiMoAudioSFTModel, audio_base_dir: str, llm_backbone_dir: Optional[str]) -> None:
    """Log each top-level component, its weight source, and whether it is trainable."""
    logger.info('[MiMo-Audio] Component status:')
    logger.info(f'{"Component":<35} | {"Source":<50} | {"Trainable":<9}')
    logger.info('-' * 100)
    for name, module in model.named_children():
        if not any(p is not None for p in module.parameters(recurse=True)):
            continue
        has_trainable = any(p.requires_grad for p in module.parameters(recurse=True))
        if name == 'model' and llm_backbone_dir:
            source = f'LLM backbone ({llm_backbone_dir})'
        else:
            source = f'audio base ({audio_base_dir})'
        logger.info(f'{name:<35} | {source:<50} | {"Yes" if has_trainable else "No":<9}')


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
class MiMoAudioTemplate(Template):
    support_padding_free = False
    placeholder_tokens = ['<audio>']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = _ensure_mimo_special_tokens(self.tokenizer)
        # Lazy init audio tokenizer only when on-demand audio encoding is needed.
        self._audio_tokenizer = None
        self.group_size = 4
        self.audio_channels = 8
        self.speech_zeroemb_idx = [1024, 1024, 128, 128, 128, 128, 128, 128]
        self.empty_idx = self.tokenizer.convert_tokens_to_ids('<|empty|>')
        self._mel_transform = None

    @property
    def audio_tokenizer(self):
        if self._audio_tokenizer is None:
            self._audio_tokenizer = _AudioTokenizerHelper.get()
        return self._audio_tokenizer

    def _encode_audio(self, wav_path: str) -> torch.Tensor:
        """Return audio RVQ codes of shape [T*audio_channels] on CPU.

        Kept for backward compatibility / inference use; training data is
        pre-encoded by the preprocessor so dataloader workers never call this.
        """
        return torch.tensor(_encode_audio_to_list(wav_path, self.audio_channels, self.group_size), dtype=torch.long)

    def _build_prompt_input_ids(
        self,
        prompt_text: str,
        audio_tokenized: torch.Tensor,
        response_text: str,
    ) -> torch.Tensor:
        """Build 2D input_ids [audio_channels+1, T]."""
        segments = [
            InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=self.speech_zeroemb_idx, text_zeroemb_idx=self.empty_idx),
            InputSegment(audio=audio_tokenized, speech_zeroemb_idx=self.speech_zeroemb_idx, text_zeroemb_idx=self.empty_idx),
            InputSegment(text=prompt_text, speech_zeroemb_idx=self.speech_zeroemb_idx, text_zeroemb_idx=self.empty_idx),
            InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=self.speech_zeroemb_idx, text_zeroemb_idx=self.empty_idx),
            InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=self.speech_zeroemb_idx, text_zeroemb_idx=self.empty_idx),
            InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=self.speech_zeroemb_idx, text_zeroemb_idx=self.empty_idx),
            InputSegment(text=response_text, speech_zeroemb_idx=self.speech_zeroemb_idx, text_zeroemb_idx=self.empty_idx),
            InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=self.speech_zeroemb_idx, text_zeroemb_idx=self.empty_idx),
        ]
        input_ids = torch.cat([
            seg.to_input_id(self.tokenizer, self.group_size, self.audio_channels)
            for seg in segments
        ], dim=1)
        return input_ids

    def _encode(self, inputs, return_length: bool = False, **kwargs) -> Dict[str, Any]:
        messages = inputs.messages
        if len(messages) != 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
            raise ValueError(f'MiMoAudioTemplate expects one user turn and one assistant turn: {messages}')

        prompt_text = messages[0]['content'].replace('<audio>', '').strip()
        response_text = messages[1]['content']

        audio_tokens = inputs.objects.get('audio_tokens') if inputs.objects else None
        if audio_tokens is None:
            audio_tokens = getattr(inputs, 'audio_tokens', None) or inputs.extra_kwargs.get('audio_tokens')
        if audio_tokens is not None:
            audio_tokenized = torch.tensor(audio_tokens, dtype=torch.long)
        elif inputs.audios:
            audio_tokenized = self._encode_audio(_resolve_project_path(inputs.audios[0]))
        else:
            raise ValueError('MiMoAudioTemplate requires audio_tokens or an audio path.')

        input_ids = self._build_prompt_input_ids(prompt_text, audio_tokenized, response_text)
        # input_ids: [audio_channels+1, T]

        # Extract text positions: one text token per group
        text_ids = input_ids[0, ::self.group_size].clone()  # [T_groups]
        T_groups = text_ids.shape[0]

        # Build labels and loss mask
        labels = text_ids.clone()
        text_loss_mask = torch.zeros(T_groups, dtype=torch.bool)

        # Mark assistant response positions as loss positions.
        # Heuristic: after the assistant header '<|im_start|>assistant\n', everything is response.
        assistant_header_ids = self.tokenizer('<|im_start|>assistant\n', add_special_tokens=False)['input_ids']
        assistant_header_ids = torch.tensor(assistant_header_ids, dtype=text_ids.dtype)

        # Find the position where assistant header starts
        for start in range(T_groups - len(assistant_header_ids) + 1):
            if torch.equal(text_ids[start:start + len(assistant_header_ids)], assistant_header_ids):
                # Loss starts after the assistant header and the fixed <think>\n\n</think>\n prefix.
                think_prefix = self.tokenizer('<think>\n\n</think>\n', add_special_tokens=False)['input_ids']
                loss_start = start + len(assistant_header_ids) + len(think_prefix)
                if loss_start < T_groups:
                    text_loss_mask[loss_start:] = True
                break

        # attention_mask and position_ids for T_groups
        attention_mask = torch.ones(T_groups, dtype=torch.bool)
        position_ids = torch.arange(T_groups, dtype=torch.long)

        # Keep labels unshifted: labels[i] == text_ids[i].  The model's forward()
        # internally shifts logits/labels for causal LM, and ms-swift's metric
        # compute_acc also shifts labels once. Returning unshifted labels makes the
        # two shifts align, so token_acc correctly measures response accuracy.
        labels[text_loss_mask == False] = -100

        encoded = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels,
            'text_loss_mask': text_loss_mask,
        }

        if return_length:
            encoded['length'] = T_groups
        return encoded

    def print_inputs(self, inputs: Dict[str, Any]) -> None:
        """Override to handle 2D input_ids [audio_channels+1, T]."""
        val = inputs.get('input_ids')
        if val is not None:
            logger.info(f'[INPUT_IDS] shape={tuple(val.shape)}, dtype={val.dtype}')
            if isinstance(val, torch.Tensor):
                text_ids = val[0, ::self.group_size].tolist()
                text_str = self.safe_decode(text_ids)
                logger.info(f'[INPUT] text_channel_decoded={text_str}')
        labels = inputs.get('labels')
        if labels is not None:
            logger.info(f'[LABELS] shape={tuple(labels.shape)}, dtype={labels.dtype}')
        loss_mask = inputs.get('text_loss_mask')
        if loss_mask is not None:
            logger.info(f'[TEXT_LOSS_MASK] sum={loss_mask.sum().item()}')

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pad_token_id = self.tokenizer.pad_token_id or 0
        max_len = max(item['input_ids'].shape[-1] for item in batch)
        max_groups = max(item['labels'].shape[0] for item in batch)
        if padding_to is not None:
            max_len = max(max_len, padding_to)
            max_groups = max(max_groups, padding_to)

        B = len(batch)
        C = batch[0]['input_ids'].shape[0]

        # Channel 0 is text; remaining channels are RVQ audio tokens. Padding the
        # audio channels with the text pad_token_id would produce indices far
        # beyond the audio embedding vocab sizes and trigger a CUDA assert.
        # Use each audio channel's empty (padding) index instead.
        padded_input_ids = torch.full((B, C, max_len), pad_token_id, dtype=torch.long)
        for c in range(1, C):
            padded_input_ids[:, c, :] = self.speech_zeroemb_idx[c - 1]

        padded_labels = torch.full((B, max_groups), -100, dtype=torch.long)
        padded_attention_mask = torch.zeros((B, max_groups), dtype=torch.bool)
        padded_position_ids = torch.full((B, max_groups), 0, dtype=torch.long)
        padded_text_loss_mask = torch.zeros((B, max_groups), dtype=torch.bool)

        for i, item in enumerate(batch):
            seq_len = item['input_ids'].shape[-1]
            groups = item['labels'].shape[0]
            padded_input_ids[i, :, :seq_len] = item['input_ids']
            padded_labels[i, :groups] = item['labels']
            padded_attention_mask[i, :groups] = item['attention_mask']
            padded_position_ids[i, :groups] = item['position_ids']
            padded_text_loss_mask[i, :groups] = item['text_loss_mask']

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'position_ids': padded_position_ids,
            'labels': padded_labels,
            'text_loss_mask': padded_text_loss_mask,
        }


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------
register_model_arch(
    MultiModelKeys(
        'mimo_audio',
        language_model=['model.embed_tokens', 'model.layers', 'model.norm'],
        aligner=['speech_embeddings', 'input_local_transformer', 'speech_group_downcast'],
        generator=['lm_head'],
    ))

register_template(
    TemplateMeta(
        template_type='mimo_audio',
        prefix=[],
        prompt=[],
        chat_sep=[],
        template_cls=MiMoAudioTemplate,
    ))

register_model(
    ModelMeta(
        model_type='mimo_audio',
        model_groups=[ModelGroup([Model(model_path=os.path.join(_ROOT, 'model/MiMo-Audio-7B-Base-merged-v3'))])],
        template='mimo_audio',
        get_function=get_model_tokenizer_mimo_audio,
        is_multimodal=True,
        model_arch='mimo_audio',
        tags=['audio', 'asr'],
    ))


# ---------------------------------------------------------------------------
# Dataset registrations
# ---------------------------------------------------------------------------
register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_aishell-1.jsonl'),
        dataset_name='combined_asr_aishell_1',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_aishell-1_cached.jsonl'),
        dataset_name='combined_asr_aishell_1_cached',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_aishell_1_cached_100.jsonl'),
        dataset_name='combined_asr_aishell_1_cached_100',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_aishell_1_cached_1000.jsonl'),
        dataset_name='combined_asr_aishell_1_cached_1000',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_zh_existing.jsonl'),
        dataset_name='reprodata_asr_zh_existing',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_zh_overfit10.jsonl'),
        dataset_name='reprodata_asr_zh_overfit10',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_zh_existing_aishell93.jsonl'),
        dataset_name='reprodata_asr_zh_existing_aishell93',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'example/ASR_BAC009S0002W0263_overfit100.jsonl'),
        dataset_name='mimo_audio_asr_overfit100',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_1000.jsonl'),
        dataset_name='reprodata_asr_1000',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/test_audio_30s_x100.jsonl'),
        dataset_name='test_audio_30s_x100',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_local_cached_100.jsonl'),
        dataset_name='combined_asr_local_cached_100',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_local_cached.jsonl'),
        dataset_name='combined_asr_local_cached',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)
