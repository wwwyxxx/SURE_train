import importlib.util
import json
import os
import sys
from types import MethodType
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from swift.llm import (
    DatasetMeta,
    Model,
    ModelGroup,
    ModelMeta,
    MultiModelKeys,
    ResponsePreprocessor,
    Template,
    TemplateMeta,
    register_dataset,
    register_model,
    register_model_arch,
    register_template,
)
from swift.utils import get_logger

logger = get_logger()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_TASU_SRC = os.path.join(_ROOT, 'ps-slm', 'Multitask')
if _TASU_SRC not in sys.path:
    sys.path.insert(0, _TASU_SRC)

# Ensure RANK exists so that TASU's setup helpers do not fail on logging.
os.environ.setdefault('RANK', '0')
os.environ.setdefault('WORLD_SIZE', '1')


def _resolve_project_path(path: str) -> str:
    if path is None or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    candidates = [
        os.path.join(_ROOT, path),
        os.path.join('/workspace', path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _load_tasu_source_module():
    """Load ps-slm/Multitask/model/ps-slm.py as a regular Python module."""
    module_path = os.path.join(_TASU_SRC, 'model', 'ps-slm.py')
    spec = importlib.util.spec_from_file_location('tasu_ps_slm', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['tasu_ps_slm'] = module
    spec.loader.exec_module(module)
    return module


from aispeech_asr_config import ModelConfig, TrainConfig

_tasu_module = _load_tasu_source_module()
setup_tokenizer = _tasu_module.setup_tokenizer
setup_llm = _tasu_module.setup_llm
setup_encoder = _tasu_module.setup_encoder
setup_encoder_projector = _tasu_module.setup_encoder_projector
slam_model_asr = _tasu_module.slam_model_asr


class _SenseVoiceFeatureExtractor:
    """Lazy singleton that loads SenseVoiceSmall once per process and exposes its frontend."""
    _instance = None

    def __new__(cls, encoder_path: str):
        if cls._instance is None:
            from model.SenseVoice import SenseVoiceSmall
            encoder, kwargs = SenseVoiceSmall.from_pretrained(encoder_path)
            frontend = kwargs.get('frontend')
            if frontend is None:
                raise RuntimeError('SenseVoice frontend not found in loaded model kwargs')
            cls._instance = (encoder, frontend)
        return cls._instance


def _extract_sensevoice_fbank(audio_path: str, encoder_path: str):
    """Extract SenseVoice fbank features [T, 560] and length T."""
    import soundfile as sf
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

    audio_path = _resolve_project_path(audio_path)
    waveform, sr = sf.read(audio_path, dtype='float32')
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != 16000:
        try:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        except Exception:
            pass

    _, frontend = _SenseVoiceFeatureExtractor(encoder_path)

    audio_list = load_audio_text_image_video(
        [waveform],
        fs=frontend.fs,
        audio_fs=16000,
        data_type='sound',
    )
    input_features, input_feature_length = extract_fbank(
        audio_list,
        data_type='sound',
        frontend=frontend,
    )
    # [1, T, 560] -> [T, 560]
    input_features = input_features[0]
    input_feature_length = int(input_feature_length[0])
    return input_features, input_feature_length


# ---------------------------------------------------------------------------
# Dataset preprocessor
# ---------------------------------------------------------------------------
class TasuASRPreprocessor(ResponsePreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        text = row.get('txt') or row.get('text') or row.get('response')
        prompt = row.get('prompt') or 'Transcribe the speech to text.'
        if wav is None or text is None:
            return None
        wav = _resolve_project_path(wav)
        return {
            'messages': [
                {'role': 'user', 'content': f'{prompt}<speech>'},
                {'role': 'assistant', 'content': text},
            ],
            'audios': [wav],
        }


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
def _freeze_for_tasu_train(model) -> None:
    """Freeze encoder and LLM; keep the projector trainable."""
    for name, p in model.named_parameters():
        if name.startswith('llm.') or name.startswith('encoder.'):
            p.requires_grad = False
        else:
            p.requires_grad = True
    frozen = sum(p.numel() for n, p in model.named_parameters()
                 if n.startswith('llm.') or n.startswith('encoder.'))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'[TASU] Frozen encoder + LLM ({frozen / 1e6:.1f}M). Trainable: {trainable / 1e6:.1f}M.')


def _build_tasu_model(model_dir: str, torch_dtype: torch.dtype):
    """Build TASU (SenseVoiceSmall + projector + Qwen2.5-1.5B) from local paths."""
    from aispeech_asr_config import ModelConfig, TrainConfig

    encoder_path = os.path.join(_ROOT, 'model', 'SenseVoiceSmall')
    model_config = ModelConfig(
        llm_name='Qwen2.5-1.5B-Instruct',
        llm_path=model_dir,
        llm_type='decoder_only',
        llm_dim=1536,
        encoder_name='sensevoice',
        encoder_path=encoder_path,
        encoder_dim=25055,
        encoder_projector='linear-silu',
        encoder_projector_ds_rate=1,
        ctc_linear=None,
    )
    model_config = OmegaConf.structured(model_config)
    train_config = TrainConfig(
        freeze_llm=True,
        freeze_encoder=True,
        freeze_projector=False,
        ctc_posterior=True,
        do_psd=True,
        gt_emb=False,
        gt_emb_noise=False,
        voca_trans=False,
        top1_emb=False,
        cross_attn=False,
        use_peft=False,
        quantization=False,
    )
    train_config = OmegaConf.structured(train_config)

    tokenizer = setup_tokenizer(train_config, model_config)
    DEFAULT_SPEECH_TOKEN = '<speech>'
    tokenizer.add_special_tokens({'additional_special_tokens': [DEFAULT_SPEECH_TOKEN]})
    tokenizer.default_speech_token = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    tokenizer.default_ignore_token = -100

    llm = setup_llm(train_config, model_config)
    # Resize embeddings to account for the newly added <speech> placeholder token.
    llm.resize_token_embeddings(len(tokenizer))

    encoder = setup_encoder(train_config, model_config)
    projector = setup_encoder_projector(train_config, model_config)

    base_model = slam_model_asr(
        encoder,
        llm,
        projector,
        tokenizer,
        train_config,
        model_config,
    )

    # Wrap forward/generate/save_pretrained so ms-swift can train and save.
    _orig_forward = base_model.forward
    _orig_generate = base_model.generate

    def _tasu_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_feature_length: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Drop ms-swift keys that TASU does not accept.
        kwargs.pop('loss_scale', None)
        kwargs.pop('text_loss_mask', None)
        kwargs.pop('channel', None)
        outputs, _ = _orig_forward(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            input_feature_length=input_feature_length,
            labels=labels,
            **kwargs,
        )
        # ms-swift's loss validator feeds an all-padding label batch. The underlying
        # LLM returns NaN in that case; replace it with a finite zero loss.
        if labels is not None and (labels == -100).all() and outputs.loss is not None:
            outputs.loss = torch.zeros_like(outputs.loss)
        # Expose the merged (audio-expanded) labels so that forward-pass eval scripts
        # can align predictions with the actual response positions.
        if labels is not None:
            outputs.final_labels = labels
        return outputs

    def _tasu_generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        input_feature_length: Optional[torch.Tensor] = None,
        max_new_tokens: int = 200,
        **kwargs,
    ):
        # Fallback for validator-style calls that omit audio features.
        if input_features is None:
            logger.warning_once('[TASU] generate() called without input_features; using a single zero frame fallback.')
            device = input_ids.device if input_ids is not None else next(self.parameters()).device
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            input_features = torch.zeros(batch_size, 1, 560, dtype=torch.bfloat16, device=device)
            input_feature_length = torch.ones(batch_size, dtype=torch.long, device=device)
        # Match the dtype of the SenseVoice encoder weights and use autocast so that
        # internal float32 masks do not upcast the bfloat16 activations.
        model_dtype = next(self.parameters()).dtype
        input_features = input_features.to(dtype=model_dtype)
        device_type = 'cuda' if input_features.is_cuda else 'cpu'
        with torch.autocast(device_type, dtype=model_dtype):
            return _orig_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                input_feature_length=input_feature_length,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

    def _tasu_save_pretrained(self, save_directory: str, *args, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        state_dict = {name: tensor for name, tensor in state_dict.items() if name in trainable_names}
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, 'trainable_state_keys.json'), 'w', encoding='utf-8') as f:
            json.dump(sorted(state_dict), f, indent=2)
        logger.info(f'[TASU] Saving trainable-only checkpoint tensors: {len(state_dict)}')
        torch.save(state_dict, os.path.join(save_directory, 'pytorch_model.bin'))
        if getattr(self, 'config', None) is not None:
            self.config.save_pretrained(save_directory)

    def _tasu_gradient_checkpointing_enable(self, **kwargs):
        # ms-swift calls this when --gradient_checkpointing true. TASU's LLM
        # backbone already supports gradient checkpointing; the wrapper itself
        # does not need extra handling.
        if hasattr(self, 'llm') and hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable(**kwargs)

    def _tasu_enable_input_require_grads(self):
        # ms-swift calls this after enabling gradient checkpointing so that
        # inputs flowing into the checkpointed blocks still receive gradients.
        if hasattr(self, 'llm') and hasattr(self.llm, 'enable_input_require_grads'):
            self.llm.enable_input_require_grads()

    base_model.forward = MethodType(_tasu_forward, base_model)
    base_model.generate = MethodType(_tasu_generate, base_model)
    base_model.save_pretrained = MethodType(_tasu_save_pretrained, base_model)
    base_model.gradient_checkpointing_enable = MethodType(_tasu_gradient_checkpointing_enable, base_model)
    base_model.enable_input_require_grads = MethodType(_tasu_enable_input_require_grads, base_model)

    # Expose common Hugging Face model attributes that ms-swift expects.
    base_model.config = llm.config
    type(base_model).device = property(lambda self: next(self.parameters()).device)
    base_model.to(dtype=torch_dtype)
    _freeze_for_tasu_train(base_model)
    return base_model, tokenizer


def get_model_tokenizer_tasu(
    model_dir: str,
    model_info,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs,
):
    torch_dtype = model_info.torch_dtype or torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not load_model:
        return None, tokenizer

    model, tokenizer = _build_tasu_model(model_dir, torch_dtype)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
class TasuTemplate(Template):
    support_padding_free = False
    placeholder_tokens = ['<speech>']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default to training mode so that template.encode keeps the assistant response.
        # ms-swift will switch to 'pt' during inference as needed.
        self.set_mode('train')
        self._sensevoice_path = os.path.join(_ROOT, 'model', 'SenseVoiceSmall')

    def _tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _encode(self, inputs, return_length: bool = False, **kwargs) -> Dict[str, Any]:
        messages = inputs.messages
        if len(messages) != 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
            raise ValueError(f'TasuTemplate expects one user turn and one assistant turn: {messages}')
        if not inputs.audios:
            raise ValueError('TasuTemplate requires one audio path.')

        prompt_text = messages[0]['content']
        # ms-swift may prepend <audio> if the prompt lacks an audio placeholder;
        # we replace it with TASU's <speech> token below.
        prompt_text = prompt_text.replace('<audio>', '').replace('<speech>', '').strip()
        response_text = messages[1]['content']
        audio_path = _resolve_project_path(inputs.audios[0])

        input_features, input_feature_length = _extract_sensevoice_fbank(audio_path, self._sensevoice_path)

        # Prompt follows the TASU ASR template:
        # <|im_start|>user\n{prompt}<speech><|im_end|>\n<|im_start|>assistant\n
        prompt_str = (
            f'<|im_start|>user\n{prompt_text}<speech><|im_end|>\n'
            f'<|im_start|>assistant\n'
        )
        prompt_ids = self._tokenize(prompt_str)
        response_ids = self._tokenize(response_text) + [self.tokenizer.eos_token_id]
        input_ids = prompt_ids + response_ids

        labels = [-100] * len(prompt_ids) + response_ids

        encoded = {
            'input_ids': input_ids,
            'labels': labels,
            'input_features': input_features,
            'input_feature_length': input_feature_length,
        }
        if return_length:
            encoded['length'] = len(input_ids)
        return encoded

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        max_len = max(len(item['input_ids']) for item in batch)
        if padding_to is not None:
            max_len = max(max_len, padding_to)

        max_feat_len = max(item['input_features'].shape[0] for item in batch)
        max_feat_dim = max(item['input_features'].shape[1] for item in batch)

        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        input_features = torch.zeros((len(batch), max_feat_len, max_feat_dim), dtype=torch.bfloat16)
        input_feature_length = torch.zeros(len(batch), dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = len(item['input_ids'])
            input_ids[i, :seq_len] = torch.tensor(item['input_ids'], dtype=torch.long)
            attention_mask[i, :seq_len] = True
            labels[i, :seq_len] = torch.tensor(item['labels'], dtype=torch.long)

            feat = item['input_features']
            feat_len = feat.shape[0]
            if not isinstance(feat, torch.Tensor):
                feat = torch.tensor(feat, dtype=torch.bfloat16)
            else:
                feat = feat.to(torch.bfloat16)
            input_features[i, :feat_len] = feat
            input_feature_length[i] = int(item['input_feature_length'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'input_features': input_features,
            'input_feature_length': input_feature_length,
        }


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------
register_model_arch(
    MultiModelKeys(
        'tasu',
        language_model=['llm.model.embed_tokens', 'llm.model.layers', 'llm.model.norm'],
        vision_tower=['encoder'],
        aligner=['encoder_projector'],
        generator=['llm.lm_head'],
    ))

register_template(
    TemplateMeta(
        template_type='tasu',
        prefix=[],
        prompt=[],
        chat_sep=[],
        template_cls=TasuTemplate,
    ))

register_model(
    ModelMeta(
        model_type='tasu',
        model_groups=[ModelGroup([Model(model_path=os.path.join(_ROOT, 'model', 'Qwen2.5-1.5B'))])],
        template='tasu',
        get_function=get_model_tokenizer_tasu,
        is_multimodal=True,
        model_arch='tasu',
        tags=['audio', 'asr'],
    ))

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data', 'combined_asr_local.jsonl'),
        dataset_name='combined_asr_local',
        preprocess_func=TasuASRPreprocessor(),
    ),
    exist_ok=True,
)

# Register smoke-test datasets so that the harness can load them by name.
register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'outputs', '20260708-203711', 'smoke_test', 'overfit1.jsonl'),
        dataset_name='tasu_smoke_overfit1',
        preprocess_func=TasuASRPreprocessor(),
    ),
    exist_ok=True,
)
register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'outputs', '20260708-203711', 'smoke_test', 'mini100.jsonl'),
        dataset_name='tasu_smoke_mini100',
        preprocess_func=TasuASRPreprocessor(),
    ),
    exist_ok=True,
)
register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data', 'test_audio_30s_x100.jsonl'),
        dataset_name='tasu_smoke_bs_test',
        preprocess_func=TasuASRPreprocessor(),
    ),
    exist_ok=True,
)
