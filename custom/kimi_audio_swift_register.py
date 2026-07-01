import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from swift.llm import (DatasetMeta, Model, ModelGroup, ModelMeta, MultiModelKeys, ResponsePreprocessor, Template,
                       TemplateMeta, register_dataset, register_model, register_model_arch, register_template)
from swift.utils import get_logger


logger = get_logger()

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_KIMI_AUDIO_ROOT = os.path.join(_ROOT, 'Kimi-Audio')
if _KIMI_AUDIO_ROOT not in sys.path:
    sys.path.insert(0, _KIMI_AUDIO_ROOT)


def _resolve_project_path(path: str) -> str:
    if path is None or os.path.exists(path):
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


_KIMIA_USED_SPECIAL_TOKENS = [
    '[BOS]',
    '[EOS]',
    '<|im_msg_end|>',
    '<|im_user_msg_start|>',
    '<|im_assistant_msg_start|>',
    '<|reserved_token_0|>',
    '<|reserved_token_1|>',
    '<|reserved_token_2|>',
    '<|reserved_token_3|>',
    '[EOT]',
    '<|reserved_token_4|>',
    '<|reserved_token_5|>',
    '<|reserved_token_6|>',
    '<|reserved_token_7|>',
    '<|reserved_token_8|>',
    '<|reserved_token_9|>',
    '<|reserved_token_10|>',
    '<|reserved_token_11|>',
    '<|im_media_begin|>',
    '<|reserved_token_12|>',
    '<|im_media_end|>',
    '<|reserved_token_13|>',
    '<|reserved_token_14|>',
    '<|im_kimia_text_blank|>',
    '<|im_kimia_text_eos|>',
    '<|reserved_token_15|>',
    '<|reserved_token_16|>',
    '<|im_kimia_user_msg_start|>',
    '<|im_kimia_assistant_msg_start|>',
    '<|reserved_token_17|>',
    '<|reserved_token_18|>',
    '<|reserved_token_19|>',
    '<|im_kimia_speech_ct_id|>',
    '<|im_kimia_speech_ctd_id|>',
]


def _patch_kimia_special_tokens(tokenizer, num_base_tokens: int = 151643, num_reserved_special_tokens: int = 421):
    autoset_special_tokens = [
        f'<|reserved_token_{i}|>' for i in range(20, num_reserved_special_tokens - len(_KIMIA_USED_SPECIAL_TOKENS) + 20)
    ]
    special_tokens = _KIMIA_USED_SPECIAL_TOKENS + autoset_special_tokens
    tokenizer.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
    tokenizer.pad_token = special_tokens[-1]
    tokenizer.pad_id = tokenizer.special_tokens[tokenizer.pad_token]
    tokenizer.pad_token_id = tokenizer.pad_id
    return tokenizer

from finetune_codes.configuration_moonshot_kimia import KimiAudioConfig  # noqa: E402
from finetune_codes.model import KimiAudioModel  # noqa: E402
from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder  # noqa: E402
from kimia_infer.utils.data import KimiAContent  # noqa: E402
from kimia_infer.utils.special_tokens import instantiate_extra_tokens  # noqa: E402


class KimiAudioASRPreprocessor(ResponsePreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        text = row.get('txt') or row.get('text') or row.get('response')
        prompt = row.get('prompt') or 'Transcribe the speech to text.'
        if wav is None or text is None:
            return None
        wav = _resolve_project_path(wav)
        return {
            'messages': [
                {
                    'role': 'user',
                    'content': f'{prompt} <audio>'
                },
                {
                    'role': 'assistant',
                    'content': text
                },
            ],
            'audios': [wav],
        }


class KimiAudioTextSFTModel(KimiAudioModel):

    def __init__(self, config):
        old_init = WhisperEncoder.__init__

        def local_whisper_init(encoder_self, model_path, mel_batch_size=40, unfreeze_online_whisper_model=False):
            if model_path == 'openai/whisper-large-v3':
                model_path = os.environ.get('KIMI_AUDIO_WHISPER_CONFIG', os.path.join(_ROOT, 'model/whisper-large-v3'))
            old_init(
                encoder_self,
                model_path,
                mel_batch_size=mel_batch_size,
                unfreeze_online_whisper_model=unfreeze_online_whisper_model,
            )

        WhisperEncoder.__init__ = local_whisper_init
        try:
            super().__init__(config)
        finally:
            WhisperEncoder.__init__ = old_init

    def save_pretrained(self, save_directory, *args, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        state_dict = {name: tensor for name, tensor in state_dict.items() if name in trainable_names}
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, 'trainable_state_keys.json'), 'w') as f:
            json.dump(sorted(state_dict), f, indent=2)
        logger.info(f'Saving trainable-only checkpoint tensors: {len(state_dict)}')
        return super().save_pretrained(save_directory, *args, state_dict=state_dict, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        text_input_ids: torch.LongTensor = None,
        whisper_input_feature: Optional[Any] = None,
        is_continuous_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[Any] = None,
        text_loss_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_mode: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if whisper_input_feature is not None:
            target_device = input_ids.device if input_ids is not None else next(self.parameters()).device
            if isinstance(whisper_input_feature, torch.Tensor):
                wav_tensors = [whisper_input_feature]
            elif isinstance(whisper_input_feature, (list, tuple)):
                wav_tensors = list(whisper_input_feature)
            else:
                raise TypeError(f'Unsupported whisper_input_feature: {type(whisper_input_feature)}')

            whisper_feat_list = []
            for wav_tensor in wav_tensors:
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
                wav_tensor = wav_tensor.to(target_device)
                feats = self.whisper_model(wav_tensor)
                feats = feats.reshape(
                    feats.shape[0],
                    int(feats.shape[1] // 4),
                    feats.shape[2] * 4,
                )
                whisper_feat_list.append(feats.squeeze(0))

            # Pad whisper features to the max length in the batch.
            max_feat_len = max(feat.shape[0] for feat in whisper_feat_list)
            feat_dim = whisper_feat_list[0].shape[1]
            whisper_feats = torch.zeros(
                len(whisper_feat_list), max_feat_len, feat_dim,
                dtype=whisper_feat_list[0].dtype, device=target_device
            )
            for i, feat in enumerate(whisper_feat_list):
                whisper_feats[i, :feat.shape[0], :] = feat
        else:
            whisper_feats = None

        outputs = super(KimiAudioModel, self).forward(
            input_ids=input_ids,
            text_input_ids=text_input_ids,
            whisper_input_feature=whisper_feats,
            is_continuous_mask=is_continuous_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            generation_mode=generation_mode,
            return_dict=True,
        )

        text_logits, _audio_logits = outputs.logits
        loss = None
        if labels is not None:
            if isinstance(labels, dict):
                text_labels = labels['text_labels']
                text_loss_mask = labels['text_loss_mask']
            elif isinstance(labels, (list, tuple)):
                text_labels, text_loss_mask = labels
            else:
                text_labels = labels
                if text_loss_mask is None:
                    raise ValueError('text_loss_mask is required when labels is a tensor.')
            # Shift labels/mask so that text_logits[i] predicts text_labels[i+1].
            pad_token = self.config.pad_token_id
            if pad_token is None:
                pad_token = getattr(self.config, 'eos_token_id', None) or 0
            text_labels = torch.cat((text_labels[:, 1:], text_labels.new_full((text_labels.shape[0], 1), pad_token)), dim=1)
            text_loss_mask = torch.cat((text_loss_mask[:, 1:], text_loss_mask.new_full((text_loss_mask.shape[0], 1), False)), dim=1)
            text_labels = text_labels.to(text_logits.device)
            text_loss_mask = text_loss_mask.to(text_logits.device)
            loss_all = F.cross_entropy(
                text_logits.reshape(-1, text_logits.shape[-1]),
                text_labels.reshape(-1),
                reduction='none',
            )
            loss = (loss_all * text_loss_mask.reshape(-1)).sum() / (text_loss_mask.reshape(-1).sum() + 1e-4)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=text_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class KimiAudioTextTemplate(Template):
    support_padding_free = False
    placeholder_tokens = ['<audio>']

    def _tokenize_text(self, text: str) -> List[int]:
        try:
            return self.tokenizer.encode(text, bos=False, eos=False)
        except TypeError:
            return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _audio_token_len(wav_len: int) -> int:
        return (wav_len - 1) // (160 * 8) + 1

    def _append_text(self, content: KimiAContent, text: str, role: str, tokenize_role: bool, has_msg_end: bool):
        extra = self.extra_tokens
        has_loss = role == 'assistant'
        if tokenize_role:
            if role == 'user':
                content.audio_append(extra.kimia_user_msg_start)
            elif role == 'assistant':
                content.audio_append(extra.kimia_assistant_msg_start)
            else:
                raise NotImplementedError(f'role: {role}')
            content.text_append(extra.kimia_text_blank)

        text_tokens = self._tokenize_text(text)
        content.text_extend(text_tokens, has_loss)
        content.audio_extend([extra.kimia_text_blank] * len(text_tokens))
        if role == 'assistant':
            content.text_append(extra.kimia_text_eos, has_loss)
            content.audio_append(extra.kimia_text_blank, audio_token_loss_mask=False)
        if has_msg_end:
            content.audio_append(extra.msg_end, audio_token_loss_mask=False)
            content.text_append(extra.kimia_text_blank)

    def _append_audio(self, content: KimiAContent, wav: torch.Tensor, role: str, tokenize_role: bool, has_ct: bool,
                      has_msg_end: bool):
        extra = self.extra_tokens
        has_loss = role == 'assistant'
        if tokenize_role:
            if role == 'user':
                content.audio_append(extra.kimia_user_msg_start)
            elif role == 'assistant':
                content.audio_append(extra.kimia_assistant_msg_start)
            else:
                raise NotImplementedError(f'role: {role}')
            content.text_append(extra.kimia_text_blank)

        speech_tokens = [extra.kimia_text_blank] * self._audio_token_len(wav.numel())
        content.audio_append(extra.media_begin)
        content.audio_extend(speech_tokens, is_continuous=True, audio_token_loss_mask=has_loss)
        content.audio_append(extra.media_end, audio_token_loss_mask=has_loss)
        content.text_extend([extra.kimia_text_blank] * (len(speech_tokens) + 2))
        if has_ct:
            content.audio_append(extra.kimia_speech_ct_id)
            content.text_append(extra.kimia_text_blank)
        if has_msg_end:
            content.audio_append(extra.msg_end, audio_token_loss_mask=False)
            content.text_append(extra.kimia_text_blank)

    def _encode(self, inputs) -> Dict[str, Any]:
        self.extra_tokens = instantiate_extra_tokens(self.tokenizer)
        messages = inputs.messages
        if len(messages) != 2 or messages[0]['role'] != 'user' or messages[1]['role'] != 'assistant':
            raise ValueError(f'KimiAudioTextTemplate currently expects one user turn and one assistant turn: {messages}')
        if not inputs.audios:
            raise ValueError('KimiAudioTextTemplate requires one audio path.')

        prompt = messages[0]['content'].replace('<audio>', '').strip()
        response = messages[1]['content']
        import librosa
        wav_np, _ = librosa.load(_resolve_project_path(inputs.audios[0]), sr=16000)
        wav = torch.tensor(wav_np, dtype=torch.float32)

        content = KimiAContent()
        self._append_text(content, prompt, role='user', tokenize_role=True, has_msg_end=False)
        self._append_audio(content, wav, role='user', tokenize_role=False, has_ct=True, has_msg_end=True)
        self._append_text(content, response, role='assistant', tokenize_role=True, has_msg_end=True)
        if not content.is_valid():
            raise ValueError('Invalid Kimi-Audio encoded content.')

        audio_input_ids, text_input_ids, is_continuous_mask, _audio_loss_mask, text_loss_mask = content.to_tensor()
        return {
            'input_ids': audio_input_ids[0],
            'text_input_ids': text_input_ids[0],
            'is_continuous_mask': is_continuous_mask[0],
            'whisper_input_feature': wav,
            'labels': text_input_ids[0],
            'text_loss_mask': text_loss_mask[0],
        }

    def data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        # Determine max sequence length in the batch.
        max_len = max(item['input_ids'].shape[0] for item in batch)
        if padding_to is not None:
            max_len = max(max_len, padding_to)

        def _pad_2d(items, pad_value, dtype):
            padded = torch.full((len(items), max_len), pad_value, dtype=dtype)
            for i, item in enumerate(items):
                length = item.shape[0]
                padded[i, :length] = item
            return padded

        input_ids = _pad_2d([item['input_ids'] for item in batch], pad_id, torch.long)
        text_input_ids = _pad_2d([item['text_input_ids'] for item in batch], pad_id, torch.long)
        is_continuous_mask = _pad_2d([item['is_continuous_mask'] for item in batch], False, torch.bool)
        labels = _pad_2d([item['labels'] for item in batch], -100, torch.long)
        text_loss_mask = _pad_2d([item['text_loss_mask'] for item in batch], False, torch.bool)

        # Apply text loss mask to labels (non-loss positions become -100).
        labels[~text_loss_mask.bool()] = -100

        # Attention mask: 1 for real tokens, 0 for padding.
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
        for i, item in enumerate(batch):
            length = item['input_ids'].shape[0]
            attention_mask[i, :length] = True

        # Whisper input features are raw waveforms of variable length; keep as a list.
        whisper_input_features = [item['whisper_input_feature'] for item in batch]

        return {
            'input_ids': input_ids,
            'text_input_ids': text_input_ids,
            'is_continuous_mask': is_continuous_mask,
            'whisper_input_feature': whisper_input_features,
            'labels': labels,
            'text_loss_mask': text_loss_mask,
            'attention_mask': attention_mask,
        }


class PretrainedWhisperEncoder(WhisperEncoder):

    def __init__(self, mel_batch_size=20, unfreeze_online_whisper_model=True):
        whisper_dir = os.environ.get('KIMI_AUDIO_WHISPER_CONFIG', os.path.join(_ROOT, 'model/whisper-large-v3'))
        super().__init__(
            whisper_dir,
            mel_batch_size=mel_batch_size,
            unfreeze_online_whisper_model=unfreeze_online_whisper_model,
        )


def _copy_qwen_weights(kimia_model: KimiAudioTextSFTModel, qwen_model) -> None:
    qwen_state = qwen_model.state_dict()
    target_state = kimia_model.state_dict()
    copied = 0
    skipped = 0
    partial_copied = 0
    remap_prefixes = [
        ('model.embed_tokens.', 'model.embed_tokens.'),
        ('model.layers.', 'model.layers.'),
        ('model.norm.', 'model.norm.'),
        ('lm_head.', 'mimo_output.'),
    ]
    with torch.no_grad():
        for q_name, q_tensor in qwen_state.items():
            target_name = None
            for src_prefix, dst_prefix in remap_prefixes:
                if q_name.startswith(src_prefix):
                    target_name = dst_prefix + q_name[len(src_prefix):]
                    break
            if target_name is None or target_name not in target_state:
                continue
            dst = target_state[target_name]
            if dst.shape == q_tensor.shape:
                dst.copy_(q_tensor.to(dtype=dst.dtype))
                copied += 1
            elif target_name == 'model.embed_tokens.weight' and dst.ndim == 2 and q_tensor.ndim == 2:
                rows = min(dst.shape[0], q_tensor.shape[0])
                cols = min(dst.shape[1], q_tensor.shape[1])
                dst[:rows, :cols].copy_(q_tensor[:rows, :cols].to(dtype=dst.dtype))
                partial_copied += 1
            elif target_name == 'mimo_output.weight' and dst.ndim == 2 and q_tensor.ndim == 2:
                rows = min(dst.shape[0], q_tensor.shape[0])
                cols = min(dst.shape[1], q_tensor.shape[1])
                dst[:rows, :cols].copy_(q_tensor[:rows, :cols].to(dtype=dst.dtype))
                partial_copied += 1
            else:
                skipped += 1
    logger.info(
        f'Kimi-Audio initialized from Qwen shared weights. copied={copied}, '
        f'partial_copied={partial_copied}, skipped={skipped}')


def _freeze_for_text_overfit(model: KimiAudioTextSFTModel) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False
    if os.environ.get('KIMI_AUDIO_TEXTHEAD_ONLY', '0') == '1':
        # Minimal debug mode: text head only.
        train_prefixes = ['mimo_output.']
    else:
        # Adaptor + mimo branch + text head are trained by default.
        # mimo_layers/mimo_norm are randomly initialized in the SFT-from-Qwen setup,
        # so they must be trainable; shared LLM (embed_tokens/layers/norm) stays frozen.
        train_prefixes = ['model.vq_adaptor.', 'model.mimo_layers.', 'model.mimo_norm.', 'mimo_output.']
        if os.environ.get('KIMI_AUDIO_TRAIN_WHISPER', '0') == '1':
            train_prefixes.append('whisper_model.')
    train_prefixes = tuple(train_prefixes)
    for name, p in model.named_parameters():
        if name.startswith(train_prefixes):
            p.requires_grad = True
    logger.info(f'Frozen shared LLM/audio head. Trainable prefixes: {", ".join(train_prefixes)}.')


def _patch_text_token_acc_metric() -> None:
    from swift.trainers.mixin import SwiftMixin

    if getattr(SwiftMixin, '_kimi_audio_text_token_acc_patched', False):
        return

    origin_compute_acc = SwiftMixin._compute_acc

    def compute_acc_with_text_labels(self, outputs, labels, cu_seqlens=None, attention_mask=None):
        if isinstance(labels, dict) and {'text_labels', 'text_loss_mask'} <= labels.keys():
            logits = outputs.logits
            text_labels = labels['text_labels'].to(logits.device)
            text_loss_mask = labels['text_loss_mask'].to(logits.device).bool()
            preds = logits.argmax(dim=-1)
            if preds.shape == text_labels.shape:
                valid = text_loss_mask & (text_labels != -100)
                if valid.any():
                    metric = (preds[valid] == text_labels[valid]).tolist()
                    mode = 'train' if self.model.training else 'eval'
                    self.custom_metrics[mode]['text_token_acc'].update(metric)
            return
        return origin_compute_acc(self, outputs, labels, cu_seqlens=cu_seqlens, attention_mask=attention_mask)

    SwiftMixin._compute_acc = compute_acc_with_text_labels
    SwiftMixin._kimi_audio_text_token_acc_patched = True
    logger.info('Patched SwiftMixin to log text_token_acc from text_labels/text_loss_mask.')


def get_model_tokenizer_kimi_audio_text(model_dir: str,
                                        model_info,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        **kwargs):
    _patch_text_token_acc_metric()
    torch_dtype = model_info.torch_dtype or torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _patch_kimia_special_tokens(tokenizer)
    qwen_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    config = KimiAudioConfig(
        vocab_size=163840,
        hidden_size=qwen_config.hidden_size,
        intermediate_size=qwen_config.intermediate_size,
        num_hidden_layers=qwen_config.num_hidden_layers,
        num_attention_heads=qwen_config.num_attention_heads,
        num_key_value_heads=qwen_config.num_key_value_heads,
        hidden_act=qwen_config.hidden_act,
        rms_norm_eps=qwen_config.rms_norm_eps,
        rope_theta=getattr(qwen_config, 'rope_theta', 10000.0),
        tie_word_embeddings=False,
        torch_dtype=torch_dtype,
    )
    if not load_model:
        return None, tokenizer

    model = KimiAudioTextSFTModel(config)
    model.whisper_model = PretrainedWhisperEncoder(mel_batch_size=20, unfreeze_online_whisper_model=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    _copy_qwen_weights(model, qwen_model)
    del qwen_model
    model.to(dtype=torch_dtype)
    if os.environ.get('KIMI_AUDIO_FREEZE_LLM', '1') != '0':
        _freeze_for_text_overfit(model)
    return model, tokenizer


register_model_arch(
    MultiModelKeys(
        'kimi_audio_text',
        language_model=['model.embed_tokens', 'model.layers', 'model.norm'],
        vision_tower=['whisper_model'],
        aligner=['model.vq_adaptor', 'mimo_output'],
        generator=['lm_head'],
    ))

register_template(
    TemplateMeta(
        template_type='kimi_audio_text',
        prefix=[],
        prompt=[],
        chat_sep=[],
        template_cls=KimiAudioTextTemplate,
    ))

register_model(
    ModelMeta(
        model_type='kimi_audio_text',
        model_groups=[ModelGroup([Model(model_path=os.path.join(_ROOT, 'model/Qwen2.5-7B'))])],
        template='kimi_audio_text',
        get_function=get_model_tokenizer_kimi_audio_text,
        is_multimodal=True,
        model_arch='kimi_audio_text',
        tags=['audio', 'asr'],
    ))

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'example/ASR_BAC009S0002W0263_overfit100.jsonl'),
        dataset_name='kimi_audio_asr_overfit100',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_existing.jsonl'),
        dataset_name='reprodata_asr_existing',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_overfit100.jsonl'),
        dataset_name='reprodata_asr_overfit100',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_zh_existing.jsonl'),
        dataset_name='reprodata_asr_zh_existing',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_zh_overfit10.jsonl'),
        dataset_name='reprodata_asr_zh_overfit10',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/reprodata_asr_zh_existing_aishell93.jsonl'),
        dataset_name='reprodata_asr_zh_existing_aishell93',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_aishell-1.jsonl'),
        dataset_name='combined_asr_aishell_1',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.abspath(os.path.join(_ROOT, '..', 'combined_asr_local.jsonl')),
        dataset_name='combined_asr_local',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/test_audio_30s_x100.jsonl'),
        dataset_name='test_audio_30s_x100',
        preprocess_func=KimiAudioASRPreprocessor(),
    ),
    exist_ok=True)
