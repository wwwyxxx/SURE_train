import json
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, WavLMModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from swift.llm import (DatasetMeta, Model, ModelGroup, ModelMeta, MultiModelKeys, Template, TemplateMeta,
                       register_dataset, register_model, register_model_arch, register_template, RowPreprocessor)
from swift.utils import get_logger

logger = get_logger()


def _find_project_root(start: str) -> str:
    start = os.path.abspath(start)
    current = start
    for _ in range(5):
        if os.path.isdir(os.path.join(current, 'data')) and os.path.isdir(os.path.join(current, 'model')):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


_ROOT = _find_project_root(os.path.dirname(__file__))
_SLAM_ENCODER_DIM = 1024
_SLAM_LLM_DIM = 4096
_SLAM_PROJECTOR_DS_RATE = 5


class SlamLLMASRPreprocessor(RowPreprocessor):
    """Map combined_asr jsonl rows to ms-swift messages + audios format."""

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wav = row.get('wav') or row.get('audio') or row.get('source')
        text = row.get('txt') or row.get('text') or row.get('target')
        prompt = row.get('prompt') or 'Transcribe the speech to text.'
        if wav is None or text is None:
            return None
        return {
            'messages': [
                {'role': 'user', 'content': f'{prompt} <audio>'},
                {'role': 'assistant', 'content': text},
            ],
            'audios': [wav],
        }


class EncoderProjectorConcat(nn.Module):
    """SLAM-LLM linear projector: k-frame concat -> 2048 -> llm_dim."""

    def __init__(self, encoder_dim: int, llm_dim: int, k: int = 5):
        super().__init__()
        self.k = k
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(encoder_dim * k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, llm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        discard = T % self.k
        if discard > 0:
            x = x[:, :-discard, :]
        T = x.size(1)
        x = x.contiguous().view(B, T // self.k, D * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SlamLLMModelASR(nn.Module):
    """Wrapper that combines WavLM encoder, linear projector and Vicuna LLM for ASR."""

    def __init__(self, encoder: nn.Module, llm: nn.Module, encoder_projector: nn.Module, tokenizer,
                 freeze_encoder: bool, freeze_llm: bool):
        super().__init__()
        self.encoder = encoder
        self.llm = llm
        self.encoder_projector = encoder_projector
        self.tokenizer = tokenizer

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()

    def gradient_checkpointing_enable(self, gradient_checkpointing_func=None, **kwargs):
        # Older transformers PreTrainedModel.gradient_checkpointing_enable does not accept
        # gradient_checkpointing_func; drop it before delegating to the underlying LLM.
        kwargs.pop('gradient_checkpointing_func', None)
        self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory: str, *args, state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = self.state_dict()
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        state_dict = {name: tensor for name, tensor in state_dict.items() if name in trainable_names}
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, 'trainable_state_keys.json'), 'w') as f:
            json.dump(sorted(state_dict), f, indent=2)
        logger.info(f'Saving trainable-only checkpoint tensors: {len(state_dict)}')
        return torch.save(state_dict, os.path.join(save_directory, 'pytorch_model.bin'))

    def _compute_audio_embeds(self, audio: torch.Tensor, audio_mask: torch.Tensor):
        encoder_outputs = self.encoder(
            audio.float(),
            attention_mask=audio_mask,
            return_dict=True
        )
        encoder_outs = encoder_outputs.last_hidden_state

        wav_lens = audio_mask.sum(dim=1).to(device=audio.device)
        feat_lens = self.encoder._get_feat_extract_output_lengths(wav_lens)
        proj_lens = feat_lens // self.encoder_projector.k

        proj_dtype = next(self.encoder_projector.parameters()).dtype
        encoder_outs = encoder_outs.to(dtype=proj_dtype)
        audio_embeds = self.encoder_projector(encoder_outs)

        return audio_embeds, proj_lens

    def _merge_audio_into_text_embeds(
        self,
        inputs_embeds: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = inputs_embeds.clone()
        audio_embeds, proj_lens = self._compute_audio_embeds(audio, audio_mask)
        audio_embeds = audio_embeds.to(dtype=inputs_embeds.dtype)

        B = inputs_embeds.shape[0]
        for i in range(B):
            idx = modality_mask[i].nonzero(as_tuple=True)[0]
            placeholder_len = int(idx.numel())
            proj_len_i = int(proj_lens[i].item())

            if placeholder_len != proj_len_i:
                raise RuntimeError(
                    f'[SLAM-LLM] audio length mismatch: '
                    f'sample={i}, placeholder_len={placeholder_len}, '
                    f'proj_len_i={proj_len_i}, '
                    f'audio_samples={int(audio_mask[i].sum().item())}'
                )

            inputs_embeds[i, idx] = audio_embeds[i, :proj_len_i]

        return inputs_embeds

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                audio: Optional[torch.Tensor] = None,
                audio_mask: Optional[torch.Tensor] = None,
                modality_mask: Optional[torch.Tensor] = None,
                **kwargs):

                # 保证 frozen encoder / llm 不进入 train mode，避免 dropout 干扰
        self.encoder.eval()
        self.llm.eval()

        if os.environ.get('SLAM_DEBUG', '0') == '1':
            logger.info(
                '[MODE] '
                f'encoder.training={self.encoder.training} '
                f'llm.training={self.llm.training} '
                f'projector.training={self.encoder_projector.training}'
            )

        input_ids = input_ids.clone()
        input_ids[input_ids == -1] = 0
        inputs_embeds = self.llm.model.embed_tokens(input_ids)

        if audio is not None:
            inputs_embeds = self._merge_audio_into_text_embeds(inputs_embeds, audio, audio_mask, modality_mask)

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True)
        return outputs

    @torch.no_grad()
    def generate(self,
                 input_ids: torch.LongTensor = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 audio: Optional[torch.Tensor] = None,
                 audio_mask: Optional[torch.Tensor] = None,
                 modality_mask: Optional[torch.Tensor] = None,
                 **kwargs):
        # Replace audio placeholder -1 with a valid id for embedding lookup.
        input_ids_internal = input_ids.clone()
        input_ids_internal[input_ids_internal == -1] = 0
        inputs_embeds = self.llm.model.embed_tokens(input_ids_internal)

        if audio is not None:
            inputs_embeds = self._merge_audio_into_text_embeds(inputs_embeds, audio, audio_mask, modality_mask)

        generated_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        # Transformers generate() with inputs_embeds returns only newly generated tokens.
        # Concatenate the original input_ids so callers get the full sequence, and replace
        # audio placeholder -1 with a valid token id so downstream decoding does not crash.
        input_ids = input_ids.to(generated_ids.device)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.where(input_ids == -1, torch.tensor(pad_id, device=input_ids.device, dtype=input_ids.dtype), input_ids)
        return torch.cat([input_ids, generated_ids], dim=1)


class SlamLLMTemplate(Template):
    """Template for SLAM-LLM ASR: prefix audio tokens, then prompt/answer."""

    placeholder_tokens = ['<audio>']
    normalize = True

    @staticmethod
    def _wavlm_output_length(wav_len: int) -> int:
        """Exact WavLM-Large feature-extractor output length for raw waveform."""
        # conv1: kernel=10, stride=5
        #t = (wav_len - 10) // 5 + 1
        # conv2-7: kernel=3, stride=2
        '''
        for _ in range(6):
            t = (t - 3) // 2 + 1
        '''
        kernels = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
        t = int(wav_len)
        for k, s in zip(kernels, strides):
            t = (t - k) // s + 1
        return t

    @classmethod
    def _compute_audio_length(cls, wav: torch.Tensor) -> int:
        feat_len = cls._wavlm_output_length(wav.shape[0])
        return feat_len // _SLAM_PROJECTOR_DS_RATE

    def _encode(self, inputs) -> Dict[str, Any]:
        messages = inputs.messages
        if not messages or messages[0]['role'] != 'user':
            raise ValueError('SlamLLMTemplate expects a user turn.')
        if not inputs.audios:
            raise ValueError('SlamLLMTemplate requires one audio path.')

        import librosa
        wav, sr = librosa.load(inputs.audios[0], sr=16000)
        wav = torch.tensor(wav, dtype=torch.float32)
        if self.normalize:
            wav = F.layer_norm(wav, wav.shape)

        audio_length = self._compute_audio_length(wav)
        #DEBUG
        if os.environ.get('SLAM_DEBUG', '0') == '1':
            feat_len_est = self._wavlm_output_length(wav.shape[0])
            print(
                '[ENCODE]',
                f'wav_samples={wav.shape[0]}',
                f'feat_len_est={feat_len_est}',
                f'audio_length_est={audio_length}',
                f'prompt_len_pending=True',
            )




        prompt_text = messages[0]['content'].replace('<audio>', '').strip()
        prompt = f'USER: {prompt_text}\n ASSISTANT:'
        prompt_ids = self.tokenizer.encode(prompt)
        audio_pseudo = [-1] * audio_length
        input_ids = audio_pseudo + prompt_ids
        labels = [-100] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        # Training sample includes assistant answer + EOS.
        if len(messages) >= 2 and messages[1]['role'] == 'assistant':
            answer = messages[1]['content']
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            eos_id = self.tokenizer.eos_token_id
            input_ids = input_ids + answer_ids + [eos_id]
            labels = labels + answer_ids + [eos_id]
            attention_mask = [1] * len(input_ids)

        audio_mask = [1] * wav.shape[0]
        modality_mask = [True] * audio_length + [False] * (len(input_ids) - audio_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'audio': wav,
            'audio_mask': torch.tensor(audio_mask, dtype=torch.long),
            'modality_mask': torch.tensor(modality_mask, dtype=torch.bool),
        }

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super()._data_collator(batch, padding_to=padding_to)
        max_seq_len = res['input_ids'].shape[1]

        max_audio_len = max(b['audio'].shape[0] for b in batch)
        audio_batch = torch.zeros(len(batch), max_audio_len, dtype=torch.float32)
        audio_mask = torch.zeros(len(batch), max_audio_len, dtype=torch.long)
        for i, b in enumerate(batch):
            audio = b['audio']
            audio_batch[i, :audio.shape[0]] = audio
            audio_mask[i, :audio.shape[0]] = 1
        '''
        modality_mask = torch.zeros(len(batch), max_seq_len, dtype=torch.bool)
        for i, b in enumerate(batch):
            mask = b['modality_mask']
            modality_mask[i, :mask.shape[0]] = mask
        '''
        modality_mask = res['input_ids'].eq(-1)
        #DEBUG
        if os.environ.get('SLAM_DEBUG', '0') == '1':
            ids_mask = res['input_ids'].eq(-1)

            print('[COLLATOR]')
            print('  input_ids.shape:', tuple(res['input_ids'].shape))
            print('  labels.shape:', tuple(res['labels'].shape))
            print('  attention_mask.shape:', tuple(res['attention_mask'].shape))
            print('  ids==-1 per sample:', ids_mask.sum(dim=1).tolist())
            print('  copied modality_mask per sample:', modality_mask.sum(dim=1).tolist())
            print('  mask_equal:', torch.equal(ids_mask, modality_mask))

            for i in range(min(4, len(batch))):
                ids_pos = ids_mask[i].nonzero(as_tuple=True)[0]
                mod_pos = modality_mask[i].nonzero(as_tuple=True)[0]

                ids_range = (
                    (int(ids_pos[0]), int(ids_pos[-1]))
                    if ids_pos.numel() > 0 else None
                )
                mod_range = (
                    (int(mod_pos[0]), int(mod_pos[-1]))
                    if mod_pos.numel() > 0 else None
                )

                print(
                    f'  sample={i}',
                    f'ids_-1_count={ids_pos.numel()}',
                    f'ids_-1_range={ids_range}',
                    f'mod_count={mod_pos.numel()}',
                    f'mod_range={mod_range}',
                    f'labels_on_audio_all_ignore={bool((res["labels"][i][ids_mask[i]] == -100).all())}',
                    f'attn_on_audio_all_1={bool((res["attention_mask"][i][ids_mask[i]] == 1).all())}',
                )

        res['audio'] = audio_batch
        res['audio_mask'] = audio_mask
        res['modality_mask'] = modality_mask
        return res


def get_model_tokenizer_slam_llm(model_dir: str, model_info, model_kwargs: Dict[str, Any], load_model: bool = True,
                                 **kwargs):
    torch_dtype = model_info.torch_dtype or torch.bfloat16
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not load_model:
        return None, tokenizer

    llm = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    encoder_path = os.environ.get('SLAM_LLM_ENCODER_PATH', os.path.join(_ROOT, 'model/wavlm-large'))
    encoder = WavLMModel.from_pretrained(encoder_path)
    encoder.to(dtype=torch.float32)

    encoder_projector = EncoderProjectorConcat(_SLAM_ENCODER_DIM, _SLAM_LLM_DIM, _SLAM_PROJECTOR_DS_RATE)
    encoder_projector.to(dtype=torch_dtype)

    model = SlamLLMModelASR(encoder, llm, encoder_projector, tokenizer, freeze_encoder=True, freeze_llm=True)
    model.to(dtype=torch_dtype)
    # Keep WavLM feature extractor in float32; it does not support bfloat16/fp16 conv weights.
    model.encoder.to(dtype=torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'SLAM-LLM ASR loaded. Trainable params: {trainable / 1e6:.2f}M')
    return model, tokenizer


register_model_arch(
    MultiModelKeys(
        'slam_llm_asr',
        language_model=['llm.model.embed_tokens', 'llm.model.layers', 'llm.model.norm', 'llm.lm_head'],
        vision_tower=['encoder'],
        aligner=['encoder_projector'],
        generator=['llm.lm_head'],
    ))

register_template(
    TemplateMeta(
        template_type='slam_llm_asr',
        prefix=[],
        prompt=['{{QUERY}}'],
        chat_sep=[],
        suffix=[],
        system_prefix=[],
        template_cls=SlamLLMTemplate,
    ))

register_model(
    ModelMeta(
        model_type='slam_llm_asr',
        model_groups=[ModelGroup([Model(model_path=os.path.join(_ROOT, 'model/vicuna-7b-v1.5'))])],
        template='slam_llm_asr',
        get_function=get_model_tokenizer_slam_llm,
        is_multimodal=True,
        model_arch='slam_llm_asr',
        tags=['audio', 'asr'],
    ),
    exist_ok=True)

register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_local.jsonl'),
        dataset_name='combined_asr_local',
        preprocess_func=SlamLLMASRPreprocessor(),
    ),
    exist_ok=True)

# Smoke-test datasets (run-specific).
_SMOKE_DIR = os.path.join(_ROOT, 'outputs/20260709-144622/smoke_test')
if os.path.exists(os.path.join(_SMOKE_DIR, 'overfit1.jsonl')):
    register_dataset(
        DatasetMeta(
            dataset_path=os.path.join(_SMOKE_DIR, 'overfit1.jsonl'),
            dataset_name='slam_llm_smoke_overfit1',
            preprocess_func=SlamLLMASRPreprocessor(),
        ),
        exist_ok=True)
if os.path.exists(os.path.join(_SMOKE_DIR, 'mini100.jsonl')):
    register_dataset(
        DatasetMeta(
            dataset_path=os.path.join(_SMOKE_DIR, 'mini100.jsonl'),
            dataset_name='slam_llm_smoke_mini100',
            preprocess_func=SlamLLMASRPreprocessor(),
        ),
        exist_ok=True)
if os.path.exists(os.path.join(_SMOKE_DIR, 'mini100_zh.jsonl')):
    register_dataset(
        DatasetMeta(
            dataset_path=os.path.join(_SMOKE_DIR, 'mini100_zh.jsonl'),
            dataset_name='slam_llm_smoke_mini100_zh',
            preprocess_func=SlamLLMASRPreprocessor(),
        ),
        exist_ok=True)
_DEBUG_SINGLE_PATH = os.path.join(_ROOT, 'outputs/20260709-144622/debug_single.jsonl')
if os.path.exists(_DEBUG_SINGLE_PATH):
    register_dataset(
        DatasetMeta(
            dataset_path=_DEBUG_SINGLE_PATH,
            dataset_name='slam_llm_debug_single',
            preprocess_func=SlamLLMASRPreprocessor(),
        ),
        exist_ok=True)
if os.path.exists(os.path.join(_ROOT, 'data/test_audio_30s_x100.jsonl')):
    register_dataset(
        DatasetMeta(
            dataset_path=os.path.join(_ROOT, 'data/test_audio_30s_x100.jsonl'),
            dataset_name='slam_llm_smoke_bs_test',
            preprocess_func=SlamLLMASRPreprocessor(),
        ),
        exist_ok=True)
