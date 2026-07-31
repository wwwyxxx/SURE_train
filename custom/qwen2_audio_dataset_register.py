import os
from functools import partial
from typing import Any, Dict, List, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset
from swift.llm.template import Template, register_template
from swift.llm.template.constant import MLLMTemplateType
from swift.llm.template.template.qwen import Qwen2AudioTemplate, QwenTemplateMeta
from swift.llm.template.vision_utils import load_batch, load_audio
from transformers import AutoConfig, Qwen2AudioForConditionalGeneration
from swift.llm.model import register_model, ModelMeta, ModelGroup, Model
import torch

class CombinedASRPreprocessor(ResponsePreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        text = row.get('txt') or row.get('text') or row.get('response')
        prompt = row.get('prompt') or 'Transcribe the speech to text:'
        if wav is None or text is None:
            return None
        if not os.path.isabs(wav):
            wav = os.path.abspath(wav)
        return {
            'messages': [
                # Qwen2-Audio base expects audio tokens BEFORE the text prompt
                {'role': 'user', 'content': f'<audio>{prompt}'},
                {'role': 'assistant', 'content': text},
            ],
            'audios': [wav],
        }


class Qwen2AudioTemplateFixed(Qwen2AudioTemplate):
    """Qwen2Audio template that expands the single <|AUDIO|> placeholder to the
    correct number of tokens matching the audio feature length."""

    def _encode(self, inputs) -> Dict[str, Any]:
        encoded = Template._encode(self, inputs)
        if inputs.audios:
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=self.sampling_rate))
            audio_inputs = self.processor.feature_extractor(
                audios, sampling_rate=self.sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            encoded.update(audio_inputs)

            # Expand the single <|AUDIO|> placeholder to match audio feature frames.
            audio_token_id = self._tokenize('<|AUDIO|>')[0]
            input_ids = encoded['input_ids']
            audio_idx = next((i for i, tid in enumerate(input_ids) if tid == audio_token_id), None)
            if audio_idx is not None:
                # Reproduce Qwen2AudioEncoder._get_feat_extract_output_lengths:
                # conv1 (stride=1, padding=1) keeps length; conv2 (stride=2, padding=1)
                # gives (L - 1) // 2 + 1; avg_pooler (kernel=2, stride=2) gives
                # (feat_len - 2) // 2 + 1.
                num_frames = audio_inputs['feature_attention_mask'].sum(-1)
                feat_lengths = (num_frames - 1) // 2 + 1
                num_audio_tokens = ((feat_lengths - 2) // 2 + 1).tolist()
                if isinstance(num_audio_tokens, int):
                    num_audio_tokens = [num_audio_tokens]
                # For simplicity support the common single-audio ASR case here;
                # multi-audio needs per-audio expansion matching inputs.audios order.
                n_tokens = num_audio_tokens[0]
                new_input_ids = input_ids[:audio_idx] + [audio_token_id] * n_tokens + input_ids[audio_idx + 1:]
                encoded['input_ids'] = new_input_ids
                labels = encoded.get('labels')
                if labels is not None:
                    new_labels = labels[:audio_idx] + [-100] * n_tokens + labels[audio_idx + 1:]
                    encoded['labels'] = new_labels
        return encoded


# Override the default qwen2_audio template with the fixed version.
register_template(
    QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplateFixed),
    exist_ok=True,
)


register_dataset(
    DatasetMeta(
        dataset_path='data/combined_asr_local.jsonl',
        dataset_name='combined_asr_local',
        preprocess_func=CombinedASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path='data/test_audio_30s_x100.jsonl',
        dataset_name='test_audio_30s_x100',
        preprocess_func=CombinedASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path='outputs/20260629-183012/smoke_test/overfit1.jsonl',
        dataset_name='combined_asr_local_smoke_overfit1',
        preprocess_func=CombinedASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path='outputs/20260629-183012/smoke_test/overfit2.jsonl',
        dataset_name='combined_asr_local_smoke_overfit2',
        preprocess_func=CombinedASRPreprocessor(),
    ),
    exist_ok=True,
)

register_dataset(
    DatasetMeta(
        dataset_path='outputs/20260629-183012/smoke_test/mini100.jsonl',
        dataset_name='combined_asr_local_smoke_mini100',
        preprocess_func=CombinedASRPreprocessor(),
    ),
    exist_ok=True,
)


# Custom Qwen2-Audio model loader that enables flash attention only for the LLM,
# not for the audio encoder, to avoid CUDA index-out-of-bounds errors.
def get_model_tokenizer_qwen2_audio_flash_llm(model_dir: str,
                                              model_info,
                                              model_kwargs: Dict[str, Any],
                                              load_model: bool = True,
                                              **kwargs):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # Enable flash attention only on the text (LLM) config; leave audio_config untouched.
    if hasattr(model_config, 'text_config') and model_config.text_config is not None:
        model_config.text_config._attn_implementation = 'flash_attention_2'
        model_config.text_config.attn_implementation = 'flash_attention_2'
    kwargs['tokenizer'] = processor.tokenizer
    kwargs['model_config'] = model_config
    kwargs['automodel_class'] = Qwen2AudioForConditionalGeneration
    # Do not pass attn_impl down; we set it manually above.
    kwargs.pop('attn_impl', None)
    from swift.llm.model.register import get_model_tokenizer_from_local
    model, _ = get_model_tokenizer_from_local(model_dir, model_info, model_kwargs, load_model, **kwargs)
    return model, processor


register_model(
    ModelMeta(
        model_type='qwen2_audio_flash_llm',
        model_groups=[ModelGroup([Model(model_path='/workspace/outputs/20260629-183012/assembled_model')])],
        template='qwen2_audio',
        get_function=get_model_tokenizer_qwen2_audio_flash_llm,
        is_multimodal=True,
        model_arch='qwen2_audio',
        architectures=['Qwen2AudioForConditionalGeneration'],
        tags=['audio'],
    ))
