"""Adapter for runtime registration of step_audio2_mini with component-wise initialization.

Component mapping:
  - language_model -> model.model (Qwen2Model) initialized from Qwen2.5-7B
  - vision_tower   -> model.encoder (AudioEncoder) initialized from Qwen2-Audio-7B audio_tower
  - aligner        -> model.adapter randomly initialized
  - text_head      -> base lm_head kept as-is from Step-Audio-2-mini-assembled
"""
from string import Template


def render(config: dict) -> str:
    derived = config['derived_model_type']
    func_name = f'get_model_tokenizer_{derived.replace("-", "_")}'
    base = config['base_model_path']
    component_paths = config['component_paths']
    component_trainable = config.get('component_trainable', {})
    llm_path = component_paths.get('language_model', 'model/Qwen2.5-7B')
    encoder_path = component_paths.get('vision_tower', 'model/Qwen2-Audio-7B')
    train_llm = False
    train_encoder = False
    train_aligner = True
    train_text_head = True

    template = Template('''"""Auto-generated custom model registration for $derived.

Base model: $base
Component replacements:
  language_model: $llm_path
  vision_tower: $encoder_path
  aligner: random
"""
import gc

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2AudioForConditionalGeneration
from swift.llm import Model, ModelGroup, ModelMeta, register_model
from swift.utils import get_logger


logger = get_logger()


def _replace_llm(model, llm_path: str, torch_dtype):
    logger.info(f'[$derived] Loading LLM from {llm_path}')
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch_dtype,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.model = llm_model.model
    del llm_model


def _replace_encoder(model, encoder_path: str, torch_dtype):
    logger.info(f'[$derived] Loading audio encoder from {encoder_path}')
    enc_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        encoder_path,
        torch_dtype=torch_dtype,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    src_enc = enc_model.audio_tower
    tgt_enc = model.encoder

    tgt_enc.conv1.load_state_dict(src_enc.conv1.state_dict())
    tgt_enc.conv2.load_state_dict(src_enc.conv2.state_dict())
    tgt_enc.positional_embedding.load_state_dict(src_enc.embed_positions.state_dict())

    for src_layer, tgt_block in zip(src_enc.layers, tgt_enc.blocks):
        tgt_block.attn.query.weight.data = src_layer.self_attn.q_proj.weight.data
        tgt_block.attn.key.weight.data = src_layer.self_attn.k_proj.weight.data
        tgt_block.attn.value.weight.data = src_layer.self_attn.v_proj.weight.data
        tgt_block.attn.out.weight.data = src_layer.self_attn.out_proj.weight.data
        tgt_block.attn_ln.load_state_dict(src_layer.self_attn_layer_norm.state_dict())
        tgt_block.mlp_ln.load_state_dict(src_layer.final_layer_norm.state_dict())
        tgt_block.mlp[0].weight.data = src_layer.fc1.weight.data
        tgt_block.mlp[0].bias.data = src_layer.fc1.bias.data
        tgt_block.mlp[2].weight.data = src_layer.fc2.weight.data
        tgt_block.mlp[2].bias.data = src_layer.fc2.bias.data

    tgt_enc.after_norm.load_state_dict(src_enc.layer_norm.state_dict())
    tgt_enc.avg_pooler.load_state_dict(src_enc.avg_pooler.state_dict())
    del enc_model


def _reinit_adapter(model):
    logger.info(f'[$derived] Re-initializing adapter randomly')
    def _init_module(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.adapter.apply(_init_module)


def _patch_insert_audio_features(model):
    """Patch _insert_audio_features to avoid in-place modification of embedding output."""
    original_insert = model._insert_audio_features
    def _insert_audio_features_patched(input_ids, hidden_states, wavs, wav_lens):
        if wavs is None:
            return hidden_states
        hidden_states = hidden_states.clone()
        return original_insert(input_ids, hidden_states, wavs, wav_lens)
    model._insert_audio_features = _insert_audio_features_patched


def $func_name(
    model_dir,
    model_info,
    model_kwargs,
    load_model,
    **kwargs,
):
    torch_dtype = model_kwargs.get('torch_dtype', torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    if not load_model:
        return None, processor

    logger.info(f'[$derived] Loading base Step-Audio-2-mini from {model_dir}')
    base_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    _replace_llm(base_model, '$llm_path', torch_dtype)
    _replace_encoder(base_model, '$encoder_path', torch_dtype)
    _reinit_adapter(base_model)
    _patch_insert_audio_features(base_model)

    model_info.config = base_model.config

    # Set requires_grad according to component_trainable strategy.
    logger.info(f'[$derived] Setting requires_grad: language_model=$train_llm, vision_tower=$train_encoder, aligner=$train_aligner, text_head=$train_text_head')
    for name, param in base_model.named_parameters():
        if name.startswith('model.'):
            param.requires_grad = $train_llm
        elif name.startswith('encoder.'):
            param.requires_grad = $train_encoder
        elif name.startswith('adapter.'):
            param.requires_grad = $train_aligner
        elif name.startswith('lm_head.'):
            param.requires_grad = $train_text_head
        else:
            param.requires_grad = False

    gc.collect()
    torch.cuda.empty_cache()
    return base_model, processor


register_model(
    ModelMeta(
        '$derived',
        [
            ModelGroup([
                Model(model_path='$base'),
            ]),
        ],
        'step_audio2_mini',
        $func_name,
        model_arch=None,
        architectures=['StepAudio2ForCausalLM'],
        torch_dtype=torch.bfloat16,
        is_multimodal=True,
        task_type='causal_lm',
        tags=['audio'],
    ),
    exist_ok=True,
)

logger.info('[$derived] Model type $derived registered successfully')
''')
    return template.substitute(
        derived=derived,
        func_name=func_name,
        base=base,
        llm_path=llm_path,
        encoder_path=encoder_path,
        train_llm=train_llm,
        train_encoder=train_encoder,
        train_aligner=train_aligner,
        train_text_head=train_text_head,
    )
