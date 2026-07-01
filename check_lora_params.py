import os
import sys
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MIMO_AUDIO_SRC = os.path.join(_ROOT, 'MiMo-Audio', 'src')
if _MIMO_AUDIO_SRC not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_SRC)

from swift.llm import get_model_tokenizer


def main():
    model_dir = '/workspace/model/MiMo-Audio-7B-Base-merged'

    # Import custom register so mimo_audio model_type is available
    import_external_path = os.path.join(_ROOT, 'custom', 'mimo_audio_swift_register.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location('custom_register', import_external_path)
    custom_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_register)

    # Load with LoRA config matching training script
    os.environ['MIMO_AUDIO_FREEZE_LLM'] = '0'
    model, tokenizer = get_model_tokenizer(
        model_type='mimo_audio',
        model_id_or_path=model_dir,
        load_model=True,
        torch_dtype=torch.bfloat16,
    )

    # Manually apply the same LoRA config as training
    from peft import LoraConfig, get_peft_model
    from swift.llm.train.tuner import prepare_adapter

    # Easiest: directly use PEFT with our config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules='all-linear',
        lora_dropout=0.05,
        modules_to_save=[
            'speech_embeddings.0', 'speech_embeddings.1', 'speech_embeddings.2',
            'speech_embeddings.3', 'speech_embeddings.4', 'speech_embeddings.5',
            'speech_embeddings.6', 'speech_embeddings.7',
            'speech_group_downcast',
            'lm_head',
        ],
    )
    peft_model = get_peft_model(model, lora_config)

    lora_names = []
    trainable_names = []
    for name, p in peft_model.named_parameters():
        if 'lora' in name.lower():
            lora_names.append((name, p.numel()))
        if p.requires_grad:
            trainable_names.append((name, p.numel()))

    print(f'Total parameters: {sum(p.numel() for p in peft_model.parameters()) / 1e6:.1f}M')
    print(f'Trainable parameters: {sum(n for _, n in trainable_names) / 1e6:.1f}M')
    print(f'LoRA parameters: {len(lora_names)} tensors, total {sum(n for _, n in lora_names) / 1e6:.1f}M')
    print('\nFirst 30 LoRA parameter names:')
    for name, numel in lora_names[:30]:
        print(f'  {name:<80s} {numel:>12d}')
    if len(lora_names) > 30:
        print(f'  ... and {len(lora_names) - 30} more')

    print('\nModules with requires_grad=True (non-LoRA trainable):')
    non_lora_trainable = [(n, sz) for n, sz in trainable_names if 'lora' not in n.lower()]
    for name, numel in non_lora_trainable[:30]:
        print(f'  {name:<80s} {numel:>12d}')
    if len(non_lora_trainable) > 30:
        print(f'  ... and {len(non_lora_trainable) - 30} more')


if __name__ == '__main__':
    main()
