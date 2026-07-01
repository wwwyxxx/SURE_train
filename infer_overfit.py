import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='output/kimi_audio_asr_overfit100/v8-20260612-063053/checkpoint-300')
    parser.add_argument('--audio', default='example/BAC009S0002W0263.wav')
    parser.add_argument('--prompt', default='Transcribe the speech to text.')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()

    root = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, root)

    import custom.kimi_audio_swift_register as reg

    model_dir = os.path.join(root, 'model/Qwen2.5-7B')
    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(root, args.checkpoint)
    audio = args.audio if os.path.isabs(args.audio) else os.path.join(root, args.audio)

    class ModelInfo:
        torch_dtype = torch.bfloat16

    model, tokenizer = reg.get_model_tokenizer_kimi_audio_text(model_dir, ModelInfo(), {}, load_model=True)
    state = {}
    from safetensors.torch import load_file

    index_path = os.path.join(checkpoint, 'model.safetensors.index.json')
    if os.path.exists(index_path):
        import json
        with open(index_path, 'r') as f:
            weight_map = json.load(f)['weight_map']
        for shard in sorted(set(weight_map.values())):
            state.update(load_file(os.path.join(checkpoint, shard)))
    else:
        state.update(load_file(os.path.join(checkpoint, 'model.safetensors')))

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'missing={len(missing)} unexpected={len(unexpected)}')
    if os.path.exists(os.path.join(checkpoint, 'trainable_state_keys.json')):
        print('loaded trainable-only checkpoint')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    import librosa
    from kimia_infer.utils.data import KimiAContent

    template = object.__new__(reg.KimiAudioTextTemplate)
    template.processor = tokenizer
    template.extra_tokens = reg.instantiate_extra_tokens(tokenizer)

    wav_np, _ = librosa.load(audio, sr=16000)
    wav_tensor = torch.tensor(wav_np, dtype=torch.float32)
    content = KimiAContent()
    template._append_text(content, args.prompt, role='user', tokenize_role=True, has_msg_end=False)
    template._append_audio(content, wav_tensor, role='user', tokenize_role=False, has_ct=True, has_msg_end=True)
    content.audio_append(template.extra_tokens.kimia_assistant_msg_start)
    content.text_append(template.extra_tokens.kimia_text_blank)
    audio_input_ids, text_input_ids, is_continuous_mask, _, _ = content.to_tensor()
    encoded = {
        'input_ids': audio_input_ids[0],
        'text_input_ids': text_input_ids[0],
        'is_continuous_mask': is_continuous_mask[0],
        'whisper_input_feature': wav_tensor,
    }

    input_ids = encoded['input_ids'].unsqueeze(0).to(device)
    text_input_ids = encoded['text_input_ids'].unsqueeze(0).to(device)
    is_continuous_mask = encoded['is_continuous_mask'].unsqueeze(0).to(device)
    wav = encoded['whisper_input_feature'].unsqueeze(0).to(device)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    past_key_values = None

    extra = template.extra_tokens
    kimia_token_offset = getattr(model.config, 'kimia_token_offset', 152064)
    generated = []
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            outputs = model(
                input_ids=input_ids,
                text_input_ids=text_input_ids,
                whisper_input_feature=wav,
                is_continuous_mask=is_continuous_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1]
            past_key_values = outputs.past_key_values
            next_text = logits.argmax(dim=-1)
            token_id = int(next_text.item())
            if token_id == extra.kimia_text_eos:
                break
            if token_id < kimia_token_offset:
                generated.append(token_id)

            next_audio = torch.full((1, 1), extra.kimia_text_blank, dtype=torch.long, device=device)
            input_ids = next_audio
            text_input_ids = next_text.view(1, 1)
            is_continuous_mask = torch.zeros((1, 1), dtype=torch.bool, device=device)
            wav = None
            position_ids = position_ids[:, -1:] + 1

    print('generated_token_ids:', generated)
    print('decoded:', tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == '__main__':
    main()
