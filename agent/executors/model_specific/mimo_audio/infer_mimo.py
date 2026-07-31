import argparse
import json
import os
import sys

import torch
import torchaudio

_ROOT = os.path.abspath(os.path.dirname(__file__))

from mimo_audio_infer.mimo_audio.modeling_mimo_audio import MiMoAudioArguments, MiMoAudioForCausalLM
from mimo_audio_infer.mimo_audio.process_speechdata import InputSegment
from mimo_audio_infer.mimo_audio_tokenizer import MiMoAudioTokenizer
from transformers import AutoTokenizer


def _resolve_wav_path(wav: str) -> str:
    if os.path.isabs(wav):
        for host_root in ('/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train',
                          '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train'):
            if wav.startswith(host_root):
                return '/workspace' + wav[len(host_root):]
        return wav
    return os.path.join(_ROOT, wav)


def _encode_audio(wav_path: str, audio_tokenizer, device: str, group_size: int, audio_channels: int):
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    target_sr = audio_tokenizer.config.sampling_rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.to(device)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=audio_tokenizer.config.sampling_rate,
        n_fft=audio_tokenizer.config.nfft,
        hop_length=audio_tokenizer.config.hop_length,
        win_length=audio_tokenizer.config.window_size,
        f_min=audio_tokenizer.config.fmin,
        f_max=audio_tokenizer.config.fmax,
        n_mels=audio_tokenizer.config.n_mels,
        power=1.0,
        center=True,
    ).to(device)
    mel = torch.log(torch.clamp(mel_transform(wav[None, :]), min=1e-7)).squeeze().transpose(0, 1)

    input_len = mel.size(0)
    segment_size = 6000
    input_len_seg = [segment_size] * (input_len // segment_size)
    if input_len % segment_size > 0:
        input_len_seg.append(input_len % segment_size)

    codes_list = []
    for features, lengths in zip(torch.split(mel, input_len_seg),
                                 [torch.tensor(x, device=device) for x in input_len_seg]):
        with torch.no_grad():
            codes, _ = audio_tokenizer.encoder.encode(
                input_features=features,
                input_lens=lengths[None],
                return_codes_only=True,
            )
        codes_list.append(codes)
    codes = torch.cat(codes_list, dim=-1)  # [num_quantizers, T]
    audio_codes = codes[:audio_channels].transpose(0, 1)  # [T, audio_channels]

    num_timesteps = audio_codes.shape[0]
    if num_timesteps % group_size != 0:
        padding_needed = group_size - (num_timesteps % group_size)
        audio_codes = torch.cat([audio_codes, audio_codes[-1:].repeat(padding_needed, 1)], dim=0)
    return audio_codes.reshape(-1).to('cpu')


def _build_asr_prompt(tokenizer, audio_tokens, empty_idx, speech_zeroemb_idx, group_size, audio_channels):
    template = 'Transcribe the speech to text.'
    segments = [
        InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(audio=audio_tokens, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text=template, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
    ]
    input_ids = torch.cat([seg.to_input_id(tokenizer, group_size, audio_channels) for seg in segments], dim=1)
    return input_ids


def _greedy_generate(model, tokenizer, input_ids, empty_idx, speech_zeroemb_idx, group_size, audio_channels,
                     max_new_tokens=128):
    device = next(model.parameters()).device
    input_ids = input_ids.unsqueeze(0).to(device)
    eos_id = tokenizer.eos_token_id
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    stop_ids = {eos_id, im_end_id}

    generated = []
    for _ in range(max_new_tokens):
        seq_len = input_ids.shape[-1]
        assert seq_len % group_size == 0
        t_groups = seq_len // group_size
        attention_mask = torch.ones(1, t_groups, dtype=torch.bool, device=device)
        position_ids = torch.arange(t_groups, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        next_text_token = int(outputs.text_logits[:, -1, :].argmax(dim=-1).item())
        if next_text_token in stop_ids:
            break
        generated.append(next_text_token)

        # Append one new group: one real text token + (group_size-1) padding, and
        # group_size empty tokens per audio channel.
        new_text_channel = torch.full((1, group_size), -100, dtype=torch.long, device=device)
        new_text_channel[0, 0] = next_text_token
        new_audio_channels = torch.stack([
            torch.full((group_size,), speech_zeroemb_idx[c], dtype=torch.long, device=device)
            for c in range(audio_channels)
        ])
        new_group = torch.cat([new_text_channel, new_audio_channels], dim=0).unsqueeze(0)
        input_ids = torch.cat([input_ids, new_group], dim=-1)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--output', type=str, default=None, help='Path to save decode results as JSONL')
    args = parser.parse_args()

    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_ROOT, args.checkpoint)
    dataset = args.dataset if os.path.isabs(args.dataset) else os.path.join(_ROOT, args.dataset)
    tokenizer_path = os.path.join(_ROOT, 'model/MiMo-Audio-Tokenizer')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f'Loading tokenizer from {checkpoint}')
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    special_tokens = ['<|sosp|>', '<|eosp|>', '<|empty|>', '<|Human|>', '<|SpeechLM|>', '<|sostm|>', '<|eostm|>', '<|eot|>']
    for tok in special_tokens:
        if tok not in tokenizer.get_vocab():
            tokenizer.add_tokens([tok], special_tokens=True)

    empty_idx = tokenizer.convert_tokens_to_ids('<|empty|>')
    mimo_args = MiMoAudioArguments(
        model_name_or_path=checkpoint,
        sosp_idx=tokenizer.convert_tokens_to_ids('<|sosp|>'),
        eosp_idx=tokenizer.convert_tokens_to_ids('<|eosp|>'),
        empty_idx=empty_idx,
        sostm_idx=tokenizer.convert_tokens_to_ids('<|sostm|>'),
        eostm_idx=tokenizer.convert_tokens_to_ids('<|eostm|>'),
        eot_idx=tokenizer.convert_tokens_to_ids('<|eot|>'),
    )

    print(f'Loading model from {checkpoint} on {device}')
    model = MiMoAudioForCausalLM.from_pretrained(
        checkpoint,
        args=mimo_args,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={'': device},
    )
    model.eval()

    print(f'Loading audio tokenizer from {tokenizer_path}')
    audio_tokenizer = MiMoAudioTokenizer.from_pretrained(tokenizer_path)
    audio_tokenizer.eval().bfloat16().to(device)

    group_size = model.config.group_size
    audio_channels = model.config.audio_channels
    speech_zeroemb_idx = model.speech_empty_ids

    with open(dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    selected = rows[args.start:args.start + args.num_samples]
    correct = 0
    out_f = open(args.output, 'w', encoding='utf-8') if args.output else None
    try:
        for idx, row in enumerate(selected, start=args.start):
            wav_path = _resolve_wav_path(row['wav'])
            target = row.get('txt') or row.get('text') or row.get('response') or ''
            try:
                audio_tokens = _encode_audio(wav_path, audio_tokenizer, device, group_size, audio_channels)
                input_ids = _build_asr_prompt(tokenizer, audio_tokens, empty_idx, speech_zeroemb_idx, group_size, audio_channels)
                pred = _greedy_generate(model, tokenizer, input_ids, empty_idx, speech_zeroemb_idx, group_size,
                                        audio_channels, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                pred = f'<ERROR: {e}>'
            ok = pred.strip() == target.strip()
            if ok:
                correct += 1
            print(f'[{idx}] {"OK" if ok else "BAD"}')
            print(f'target: {target}')
            print(f'pred:   {pred}')
            print('-' * 60)
            if out_f is not None:
                out_f.write(json.dumps({
                    'idx': idx,
                    'wav': row.get('wav'),
                    'target': target,
                    'pred': pred,
                    'correct': ok,
                }, ensure_ascii=False) + '\n')
                out_f.flush()
    finally:
        if out_f is not None:
            out_f.close()

    print(f'Accuracy: {correct}/{len(selected)} = {correct / len(selected):.2%}')


if __name__ == '__main__':
    main()
