import json
import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
_MIMO_AUDIO_SRC = os.path.join(_ROOT, 'MiMo-Audio', 'src')
if _MIMO_AUDIO_SRC not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_SRC)

import torch
import torchaudio
from transformers import AutoTokenizer
from mimo_audio.process_speechdata import InputSegment
from mimo_audio_tokenizer import MiMoAudioTokenizer


def encode_audio(wav_path, audio_tokenizer, device):
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
            codes, _ = audio_tokenizer.encoder.encode(input_features=features, input_lens=lengths[None],
                                                      return_codes_only=True)
        codes_list.append(codes)
    codes = torch.cat(codes_list, dim=-1)
    audio_codes = codes[:8].transpose(0, 1)
    num_timesteps = audio_codes.shape[0]
    group_size = 4
    if num_timesteps % group_size != 0:
        padding_needed = group_size - (num_timesteps % group_size)
        audio_codes = torch.cat([audio_codes, audio_codes[-1:].repeat(padding_needed, 1)], dim=0)
    return audio_codes.reshape(-1).cpu()


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(_ROOT, 'output/mimo_audio_overfit100_7gpu_v2/v0-20260628-100545/checkpoint-14'),
        trust_remote_code=True,
    )
    for tok in ['<|sosp|>', '<|eosp|>', '<|empty|>', '<|Human|>', '<|SpeechLM|>', '<|sostm|>', '<|eostm|>', '<|eot|>']:
        if tok not in tokenizer.get_vocab():
            tokenizer.add_tokens([tok], special_tokens=True)

    audio_tokenizer = MiMoAudioTokenizer.from_pretrained(os.path.join(_ROOT, 'model/MiMo-Audio-Tokenizer'))
    audio_tokenizer.eval().bfloat16().to('cuda:0')

    with open(os.path.join(_ROOT, 'example/ASR_BAC009S0002W0263_overfit100.jsonl')) as f:
        row = json.loads(f.readline())

    wav_path = row['wav']
    if wav_path.startswith('/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train'):
        wav_path = '/workspace' + wav_path[len('/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train'):]

    audio_tokens = encode_audio(wav_path, audio_tokenizer, 'cuda:0')
    print('audio_tokens length:', len(audio_tokens), 'groups:', len(audio_tokens) // (8 * 4))

    empty_idx = tokenizer.convert_tokens_to_ids('<|empty|>')
    speech_zeroemb_idx = [1024, 1024, 128, 128, 128, 128, 128, 128]
    group_size = 4
    audio_channels = 8

    segments = [
        InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(audio=audio_tokens, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='Transcribe the speech to text.', speech_zeroemb_idx=speech_zeroemb_idx,
                     text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text=row['txt'], speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
    ]
    input_ids = torch.cat([seg.to_input_id(tokenizer, group_size, audio_channels) for seg in segments], dim=1)
    text_ids = input_ids[0, ::group_size].clone()
    T_groups = text_ids.shape[0]

    labels = text_ids.clone()
    text_loss_mask = torch.zeros(T_groups, dtype=torch.bool)
    assistant_header_ids = torch.tensor(tokenizer('<|im_start|>assistant\n', add_special_tokens=False)['input_ids'],
                                        dtype=text_ids.dtype)
    match_start = None
    for start in range(T_groups - len(assistant_header_ids) + 1):
        if torch.equal(text_ids[start:start + len(assistant_header_ids)], assistant_header_ids):
            match_start = start
            break
    think_prefix_ids = tokenizer('<think>\n\n</think>\n', add_special_tokens=False)['input_ids']
    loss_start = match_start + len(assistant_header_ids) + len(think_prefix_ids) if match_start is not None else None
    if loss_start is not None and loss_start < T_groups:
        text_loss_mask[loss_start:] = True

    labels = torch.cat([labels[1:], torch.tensor([tokenizer.pad_token_id or 0], dtype=labels.dtype)])
    labels[:-1][text_loss_mask[1:] == False] = -100
    labels[-1] = -100

    print('T_groups:', T_groups)
    print('assistant_header match_start:', match_start, 'ids:', assistant_header_ids.tolist())
    print('think_prefix_ids:', think_prefix_ids, 'len:', len(think_prefix_ids))
    print('loss_start:', loss_start)
    print('text_loss_mask sum:', int(text_loss_mask.sum()))

    print('\nAll text groups:')
    for i, tid in enumerate(text_ids.tolist()):
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        lab = labels[i].item()
        print(f'{i:3d}: text_id={tid:7d} {tok!r:20s} label={lab:7d} {tokenizer.decode([lab], skip_special_tokens=False) if lab >= 0 else "-100"!r:20s} {"LOSS" if text_loss_mask[i] else ""}')


if __name__ == '__main__':
    main()
