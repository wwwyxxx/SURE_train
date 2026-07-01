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
from mimo_audio.modeling_mimo_audio import MiMoAudioArguments
from mimo_audio.process_speechdata import InputSegment
from mimo_audio_tokenizer import MiMoAudioTokenizer
from custom.mimo_audio_swift_register import MiMoAudioSFTModel


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


def build_prompt(tokenizer, audio_tokens, response_text, empty_idx, speech_zeroemb_idx, group_size, audio_channels):
    segments = [
        InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(audio=audio_tokens, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='Transcribe the speech to text.', speech_zeroemb_idx=speech_zeroemb_idx,
                     text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text=response_text, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
    ]
    return torch.cat([seg.to_input_id(tokenizer, group_size, audio_channels) for seg in segments], dim=1)


def main():
    checkpoint = os.path.join(_ROOT, 'output/mimo_audio_overfit100_7gpu_v2/v0-20260628-100545/checkpoint-14')
    tokenizer_path = os.path.join(_ROOT, 'model/MiMo-Audio-Tokenizer')
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    for tok in ['<|sosp|>', '<|eosp|>', '<|empty|>', '<|Human|>', '<|SpeechLM|>', '<|sostm|>', '<|eostm|>', '<|eot|>']:
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
    model = MiMoAudioSFTModel.from_pretrained(checkpoint, args=mimo_args, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True, device_map={'': device})
    model.eval()

    audio_tokenizer = MiMoAudioTokenizer.from_pretrained(tokenizer_path)
    audio_tokenizer.eval().bfloat16().to(device)

    with open(os.path.join(_ROOT, 'example/ASR_BAC009S0002W0263_overfit100.jsonl')) as f:
        row = json.loads(f.readline())

    wav_path = row['wav']
    if wav_path.startswith('/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train'):
        wav_path = '/workspace' + wav_path[len('/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train'):]

    audio_tokens = encode_audio(wav_path, audio_tokenizer, device)
    input_ids = build_prompt(tokenizer, audio_tokens, row['txt'], empty_idx, model.speech_empty_ids,
                             model.config.group_size, model.config.audio_channels)
    input_ids = input_ids.unsqueeze(0).to(device)
    t_groups = input_ids.shape[-1] // model.config.group_size
    attention_mask = torch.ones(1, t_groups, dtype=torch.bool, device=device)
    position_ids = torch.arange(t_groups, dtype=torch.long, device=device).unsqueeze(0)

    text_ids_2d = input_ids[0, 0, ::model.config.group_size].unsqueeze(0)
    text_loss_mask = torch.zeros(1, t_groups, dtype=torch.bool, device=device)
    text_loss_mask[0, 27:] = True

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                        labels=text_ids_2d, text_loss_mask=text_loss_mask)
    logits = outputs.logits.float()
    preds = logits.argmax(dim=-1).cpu()[0].tolist()
    text_ids_list = text_ids_2d[0].cpu().tolist()

    print('group | text_id | pred_id | text_tok | pred_tok')
    for i, (tid, pid) in enumerate(zip(text_ids_list, preds)):
        ttok = tokenizer.decode([tid], skip_special_tokens=False)
        ptok = tokenizer.decode([pid], skip_special_tokens=False)
        marker = ''
        if i >= 27:
            marker = ' <- RESPONSE'
        print(f'{i:5d} | {tid:7d} | {pid:7d} | {ttok!r:20s} | {ptok!r:20s}{marker}')


if __name__ == '__main__':
    main()
