import json
import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
_MIMO_AUDIO_SRC = os.path.join(_ROOT, 'MiMo-Audio', 'src')
if _MIMO_AUDIO_SRC not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_SRC)

import torch
import torchaudio
from mimo_audio_tokenizer import MiMoAudioTokenizer

audio_tokenizer = MiMoAudioTokenizer.from_pretrained(os.path.join(_ROOT, 'model/MiMo-Audio-Tokenizer'))
audio_tokenizer.eval().bfloat16().to('cuda:0')

cfg = audio_tokenizer.config
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=cfg.sampling_rate,
    n_fft=cfg.nfft,
    hop_length=cfg.hop_length,
    win_length=cfg.window_size,
    f_min=cfg.fmin,
    f_max=cfg.fmax,
    n_mels=cfg.n_mels,
    power=1.0,
    center=True,
).to('cuda:0')

maxs = [0] * 8
mins = [10**9] * 8
count = 0
bad = []

for line in open(os.path.join(_ROOT, 'data/reprodata_asr_1000.jsonl')):
    d = json.loads(line)
    wav_path = os.path.join(_ROOT, d['wav'])
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    target_sr = cfg.sampling_rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.to('cuda:0')
    mel = torch.log(torch.clamp(mel_transform(wav[None, :]), min=1e-7)).squeeze().transpose(0, 1)

    input_len = mel.size(0)
    segment_size = 6000
    input_len_seg = [segment_size] * (input_len // segment_size)
    if input_len % segment_size > 0:
        input_len_seg.append(input_len % segment_size)

    codes_list = []
    for features, lengths in zip(torch.split(mel, input_len_seg), [torch.tensor(x, device='cuda:0') for x in input_len_seg]):
        with torch.no_grad():
            codes, _ = audio_tokenizer.encoder.encode(
                input_features=features,
                input_lens=lengths[None],
                return_codes_only=True,
            )
        codes_list.append(codes)
    codes = torch.cat(codes_list, dim=-1)  # [num_quantizers, T]
    audio_codes = codes[:8].transpose(0, 1).detach().cpu()  # [T, 8]

    for c in range(8):
        col = audio_codes[:, c]
        maxs[c] = max(maxs[c], int(col.max()))
        mins[c] = min(mins[c], int(col.min()))
        mx = int(col.max())
        if (c < 2 and mx >= 1025) or (c >= 2 and mx >= 129):
            bad.append((count, c, mx, d['wav']))
    count += 1
    if count % 100 == 0:
        print(f'processed {count}, maxs={maxs}, mins={mins}, bad={len(bad)}')

print('final maxs', maxs)
print('final mins', mins)
print('bad samples', bad[:20], 'total', len(bad))
