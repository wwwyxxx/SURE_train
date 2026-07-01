import argparse
import os
import sys

import torch
import torchaudio
from transformers import AutoTokenizer

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root)

mimo_src = os.path.join(root, 'MiMo-Audio', 'src')
if mimo_src not in sys.path:
    sys.path.insert(0, mimo_src)

from mimo_audio.modeling_mimo_audio import MiMoAudioArguments, MiMoAudioForCausalLM, MiMoSampler  # noqa: E402
from mimo_audio.process_speechdata import InputSegment  # noqa: E402
from mimo_audio_tokenizer import MiMoAudioTokenizer  # noqa: E402
from transformers import GenerationConfig, GenerationMixin  # noqa: E402

# MiMoAudioForCausalLM defines its own generate() but relies on GenerationMixin helpers.
# Copy all public/private helper methods from GenerationMixin that are not already defined.
for _helper_name in dir(GenerationMixin):
    if _helper_name.startswith('_') and not hasattr(MiMoAudioForCausalLM, _helper_name) and _helper_name != '_has_unfinished_sequences':
        setattr(MiMoAudioForCausalLM, _helper_name, getattr(GenerationMixin, _helper_name))


def _has_unfinished_sequences(self, this_peer_finished, synced_gpus, device, cur_len=None, max_length=None):
    if synced_gpus and this_peer_finished:
        return False
    if cur_len is not None and max_length is not None and cur_len >= max_length:
        return False
    return not this_peer_finished


MiMoAudioForCausalLM._has_unfinished_sequences = _has_unfinished_sequences


def ensure_special_tokens(tokenizer):
    for token in ['<|sosp|>', '<|eosp|>', '<|empty|>', '<|Human|>',
                  '<|SpeechLM|>', '<|sostm|>', '<|eostm|>', '<|eot|>']:
        if token not in tokenizer.get_vocab():
            tokenizer.add_tokens([token], special_tokens=True)
    return tokenizer


def encode_audio(wav_path, audio_tokenizer, mel_transform, group_size, audio_channels):
    wav, sr = torchaudio.load(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    target_sr = audio_tokenizer.config.sampling_rate
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    device = next(audio_tokenizer.parameters()).device
    wav = wav.to(device)
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
    audio_codes = codes[:audio_channels].transpose(0, 1).detach().cpu()  # [T, audio_channels]

    num_timesteps = audio_codes.shape[0]
    if num_timesteps % group_size != 0:
        padding_needed = group_size - (num_timesteps % group_size)
        last_tokens = audio_codes[-1:, :]
        padding_tokens = last_tokens.repeat(padding_needed, 1)
        audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)
    return audio_codes.reshape(-1)  # [T * audio_channels]


def build_prompt_input_ids(tokenizer, prompt_text, audio_tokenized, group_size, audio_channels, speech_zeroemb_idx, empty_idx):
    segments = [
        InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(audio=audio_tokenized, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text=prompt_text, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
    ]
    input_ids = torch.cat([
        seg.to_input_id(tokenizer, group_size, audio_channels)
        for seg in segments
    ], dim=1)
    return input_ids


def decode_generated(generated_1d, group_size, audio_channels, tokenizer, skip_special_tokens=True):
    """generated_1d: [total_len] flattened tokens. Extract one text token per group."""
    total_len = generated_1d.shape[-1]
    tokens_per_group = group_size * (audio_channels + 1)
    assert total_len % tokens_per_group == 0, f"total_len={total_len} not divisible by {tokens_per_group}"
    T = total_len // tokens_per_group
    # reshape to [T, group_size, audio_channels+1]; text token is at the first position of each group
    reshaped = generated_1d.reshape(T, group_size, audio_channels + 1)
    text_tokens = reshaped[:, 0, 0].reshape(-1)
    # filter padding (-100) and negative ids
    valid = text_tokens[text_tokens >= 0]
    print(f'extracted text tokens: min={valid.min().item()} max={valid.max().item()} len={valid.shape[-1]}')
    return tokenizer.decode(valid.tolist(), skip_special_tokens=skip_special_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--audio', default=os.path.join(root, 'example', 'BAC009S0002W0263.wav'))
    parser.add_argument('--prompt', default='Transcribe the speech to text.')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    tokenizer = ensure_special_tokens(tokenizer)

    mimo_args = MiMoAudioArguments(
        model_name_or_path=args.checkpoint,
        sosp_idx=tokenizer.convert_tokens_to_ids('<|sosp|>'),
        eosp_idx=tokenizer.convert_tokens_to_ids('<|eosp|>'),
        empty_idx=tokenizer.convert_tokens_to_ids('<|empty|>'),
        sostm_idx=tokenizer.convert_tokens_to_ids('<|sostm|>'),
        eostm_idx=tokenizer.convert_tokens_to_ids('<|eostm|>'),
        eot_idx=tokenizer.convert_tokens_to_ids('<|eot|>'),
    )

    model = MiMoAudioForCausalLM.from_pretrained(
        args.checkpoint,
        args=mimo_args,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=args.device,
    )
    model.eval()
    if not hasattr(model, '_supports_cache_class'):
        model._supports_cache_class = True
    if model.generation_config is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)

    audio_tokenizer = MiMoAudioTokenizer.from_pretrained(os.path.join(root, 'model', 'MiMo-Audio-Tokenizer'))
    audio_tokenizer.eval().bfloat16().to(args.device)

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
    ).to(args.device)

    group_size = 4
    audio_channels = 8
    speech_zeroemb_idx = [1024, 1024, 128, 128, 128, 128, 128, 128]
    empty_idx = tokenizer.convert_tokens_to_ids('<|empty|>')

    audio_tokenized = encode_audio(args.audio, audio_tokenizer, mel_transform, group_size, audio_channels)
    input_ids_2d = build_prompt_input_ids(
        tokenizer, args.prompt, audio_tokenized, group_size, audio_channels, speech_zeroemb_idx, empty_idx
    )
    print(f'prompt input_ids shape: {tuple(input_ids_2d.shape)}')
    # MiMoAudioForCausalLM.generate uses generation_config.max_length directly.
    prompt_groups = input_ids_2d.shape[1] // group_size
    model.generation_config.max_length = prompt_groups + args.max_new_tokens

    # reshape to [1, T * group_size * (audio_channels + 1)] as expected by MiMoAudioForCausalLM.generate
    input_ids_1d = input_ids_2d.unsqueeze(0).transpose(1, 2).reshape(1, -1).to(args.device)

    global_sampler = MiMoSampler(do_sample=False)
    local_sampler = MiMoSampler(do_sample=False)

    with torch.inference_mode():
        generated = model.generate(
            input_ids_1d,
            max_new_tokens=args.max_new_tokens,
            global_sampler=global_sampler,
            local_sampler=local_sampler,
        )

    print(f'generated 1d min={generated[0].min().item()} max={generated[0].max().item()} len={generated[0].shape[-1]}')
    generated_text = decode_generated(generated[0], group_size, audio_channels, tokenizer)
    print('--- generated ---')
    print(generated_text)


if __name__ == '__main__':
    main()
