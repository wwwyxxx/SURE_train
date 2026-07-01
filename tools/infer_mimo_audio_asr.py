#!/usr/bin/env python3
"""ASR inference on a few samples using a MiMo-Audio checkpoint."""
import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_MIMO_AUDIO_ROOT = os.path.join(_ROOT, 'MiMo-Audio')
if _MIMO_AUDIO_ROOT not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_ROOT)

# MiMoAudioForCausalLM defines generate() but does not inherit GenerationMixin in
# newer transformers, so we patch it in before MimoAudio instantiates the model.
from transformers import GenerationMixin  # noqa: E402
from src.mimo_audio.modeling_mimo_audio import MiMoAudioForCausalLM  # noqa: E402

if GenerationMixin not in MiMoAudioForCausalLM.__bases__:
    MiMoAudioForCausalLM.__bases__ = (GenerationMixin,) + MiMoAudioForCausalLM.__bases__

# GenerationMixin's _has_unfinished_sequences has a different signature than what
# MiMoAudioForCausalLM.generate() expects; override with a compatible version.
def _has_unfinished_sequences_mimo(self, this_peer_finished, synced_gpus, device, cur_len, max_length):
    return not this_peer_finished and cur_len < max_length

MiMoAudioForCausalLM._has_unfinished_sequences = _has_unfinished_sequences_mimo

from src.mimo_audio.mimo_audio import MimoAudio  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to MiMo-Audio checkpoint dir')
    parser.add_argument('--dataset', default='data/combined_asr_aishell-1.jsonl')
    parser.add_argument('--audio-tokenizer', default='model/MiMo-Audio-Tokenizer')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_ROOT, args.checkpoint)
    dataset = args.dataset if os.path.isabs(args.dataset) else os.path.join(_ROOT, args.dataset)
    audio_tokenizer = args.audio_tokenizer if os.path.isabs(args.audio_tokenizer) else os.path.join(_ROOT, args.audio_tokenizer)

    print(f'[infer] Loading model from {checkpoint}')
    model = MimoAudio(checkpoint, audio_tokenizer, device=args.device)

    print(f'[infer] Reading dataset {dataset}')
    with open(dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    selected = rows[args.start:args.start + args.num_samples]
    for idx, row in enumerate(selected, start=args.start):
        wav = row['wav'] if os.path.isabs(row['wav']) else os.path.join(_ROOT, row['wav'])
        prompt = row.get('prompt') or 'Transcribe the speech to text.'
        target = row.get('txt') or row.get('text') or row.get('response') or ''

        pred = model.audio_understanding_sft(wav, prompt)
        # Strip possible think tags / eot
        pred = pred.split('<|eot|>')[0].strip()
        ok = 'OK' if pred == target else 'BAD'
        print(f'\n[{idx}] {ok}')
        print(f'target: {target}')
        print(f'pred:   {pred}')


if __name__ == '__main__':
    main()
