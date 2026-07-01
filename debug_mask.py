import json
import os
import sys

_ROOT = os.path.abspath(os.path.dirname(__file__))
_MIMO_AUDIO_SRC = os.path.join(_ROOT, 'MiMo-Audio', 'src')
if _MIMO_AUDIO_SRC not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_SRC)

import torch
from transformers import AutoTokenizer
from mimo_audio.process_speechdata import InputSegment


def _encode_one(tokenizer, audio_tokens, response_text, group_size=4, audio_channels=8,
                speech_zeroemb_idx=None, empty_idx=None):
    if speech_zeroemb_idx is None:
        speech_zeroemb_idx = [1024, 1024, 128, 128, 128, 128, 128, 128]
    if empty_idx is None:
        empty_idx = tokenizer.convert_tokens_to_ids('<|empty|>')

    prompt_text = 'Transcribe the speech to text.'
    audio_tokenized = torch.tensor(audio_tokens, dtype=torch.long)
    segments = [
        InputSegment(text='<|im_start|>user\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(audio=audio_tokenized, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text=prompt_text, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_end|>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<|im_start|>assistant\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text='<think>\n\n</think>\n', speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
        InputSegment(text=response_text, speech_zeroemb_idx=speech_zeroemb_idx, text_zeroemb_idx=empty_idx),
    ]
    input_ids = torch.cat([seg.to_input_id(tokenizer, group_size, audio_channels) for seg in segments], dim=1)

    text_ids = input_ids[0, ::group_size].clone()
    T_groups = text_ids.shape[0]
    labels = text_ids.clone()
    text_loss_mask = torch.zeros(T_groups, dtype=torch.bool)

    assistant_header_ids = torch.tensor(
        tokenizer('<|im_start|>assistant\n', add_special_tokens=False)['input_ids'],
        dtype=text_ids.dtype,
    )
    for start in range(T_groups - len(assistant_header_ids) + 1):
        if torch.equal(text_ids[start:start + len(assistant_header_ids)], assistant_header_ids):
            think_prefix_ids = tokenizer('<think>\n\n</think>\n', add_special_tokens=False)['input_ids']
            loss_start = start + len(assistant_header_ids) + len(think_prefix_ids)
            if loss_start < T_groups:
                text_loss_mask[loss_start:] = True
            break

    labels = torch.cat([labels[1:], torch.tensor([tokenizer.pad_token_id or 0], dtype=labels.dtype)])
    return input_ids, text_ids, labels, text_loss_mask


tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(_ROOT, 'output/mimo_audio_overfit100_7gpu_v2/v6-20260628-084014/checkpoint-14'),
    trust_remote_code=True,
)
for tok in ['<|sosp|>', '<|eosp|>', '<|empty|>', '<|Human|>', '<|SpeechLM|>', '<|sostm|>', '<|eostm|>', '<|eot|>']:
    if tok not in tokenizer.get_vocab():
        tokenizer.add_tokens([tok], special_tokens=True)

with open(os.path.join(_ROOT, 'example/ASR_BAC009S0002W0263_overfit100.jsonl')) as f:
    row = json.loads(f.readline())

# Use a dummy audio token stream of one group of frames (length arbitrary,
# must be a multiple of audio_channels*group_size). The mask only depends on text.
dummy_audio = torch.full((4 * 8 * 4,), 1024, dtype=torch.long)
dummy_audio[4 * 8:] = 128  # channels 2-7 use 128 as empty
input_ids, text_ids, labels, text_loss_mask = _encode_one(tokenizer, dummy_audio.tolist(), row['txt'])
T_groups = text_ids.shape[0]

print('input_ids shape:', tuple(input_ids.shape))
print('text_ids shape:', tuple(text_ids.shape))
print('labels shape:', tuple(labels.shape))
print('text_loss_mask sum:', int(text_loss_mask.sum()), '/', T_groups)
print('\nFull text channel decoded:')
print(tokenizer.decode(text_ids.tolist(), skip_special_tokens=False))

assistant_header_ids = tokenizer('<|im_start|>assistant\n', add_special_tokens=False)['input_ids']
think_prefix_ids = tokenizer('<think>\n\n</think>\n', add_special_tokens=False)['input_ids']
print('\nAssistant header ids:', len(assistant_header_ids), assistant_header_ids)
print('Think prefix ids:', len(think_prefix_ids), think_prefix_ids)

print('\nPositions marked as loss (text_loss_mask=True):')
for i, (tid, mask) in enumerate(zip(text_ids.tolist(), text_loss_mask.tolist())):
    if mask:
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        print(f'  group {i}: {tid} -> {repr(tok)}')

print('\nAll text groups with index:')
for i, tid in enumerate(text_ids.tolist()):
    tok = tokenizer.decode([tid], skip_special_tokens=False)
    print(f'  {i:3d}: {tid:6d} {repr(tok):20s} {"LOSS" if text_loss_mask[i] else ""}')
