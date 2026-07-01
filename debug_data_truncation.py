import json
import os
import sys
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
_MIMO_AUDIO_SRC = os.path.join(_ROOT, 'MiMo-Audio', 'src')
if _MIMO_AUDIO_SRC not in sys.path:
    sys.path.insert(0, _MIMO_AUDIO_SRC)

from swift.llm import get_model_tokenizer, get_template
from custom.mimo_audio_swift_register import MiMoAudioASRPreprocessor


def main():
    dataset_path = os.path.join(_ROOT, 'data', 'combined_asr_aishell-1_cached.jsonl')
    model_dir = '/workspace/model/MiMo-Audio-7B-Base-merged'

    # Load model + tokenizer (needed by template)
    _, tokenizer = get_model_tokenizer(
        model_type='mimo_audio',
        model_id_or_path=model_dir,
        load_model=False,
    )

    # Build template exactly as training does
    template = get_template(
        template_type='mimo_audio',
        processor=tokenizer,
        max_length=2048,
        truncation_strategy='raise',
    )

    preprocessor = MiMoAudioASRPreprocessor()

    # Read first N raw rows
    rows = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            rows.append(json.loads(line))

    stats = []
    for idx, row in enumerate(rows):
        processed = preprocessor._process_row(row)
        if processed is None:
            continue

        # Mimic what ms-swift does: pass through template encode
        # Build a simple object with messages + objects
        class DummyInput:
            def __init__(self, messages, objects):
                self.messages = messages
                self.objects = objects
                self.audios = None
                self.videos = None
                self.images = None
                self.extra_kwargs = {}

        inp = DummyInput(processed['messages'], processed['objects'])
        encoded = template._encode(inp)

        input_ids = encoded['input_ids']
        labels = encoded['labels']
        text_loss_mask = encoded['text_loss_mask']

        seq_len = input_ids.shape[1]
        n_labels = labels.numel()
        n_valid_labels = (labels != -100).sum().item()
        n_text_positions = text_loss_mask.sum().item()

        stats.append({
            'idx': idx,
            'seq_len': seq_len,
            'n_labels': n_labels,
            'n_valid_labels': n_valid_labels,
            'n_text_positions': n_text_positions,
            'audio_tokens_len': len(processed['objects']['audio_tokens']),
        })

        print(f"[{idx}] audio_tokens={len(processed['objects']['audio_tokens']):6d} "
              f"seq_len={seq_len:5d} labels={n_valid_labels:4d}/{n_labels:4d} "
              f"text_positions={n_text_positions:4d} txt={row.get('txt','')[:30]}")

    print('\nSummary:')
    max_seq = max(s['seq_len'] for s in stats)
    min_valid = min(s['n_valid_labels'] for s in stats)
    print(f"max seq_len={max_seq}, min valid labels={min_valid}")
    if max_seq > 8192:
        print('WARNING: some sequences exceed 8192; model may truncate or OOM.')
    if min_valid == 0:
        print('WARNING: at least one sample has ALL labels == -100 (no loss computed).')
    elif min_valid < 5:
        print(f'WARNING: some samples have very few valid labels (min={min_valid}).')


if __name__ == '__main__':
    main()
