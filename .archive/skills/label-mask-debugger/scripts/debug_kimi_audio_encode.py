import argparse
import json
import os
import sys

"""Debug Kimi-Audio style text labels and text_loss_mask.

Copy this script into a project root that has:
- custom/kimi_audio_swift_register.py
- model/Qwen2.5-7B
- Kimi-Audio dependencies importable from the register file

It validates that masked labels equal target text tokens plus kimia_text_eos and
that pad/blank/message-end tokens are not included in the text loss mask.
"""


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return f'<decode_error:{token_id}>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/reprodata_asr_zh_existing.jsonl')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()

    root = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, root)

    import custom.kimi_audio_swift_register as reg

    model_dir = os.path.join(root, 'model/Qwen2.5-7B')
    dataset = args.dataset if os.path.isabs(args.dataset) else os.path.join(root, args.dataset)
    tokenizer = reg.AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    reg._patch_kimia_special_tokens(tokenizer)

    template = object.__new__(reg.KimiAudioTextTemplate)
    template.processor = tokenizer
    template.tokenizer = tokenizer
    template.extra_tokens = reg.instantiate_extra_tokens(tokenizer)

    with open(dataset, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]

    selected = rows[args.start:args.start + args.num_samples]
    extra = template.extra_tokens
    special_names = {
        extra.pad: 'pad',
        extra.kimia_text_blank: 'kimia_text_blank',
        extra.kimia_text_eos: 'kimia_text_eos',
        extra.kimia_user_msg_start: 'kimia_user_msg_start',
        extra.kimia_assistant_msg_start: 'kimia_assistant_msg_start',
        extra.msg_end: 'msg_end',
        extra.media_begin: 'media_begin',
        extra.media_end: 'media_end',
        extra.kimia_speech_ct_id: 'kimia_speech_ct_id',
    }

    for absolute_idx, row in enumerate(selected, start=args.start):
        processed = reg.KimiAudioASRPreprocessor().preprocess(row)
        encoded = template._encode(type('Inputs', (), processed))

        input_ids = encoded['input_ids']
        text_input_ids = encoded['text_input_ids']
        labels = encoded['labels']
        text_loss_mask = encoded['text_loss_mask'].bool()
        is_continuous_mask = encoded['is_continuous_mask'].bool()
        masked_positions = text_loss_mask.nonzero(as_tuple=False).flatten().tolist()
        masked_label_ids = labels[text_loss_mask].tolist()
        target = row.get('txt') or row.get('text') or row.get('response') or ''
        target_ids = template._tokenize_text(target) + [extra.kimia_text_eos]

        print('=' * 80)
        print(f'index: {absolute_idx}')
        print(f'wav: {row.get("wav")}')
        print(f'target: {target}')
        print(f'input_len: {input_ids.numel()}')
        print(f'text_input_len: {text_input_ids.numel()}')
        print(f'labels_len: {labels.numel()}')
        print(f'text_loss_mask_sum: {int(text_loss_mask.sum().item())}')
        print(f'is_continuous_mask_sum: {int(is_continuous_mask.sum().item())}')
        print(f'target_token_len_plus_eos: {len(target_ids)}')
        print(f'mask_positions: {masked_positions}')
        print(f'masked_label_ids: {masked_label_ids}')
        print(f'target_ids_plus_eos: {target_ids}')
        print(f'masked_ids_match_target_plus_eos: {masked_label_ids == target_ids}')
        print(f'masked_decoded_skip_special: {tokenizer.decode(masked_label_ids, skip_special_tokens=True)}')
        print('masked_tokens:')
        for pos, token_id in zip(masked_positions, masked_label_ids):
            name = special_names.get(token_id, '')
            decoded = _decode_token(tokenizer, token_id)
            print(f'  pos={pos:04d} id={token_id} {name} decoded={decoded!r}')

        bad_special = [tid for tid in masked_label_ids if tid in {extra.pad, extra.kimia_text_blank, extra.msg_end}]
        print(f'bad_special_in_masked_labels: {bad_special}')


if __name__ == '__main__':
    main()
