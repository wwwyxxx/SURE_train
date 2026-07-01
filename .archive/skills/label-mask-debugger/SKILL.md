---
name: label-mask-debugger
description: Debug model training failures caused by label shift, loss masks, padding, special tokens, or target-token alignment. Use when loss/accuracy looks wrong, padding is suspected, a custom template/collator is used, or a multimodal adapter needs proof that only the intended assistant tokens are included in the loss.
---

# Label Mask Debugger

Use this skill when training loss, token accuracy, or generation quality suggests the model may be learning the wrong labels. The goal is to prove whether the encoded training sample has:

```text
loss labels == target tokens + EOS
loss mask covers exactly those labels
padding/blank/control tokens are not in the loss
input, labels, and masks have compatible lengths
```

## Workflow

1. Inspect the template encode path and collator first. Find where `input_ids`, `labels`, `loss_mask` or task-specific masks are created.
2. Build or run a debug script that encodes raw dataset rows without training.
3. Print the masked labels, their decoded text, mask positions, target token ids, and any bad special tokens inside the mask.
4. Interpret results before changing training hyperparameters.

For Kimi-Audio in this repo, use:

```bash
python debug_encode_sample.py \
  --dataset /workspace/data/reprodata_asr_zh_existing.jsonl \
  --num-samples 5
```

The bundled script `scripts/debug_kimi_audio_encode.py` is a copyable reference for Kimi-Audio style two-stream labels.

## Required Checks

For each sampled row, compute and print:

```text
input_len
labels_len
loss_mask_sum
target_token_len_plus_eos
mask_positions
masked_label_ids
target_ids_plus_eos
masked_ids_match_target_plus_eos
masked_decoded_skip_special
bad_special_in_masked_labels
```

For multimodal models, also print modality masks such as `is_continuous_mask_sum` so you can verify the modality span exists and has plausible length.

## Pass Criteria

The sample-level check passes when:

```text
masked_ids_match_target_plus_eos: True
bad_special_in_masked_labels: []
loss_mask_sum == target_token_len_plus_eos
input_len == labels_len == mask_len
```

For batch size 1, traditional padding is usually not the cause if these checks pass. For batch size greater than 1, additionally inspect padded batch tensors after the collator:

```text
padded label positions are ignored
attention_mask is correct
loss mask is false on padding
sequence lengths did not shift labels across samples
```

## Failure Patterns

If `masked_ids_match_target_plus_eos` is false:

- Check next-token shift direction.
- Check whether the mask was shifted together with labels.
- Decode both masked labels and expected target ids token by token.

If `bad_special_in_masked_labels` is non-empty:

- Exclude `pad`, blank, separator, and message-end tokens from the loss.
- Verify assistant EOS is the intended EOS and not a stream-control token.

If lengths differ:

- Inspect the custom template and collator.
- Make sure every stream or field appends the same number of positions.
- Avoid adding padding before label/mask shift unless the collator handles it deliberately.

If all checks pass but training is poor:

- Treat padding/mask as unlikely root cause.
- Next inspect checkpoint save/load coverage, logits head selection, initialization quality, learning-rate grouping, and inference prompt parity.

## Kimi-Audio Notes

For the local Kimi-Audio adapter:

- `text_labels` are created by shifting `text_input_ids` left by one and appending `pad`.
- `text_loss_mask` is shifted the same way.
- Assistant text tokens and `kimia_text_eos` should be the only masked labels.
- `kimia_text_blank`, `msg_end`, and `pad` should not appear in masked labels.
- `is_continuous_mask_sum` should equal the number of audio feature positions inserted for the wav.
