# Adapter Harness

Use this reference when turning a model adaptation into a repeatable Codex workflow.

## Required Deliverables

For each new model adapter, produce these project-local artifacts:

```text
custom/<model>_swift_register.py
run_<model>_overfit.sh
infer_<model>_overfit.py
example/<model>_overfit*.jsonl
```

The register file must be external and loaded with `--custom_register_path`; do not edit `ms-swift/swift/` unless explicitly requested.

## Gate Order

Run gates in this order:

1. `cpu`: import/register, dataset load, template encode, collator.
2. `forward`: model init, trainable/frozen prefixes, loss finite, backward creates gradients.
3. `overfit`: one representative sample repeated 50-100 times, batch size 1, no validation split.
4. `infer`: load the overfit checkpoint and decode the same sample.

Do not proceed to the next gate until the current gate passes.

## Harness Runner

Use `scripts/run_adapter_harness.py` to print or run the standard sequence.

Kimi-Audio CPU gate:

```bash
python skills/swift-model-adapter/scripts/run_adapter_harness.py \
  --custom-register-path custom/kimi_audio_swift_register.py \
  --model /workspace/model/Qwen2.5-7B \
  --model-type kimi_audio_text \
  --template kimi_audio_text \
  --dataset kimi_audio_asr_overfit100 \
  --stages cpu \
  --required-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels text_loss_mask \
  --required-batch-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels text_loss_mask \
  --assert-text-audio-shapes
```

Kimi-Audio forward gate:

```bash
python skills/swift-model-adapter/scripts/run_adapter_harness.py \
  --custom-register-path custom/kimi_audio_swift_register.py \
  --model /workspace/model/Qwen2.5-7B \
  --model-type kimi_audio_text \
  --template kimi_audio_text \
  --dataset kimi_audio_asr_overfit100 \
  --stages forward \
  --required-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels text_loss_mask \
  --required-batch-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels text_loss_mask \
  --assert-text-audio-shapes \
  --must-train-prefix whisper_model. model.vq_adaptor. mimo_output. \
  --must-freeze-prefix model.layers.
```

Kimi-Audio overfit and infer gates:

```bash
python skills/swift-model-adapter/scripts/run_adapter_harness.py \
  --custom-register-path custom/kimi_audio_swift_register.py \
  --model /workspace/model/Qwen2.5-7B \
  --model-type kimi_audio_text \
  --template kimi_audio_text \
  --dataset kimi_audio_asr_overfit100 \
  --stages overfit infer \
  --train-type full \
  --freeze-llm true \
  --freeze-vit false \
  --freeze-aligner false \
  --bf16 \
  --gradient-checkpointing \
  --num-train-epochs 3 \
  --save-steps 300 \
  --save-total-limit 1 \
  --save-only-model \
  --output-dir output/kimi_audio_asr_overfit100 \
  --infer-script infer_overfit.py \
  --checkpoint output/kimi_audio_asr_overfit100/<run>/checkpoint-300 \
  --audio example/BAC009S0002W0263.wav
```

Use `--dry-run` first when generating commands for a new adapter.

## Pass Criteria

CPU gate passes when:

- model/template/dataset registrations are true.
- first dataset row contains valid messages and media paths.
- encoded keys match the model forward signature.
- collated batch keys match training input keys.
- task masks have positive support, for example `text_loss_mask.sum() > 0`.

Forward gate passes when:

- model initialization reports nonzero copied/matched weights if initialized from another checkpoint.
- intended trainable prefixes have parameters requiring grad.
- intended frozen prefixes do not require grad.
- forward returns finite scalar `loss`.
- backward creates gradients on intended modules.

Overfit gate passes when:

- training enters the loop.
- loss decreases or reaches a very small value on the repeated one-sample dataset.
- checkpoint saving succeeds or `--save_strategy no` was intentionally used.

Infer gate passes when:

- checkpoint loads with expected missing/unexpected key counts.
- prompt-only inference uses the same formatting as training minus assistant answer tokens.
- decoded output matches or nearly matches the overfit label.

## Trainable-Only Checkpoints

For full fine-tuning wrappers with mostly frozen base weights, avoid saving the entire base model. Override the custom model wrapper's `save_pretrained` in `custom/<model>_swift_register.py` and filter `state_dict` to parameters with `requires_grad=True`.

Pattern:

```python
def save_pretrained(self, save_directory, *args, state_dict=None, **kwargs):
    if state_dict is None:
        state_dict = self.state_dict()
    trainable = {n for n, p in self.named_parameters() if p.requires_grad}
    state_dict = {n: t for n, t in state_dict.items() if n in trainable}
    return super().save_pretrained(save_directory, *args, state_dict=state_dict, **kwargs)
```

Inference must first reconstruct the base model from the original model path, then load the trainable-only checkpoint with `strict=False`. Store a manifest such as `trainable_state_keys.json` so users know the checkpoint is a delta, not a standalone full model.

## Common Failure Map

`ArrowInvalid: cannot mix list and non-list`:
Use string `messages[*].content` plus top-level `audios/images/videos`.

`labels.device` or tuple/dict labels fail in trainer:
Keep `labels` as a Tensor when possible; pass extra masks as top-level tensors.

Loss finite but `token_acc` low:
Default ms-swift `token_acc` may use the wrong labels/mask for custom heads. Add adapter-local metric patch or ignore it; do not modify core ms-swift.

Loss is zero or tiny from the start:
Check that the loss mask has positive support and labels are shifted correctly.

Forward works but overfit does not learn:
Check trainable prefixes, gradient names, and whether copied initialization leaves the target head random.

Inference repeats or over-generates:
Check prompt-only formatting and stop token probability. Do not use training `_encode()` with an empty assistant answer for inference.
