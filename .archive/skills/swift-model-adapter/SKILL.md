---
name: swift-model-adapter
description: Adapt a new or non-standard model into the local ms-swift training framework. Use when Codex needs to register a custom ms-swift model/template/dataset, implement custom initialization from another checkpoint, bridge multimodal or non-standard batch fields, add model-side loss glue, or create a one-sample overfit path for validating a new ms-swift integration.
---

# Swift Model Adapter

Use this skill to add a new model family to `ms-swift` without modifying core framework files. Prefer an external `--custom_register_path` file unless the user explicitly wants an upstream-style patch inside `ms-swift/swift/`.

## Harness Contract

When the user asks to adapt a model to swift/ms-swift, do not stop after writing registration code. Deliver a runnable harness:

```text
custom/<model>_swift_register.py
run_<model>_overfit.sh
infer_<model>_overfit.py
example/<model>_overfit*.jsonl
```

Then run or provide the exact commands for these gates:

```text
cpu -> forward -> overfit -> infer
```

Read `references/harness.md` when building or validating the harness. Use `scripts/run_adapter_harness.py` for staged checks and `scripts/validate_swift_adapter.py` for direct validation.

## Inspect First

Use CodeGraph or `rg` to inspect the local ms-swift version before editing:

- Model registration: `ms-swift/swift/llm/model/register.py`
- Template registration: `ms-swift/swift/llm/template/register.py`
- Template metadata signature: `ms-swift/swift/llm/template/template_meta.py`
- Dataset registration: `ms-swift/swift/llm/dataset/register.py`
- Dataset preprocessor schema handling: `ms-swift/swift/llm/dataset/preprocessor/core.py`
- Custom examples: `ms-swift/examples/custom/`
- Multimodal example: `ms-swift/examples/custom/my_qwen2_5_omni/my_register.py`
- SFT data flow: `ms-swift/swift/llm/train/sft.py`
- Template base/collator: `ms-swift/swift/llm/template/base.py`

Confirm the model's native forward signature, expected batch keys, tokenizer/processor needs, and whether it returns a standard `loss/logits` or custom outputs.

## Preferred File Layout

Create one external register file in the project, for example:

```text
custom/<model>_swift_register.py
```

Training should then use:

```bash
swift sft \
  --custom_register_path custom/<model>_swift_register.py \
  --model <base-or-custom-model-dir> \
  --model_type <custom_model_type> \
  --dataset <dataset-name-or-path> \
  ...
```

## Register Model

Implement and register a custom loader:

```python
from swift.llm import Model, ModelGroup, ModelMeta, register_model

def get_model_tokenizer_custom(model_dir, model_info, model_kwargs, load_model=True, **kwargs):
    # 1. Load tokenizer/processor.
    # 2. Build the target config.
    # 3. Instantiate the target model class.
    # 4. Apply custom checkpoint initialization.
    # 5. Return (model, tokenizer_or_processor).
    ...

register_model(ModelMeta(
    model_type='my_model',
    model_groups=[ModelGroup([Model(model_path='<default-local-path>')])],
    template='my_model',
    get_function=get_model_tokenizer_custom,
    is_multimodal=True,
    model_arch='my_model',
))
```

Do not force a model into `AutoModelForCausalLM.from_pretrained` if it needs non-standard initialization. Use a custom `get_function`.

## Register Model Arch

For multimodal/full training, register module groups so ms-swift freeze and LoRA controls can work:

```python
from swift.llm import MultiModelKeys, register_model_arch

register_model_arch(MultiModelKeys(
    'my_model',
    language_model=['model.embed_tokens', 'model.layers', 'model.norm'],
    vision_tower=['audio_encoder_or_vision_tower'],
    aligner=['projector_or_adaptor', 'text_head_if_needed'],
    generator=['unused_generation_head_or_detokenizer'],
))
```

Important: for `--train_type full`, ms-swift sets all params trainable first, then applies `freeze_parameters` and `trainable_parameters`. Make the command explicit:

```bash
--train_type full \
--freeze_llm true \
--freeze_vit false \
--freeze_aligner false
```

Put unused output branches in `generator` so they remain frozen.

## Register Dataset

If raw data is simple JSONL, register a preprocessor:

```python
from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset

class MyPreprocessor(ResponsePreprocessor):
    def preprocess(self, row):
        return {
            'messages': [
                {'role': 'user', 'content': row['prompt']},
                {'role': 'assistant', 'content': row['response']},
            ],
            # Add modality fields if needed, e.g. audios/images/videos.
        }

register_dataset(DatasetMeta(
    dataset_path='<path>.jsonl',
    dataset_name='my_dataset',
    preprocess_func=MyPreprocessor(),
), exist_ok=True)
```

For ms-swift 3.12.x JSONL preprocessing, keep `messages[*].content` as strings and put media paths in top-level modality columns:

```python
{
    'messages': [
        {'role': 'user', 'content': f'{prompt} <audio>'},
        {'role': 'assistant', 'content': response},
    ],
    'audios': [wav_path],
}
```

Do not mix structured list content and string content in `messages.content` for this version. HuggingFace datasets/Arrow will fail during `dataset.map` with `pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values`. The template can read audio paths from `inputs.audios`.

## Register Template

Use a custom `Template` when the model needs non-standard fields beyond normal `input_ids/labels`.

```python
from swift.llm import Template, TemplateMeta, register_template

class MyTemplate(Template):
    support_padding_free = False

    def _encode(self, inputs):
        # Convert StdTemplateInputs into the model's expected single-sample dict.
        return {
            'input_ids': ...,
            'labels': ...,
            'custom_key': ...,
        }

    def data_collator(self, batch, *, padding_to=None):
        # Start with batch size 1 for fragile new integrations.
        assert len(batch) == 1
        return ...

register_template(TemplateMeta(
    template_type='my_model',
    prefix=[],
    prompt=[],
    chat_sep=[],
    template_cls=MyTemplate,
))
```

For the first integration, prefer a batch-size-1 collator. Add padding and multi-sample collation only after one-sample overfit works.

In ms-swift 3.12.x, `TemplateMeta` requires `prefix`, `prompt`, and `chat_sep`. If the custom `Template._encode` owns all formatting, pass empty lists for those fields. Always confirm this against the local `template_meta.py` rather than assuming the API from another ms-swift version.

## Custom Initialization

When initializing from a different checkpoint:

1. Build the target config first.
2. Instantiate the target model.
3. Load the source model state dict.
4. Copy only matching target parameters.
5. Handle embedding expansion by copying overlapping rows.
6. Leave new heads/adaptors random unless the user asks otherwise.

Pattern:

```python
def copy_partial(target_model, source_model):
    src = source_model.state_dict()
    dst = target_model.state_dict()
    with torch.no_grad():
        for src_name, src_tensor in src.items():
            dst_name = remap(src_name)
            if dst_name not in dst:
                continue
            if dst[dst_name].shape == src_tensor.shape:
                dst[dst_name].copy_(src_tensor.to(dst[dst_name].dtype))
            elif dst_name.endswith('embed_tokens.weight'):
                rows = min(dst[dst_name].shape[0], src_tensor.shape[0])
                dst[dst_name][:rows].copy_(src_tensor[:rows].to(dst[dst_name].dtype))
```

Log copied/skipped counts. Avoid silent all-random initialization.

## Custom Loss

If the native model returns tuple logits, multimodal outputs, or multiple heads, put loss glue in the model wrapper so ms-swift sees standard HuggingFace outputs:

```python
from transformers.modeling_outputs import CausalLMOutputWithPast

class MySFTModel(BaseModel):
    def forward(self, ..., labels=None, ...):
        outputs = super().forward(..., return_dict=True)
        logits = select_training_logits(outputs)
        loss = None
        if labels is not None:
            loss = compute_task_loss(logits, labels)
        return CausalLMOutputWithPast(loss=loss, logits=logits)
```

Do this instead of relying on `loss_type` when labels/logits are not standard `Tensor` causal LM labels.

Keep `labels` as a Tensor in the final collated batch whenever possible. ms-swift 3.12.x trainer expects `inputs["labels"].device` when the model returns its own loss. If the model also needs a task-specific mask, pass it as a separate top-level tensor such as `text_loss_mask`, and accept both `labels` and `text_loss_mask` in the model wrapper `forward`.

## One-Sample Overfit

Always create a tiny overfit path before full data training:

1. Pick one representative sample.
2. If using dataloader epochs, repeat it 50-100 times in a JSONL file.
3. Run `batch_size=1`, no validation split, short `max_length`.
4. Confirm training enters the loop and loss decreases quickly.

For single-GPU fragile tests:

```bash
swift sft \
  --custom_register_path custom/<model>_swift_register.py \
  --model <model-dir> \
  --model_type <custom_model_type> \
  --dataset <overfit-dataset-name-or-path> \
  --train_type full \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 256 \
  --logging_steps 1 \
  --report_to none
```

If OOM, first reduce sequence/audio length or freeze more modules. Do not increase batch complexity until the one-sample run is stable.

## Validation

For the full gate sequence and pass criteria, read `references/harness.md`. For detailed low-level checks and copyable validation commands, read `references/validation.md`.

Prefer the bundled scripts:

```bash
python skills/swift-model-adapter/scripts/run_adapter_harness.py --dry-run ...
python skills/swift-model-adapter/scripts/validate_swift_adapter.py ...
```

Use them when the user asks for automated validation, when a custom adapter fails before training, or before moving from one-sample overfit to larger data.

## Kimi-Audio Notes

For Kimi-Audio text-only ASR:

- Keep raw ASR rows as `wav/txt/prompt`.
- Convert rows online into string `messages` plus top-level `audios`; do not put list-of-dict audio content into `messages.content` on ms-swift 3.12.x.
- Use Qwen2.5-7B to initialize shared LLM layers.
- Load Whisper Encoder from local `whisper-large-v3` unless the user explicitly asks for random init. Kimi-Audio's `finetune_codes/model.py` hardcodes `WhisperEncoder("openai/whisper-large-v3")` inside `KimiAudioModel.__init__`; override or intercept that during wrapper initialization so offline runs do not call HuggingFace before the adapter can replace `whisper_model`.
- Leave adaptor/text head random if requested by the experiment.
- Ignore audio output head by excluding audio loss and freezing unused generator branches.
- Return only text logits/loss to ms-swift.

When the shared LLM is Qwen2.5 but the model architecture is Kimi-Audio:

- Use the Qwen tokenizer for normal text tokens so frozen Qwen embeddings remain aligned.
- Add Kimi-Audio special token ids to that tokenizer, matching Kimi's `tokenization_kimia.py` layout: base text vocab starts at 0, Kimi control tokens begin at `151643`, `kimia_token_offset` is `152064`, and pad is the last reserved token (`152063` for the common 421-token reserved set).
- Patch compatibility attributes expected by Kimi-Audio utilities: `tokenizer.special_tokens`, `tokenizer.pad_id`, and `tokenizer.pad_token_id`.
- Do not call HuggingFace fast tokenizer as `tokenizer.encode(text, bos=False, eos=False)`. Kimi tokenizer supports `bos/eos`; Qwen fast tokenizer does not. Use `add_special_tokens=False` for HuggingFace tokenizers.
- Verify `instantiate_extra_tokens(tokenizer)` returns correct ids before training, especially `media_begin=151661`, `media_end=151663`, `kimia_text_blank=151666`, `kimia_text_eos=151667`, and `pad=152063`.

Common non-OOM failures and source-level fixes:

- `TemplateMeta.__init__() missing prefix, prompt, chat_sep`: local ms-swift requires these fields; pass empty lists if custom `_encode` handles formatting.
- HuggingFace retries for `/openai/whisper-large-v3/resolve/main/config.json`: Kimi-Audio parent init hardcoded the remote Whisper id; redirect it to the local `model/whisper-large-v3` before or during superclass initialization.
- `pyarrow.lib.ArrowInvalid: cannot mix list and non-list`: preprocessor emitted mixed `messages.content` types; keep all message content strings and put media paths under `audios`.
- `Qwen2TokenizerFast has no attribute pad_id`: Kimi-Audio special token helper expects Kimi tokenizer fields; patch Qwen tokenizer with Kimi special token mapping and `pad_id`.
- `PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'bos'`: replace Kimi-tokenizer-only `encode(..., bos=False, eos=False)` with HF-compatible `encode(..., add_special_tokens=False)` when using Qwen tokenizer.
- `AttributeError: 'dict' object has no attribute 'device'` in `swift/trainers/trainers.py`: the custom collator returned `labels` as a dict; return tensor `labels` and put auxiliary masks like `text_loss_mask` at the top level.
