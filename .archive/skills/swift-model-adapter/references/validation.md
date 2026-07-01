# Validation For ms-swift Model Adapters

Use these checks when building a custom `--custom_register_path` integration for ms-swift. Run them in order. Do not proceed to full training until the earlier checks pass.

Prefer the bundled script for repeatable checks:

```bash
python skills/swift-model-adapter/scripts/validate_swift_adapter.py \
  --custom-register-path custom/kimi_audio_swift_register.py \
  --model /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B \
  --model-type kimi_audio_text \
  --template kimi_audio_text \
  --dataset kimi_audio_asr_overfit100 \
  --stage cpu \
  --required-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels \
  --required-batch-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels \
  --assert-text-audio-shapes
```

For GPU forward/backward:

```bash
python skills/swift-model-adapter/scripts/validate_swift_adapter.py \
  --custom-register-path custom/kimi_audio_swift_register.py \
  --model /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B \
  --model-type kimi_audio_text \
  --template kimi_audio_text \
  --dataset kimi_audio_asr_overfit100 \
  --stage forward \
  --required-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels \
  --required-batch-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels \
  --assert-text-audio-shapes \
  --must-train-prefix whisper_model. model.vq_adaptor. mimo_output. \
  --must-freeze-prefix model.layers.
```

## 1. Import And Registration

Purpose: catch missing dependencies, syntax errors, wrong registration names, and stale custom files.

```bash
python - <<'PY'
from swift.utils import import_external_file
from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING
from swift.llm.dataset.register import DATASET_MAPPING

custom_register_path = "custom/kimi_audio_swift_register.py"
model_type = "kimi_audio_text"
template_type = "kimi_audio_text"
dataset_name = "kimi_audio_asr_overfit100"

import_external_file(custom_register_path)

print("model registered:", model_type in MODEL_MAPPING)
print("template registered:", template_type in TEMPLATE_MAPPING)
print("dataset registered:", any(k == dataset_name or (isinstance(k, tuple) and dataset_name in k) for k in DATASET_MAPPING))

assert model_type in MODEL_MAPPING
assert template_type in TEMPLATE_MAPPING
PY
```

## 2. Dataset Load

Purpose: confirm the dataset resolves through ms-swift and the preprocessor emits the expected row shape.

```bash
python - <<'PY'
import os
from swift.utils import import_external_file
from swift.llm import load_dataset

import_external_file("custom/kimi_audio_swift_register.py")
ds = load_dataset(["kimi_audio_asr_overfit100"])[0]
row = ds[0]

print(ds)
print(row)
assert len(ds) > 0
assert "messages" in row
assert row["messages"][0]["role"] == "user"
assert row["messages"][-1]["role"] == "assistant"

if "audios" in row:
    assert os.path.exists(row["audios"][0]), row["audios"][0]
    assert isinstance(row["messages"][0]["content"], str)
    assert "<audio>" in row["messages"][0]["content"]
PY
```

For ms-swift 3.12.x, dataset preprocessing must not emit list-of-dict message content mixed with string assistant content. If `dataset.map` fails with `cannot mix list and non-list`, change the preprocessor to string `messages.content` plus top-level `audios/images/videos`.

## 2.5 Kimi-Audio Tokenizer Preflight

Purpose: catch Qwen tokenizer vs Kimi tokenizer interface mismatches before template encoding.

```bash
python - <<'PY'
from transformers import AutoTokenizer
from swift.utils import import_external_file
from kimia_infer.utils.special_tokens import instantiate_extra_tokens

import_external_file("custom/kimi_audio_swift_register.py")
import custom.kimi_audio_swift_register as custom

tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B",
    trust_remote_code=True,
)
if hasattr(custom, "_patch_kimia_special_tokens"):
    custom._patch_kimia_special_tokens(tokenizer)

extra = instantiate_extra_tokens(tokenizer)
print(extra)
assert tokenizer.encode("Transcribe the speech to text.", add_special_tokens=False)
assert extra.media_begin == 151661
assert extra.media_end == 151663
assert extra.kimia_text_blank == 151666
assert extra.kimia_text_eos == 151667
assert extra.pad == 152063
PY
```

If this fails with missing `pad_id`, patch the Qwen tokenizer with Kimi special token mappings. If it fails with unexpected `bos`, do not pass Kimi-tokenizer-only kwargs to HuggingFace fast tokenizers.

## 3. Template Encode

Purpose: validate single-row conversion into model input fields. This is often where multimodal adapters fail.

```bash
python - <<'PY'
from swift.utils import import_external_file
from swift.llm import get_model_tokenizer, get_template, load_dataset

custom_register_path = "custom/kimi_audio_swift_register.py"
model = "/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B"
model_type = "kimi_audio_text"
template_type = "kimi_audio_text"
dataset = "kimi_audio_asr_overfit100"

import_external_file(custom_register_path)
_, processor = get_model_tokenizer(model, model_type=model_type, load_model=False)
template = get_template(template_type, processor)
template.set_mode("train")

ds = load_dataset([dataset])[0]
encoded = template.encode(ds[0])

for k, v in encoded.items():
    print(k, type(v), getattr(v, "shape", None))

assert "input_ids" in encoded
assert "labels" in encoded
assert encoded["input_ids"].shape == encoded["text_input_ids"].shape
assert encoded["input_ids"].shape == encoded["is_continuous_mask"].shape
assert encoded["labels"]["text_labels"].shape == encoded["input_ids"].shape
assert encoded["labels"]["text_loss_mask"].shape == encoded["input_ids"].shape
assert encoded["labels"]["text_loss_mask"].sum().item() > 0
assert encoded["is_continuous_mask"].sum().item() > 0
PY
```

## 4. Collator

Purpose: confirm `template.data_collator` returns exactly the batch keys accepted by model forward.

```bash
python - <<'PY'
from swift.utils import import_external_file
from swift.llm import get_model_tokenizer, get_template, load_dataset

import_external_file("custom/kimi_audio_swift_register.py")
_, processor = get_model_tokenizer(
    "/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B",
    model_type="kimi_audio_text",
    load_model=False,
)
template = get_template("kimi_audio_text", processor)
template.set_mode("train")
row = load_dataset(["kimi_audio_asr_overfit100"])[0][0]
batch = template.data_collator([template.encode(row)])

for k, v in batch.items():
    if isinstance(v, dict):
        print(k, {kk: getattr(vv, "shape", None) for kk, vv in v.items()})
    else:
        print(k, type(v), getattr(v, "shape", None))

required = {"input_ids", "text_input_ids", "is_continuous_mask", "whisper_input_feature", "labels"}
assert required.issubset(batch.keys())
PY
```

## 5. Model Init

Purpose: validate full model construction, custom checkpoint initialization, and trainable/frozen module boundaries.

Run inside the target training container with GPU dependencies installed.

```bash
python - <<'PY'
from swift.utils import import_external_file
from swift.llm import get_model_tokenizer

import_external_file("custom/kimi_audio_swift_register.py")
model, _ = get_model_tokenizer(
    "/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B",
    model_type="kimi_audio_text",
)

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print("num trainable tensors:", len(trainable))
print("trainable examples:", trainable[:50])

assert any(n.startswith("whisper_model.") for n in trainable)
assert any(n.startswith("model.vq_adaptor.") for n in trainable)
assert any(n.startswith("mimo_output.") for n in trainable)
PY
```

If ms-swift later runs with `--train_type full`, also inspect the training log's `model_parameter_info`, because full training can reset `requires_grad` before applying freeze rules.

## 6. Forward And Backward

Purpose: catch device mismatch, shape mismatch, tuple-label issues, NaN loss, and missing gradient flow.

Run inside the target training container on a GPU.

```bash
python - <<'PY'
import torch
from swift.utils import import_external_file
from swift.llm import get_model_tokenizer, get_template, load_dataset

import_external_file("custom/kimi_audio_swift_register.py")
model, processor = get_model_tokenizer(
    "/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B",
    model_type="kimi_audio_text",
)
model.cuda()
model.train()

template = get_template("kimi_audio_text", processor)
template.set_mode("train")
row = load_dataset(["kimi_audio_asr_overfit100"])[0][0]
batch = template.data_collator([template.encode(row)])

def move(x):
    if torch.is_tensor(x):
        return x.cuda()
    if isinstance(x, dict):
        return {k: move(v) for k, v in x.items()}
    return x

batch = move(batch)
out = model(**batch)
print("loss:", out.loss.item())
assert torch.isfinite(out.loss)
out.loss.backward()

grad_names = [n for n, p in model.named_parameters() if p.grad is not None]
print("num grad tensors:", len(grad_names))
print("grad examples:", grad_names[:50])
assert grad_names
PY
```

## 7. Short Overfit

Purpose: verify the integration is trainable, not merely executable.

Use a repeated one-sample dataset and a very short run:

```bash
swift sft \
  --custom_register_path custom/kimi_audio_swift_register.py \
  --model /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B \
  --model_type kimi_audio_text \
  --dataset kimi_audio_asr_overfit100 \
  --train_type full \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 256 \
  --logging_steps 1 \
  --save_steps 50 \
  --output_dir output/adapter_overfit_debug \
  --report_to none
```

Pass condition: training enters the loop and loss clearly decreases over tens of steps. If loss is flat, inspect the loss mask, trainable parameter list, and whether the target head receives gradients.

## What To Automate

Automate these as fixed scripts:

- Import and registration.
- Dataset load.
- Template encode shape/mask assertions.
- Collator key/shape assertions.
- Optional full model init.
- Optional forward/backward on GPU.

Keep short overfit as a separate command because it is slower, writes checkpoints/logs, and depends on GPU memory.
