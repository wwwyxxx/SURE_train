# registered/02 - 模型注册

## 目标

当 ms-swift 已原生支持的模型需要按组件初始化时（例如用 Qwen2.5-7B 初始化 LLM、用 Whisper 初始化 audio encoder），生成本地自定义模型注册脚本，在运行时动态替换组件，不再离线拼接 `assembled_model` checkpoint。

## 什么时候需要这个阶段？

- `input.json` 中提供了 `component_paths`。
- 如果没有 `component_paths`，直接跳过本阶段，使用单一官方 `model_path` 进入 `registered/03-dataset-register.md`。

## Agent Checklist

- [ ] 读取 `outputs/{run_id}/registered_model_info.json`
- [ ] 确认 `base_model_path`（作为训练时 `--model` 的官方 checkpoint 路径）
- [ ] 确认 `component_paths`：每个组件的来源路径或 `"random"`
- [ ] 将 `component_paths` 写入 `outputs/{run_id}/component_paths.json`
- [ ] 调用生成器：
  ```bash
  python .swift-adapter-agent/executors/generate_registered_model_register.py \
    --model-family {model_family} \
    --model-type {model_type} \
    --base-model-path {base_model_path} \
    --component-paths-json outputs/{run_id}/component_paths.json \
    --output outputs/{run_id}/custom/{model_family}_registered_model_register.py
  ```
- [ ] 语法检查：
  ```bash
  python3 -m py_compile outputs/{run_id}/custom/{model_family}_registered_model_register.py
  ```
- [ ] 更新 `registered_model_info.json`：
  - `registered_model_register_path`
  - `registered_model_type`（例如 `{model_type}_custom`）
- [ ] 更新 `pipeline_state.json`

## 输入

来自 `registered_model_info.json`：

```json
{
  "model_family": "qwen2_5_omni",
  "model_type": "qwen2_5_omni",
  "base_model_path": "/workspace/model/Qwen2.5-Omni-7B",
  "component_paths": {
    "language_model": "/workspace/model/Qwen2.5-7B",
    "vision_tower": "/workspace/model/whisper-large-v3"
  }
}
```

## 输出

- `outputs/{run_id}/custom/{model_family}_registered_model_register.py`
- 更新后的 `outputs/{run_id}/registered_model_info.json`：

```json
{
  "model_family": "qwen2_5_omni",
  "model_type": "qwen2_5_omni",
  "base_model_path": "/workspace/model/Qwen2.5-Omni-7B",
  "registered_model_type": "qwen2_5_omni_custom",
  "registered_model_register_path": "outputs/{run_id}/custom/qwen2_5_omni_registered_model_register.py",
  "component_paths": {
    "language_model": "/workspace/model/Qwen2.5-7B",
    "vision_tower": "/workspace/model/whisper-large-v3"
  }
}
```

## 模型适配器

生成器会根据 `model_type` 自动选择适配器：
`.swift-adapter-agent/templates/model_register/{model_type}.adapter.py`

如果没有专用适配器，则回退到 `default.adapter.py`，执行整子模块替换。

当前已实现的适配器：

- `qwen2_5_omni`：替换 `thinker.model` 为 Qwen2.5-7B，替换 `thinker.audio_tower` 的 encoder 为 Whisper，保留 Omni 特有的 `ln_post`/`proj`/`audio_bos_eos_token`。
- `qwen2_audio`：替换 `language_model` 和 `audio_tower`，并包含 `Qwen2AudioTemplateFixed` 模板修复。

## 关键注意事项

### 1. 词表与 embedding / lm_head 必须保持 base model 的大小，并对齐公共 token

当替换 `language_model` 为另一个 base model 时，新 LLM 的 `vocab_size` 通常与 base model 不同。此时**不能**直接把 `vocab_size` 改成新 LLM 的大小，否则 base model 的 tokenizer 中大量特殊 token（如音频占位符、自定义控制 token）会失效。

正确做法：
- **保持 base model 的原始 `vocab_size`**。
- 比较两个 tokenizer，确认公共 token 范围。
- 对公共 token，复用新 LLM 的 `embed_tokens` 和 `lm_head` 权重。
- 对 base model 独有的 token，随机初始化对应的 embedding 和 lm_head 行。

典型例子（Step-Audio-2-mini + Qwen2.5-7B）：
- Step-Audio-2-mini `vocab_size = 158720`。
- Qwen2.5-7B 实际 tokenizer 长度为 `151665`（`config.vocab_size = 152064` 中多出的行是 padding，不应复制）。
- 前 `151665` 个 token 完全一致。
- 剩余 `158720 - 151665 = 7055` 个 token 需要随机初始化。

正确做法（在 register 脚本中）：

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# 源 LLM 的真实词表大小必须用 tokenizer 长度，不能直接用 config.vocab_size
source_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)

target_vocab_size = model.config.text_config.vocab_size  # 158720
source_vocab_size = len(source_tokenizer)                # 151665
hidden_size = model.config.text_config.hidden_size       # 3584

# 替换 LLM backbone
model.model = llm_model.model

# 重建 embed_tokens: 公共部分复用，新增部分随机初始化
new_embed = nn.Embedding(target_vocab_size, hidden_size, dtype=torch_dtype)
with torch.no_grad():
    new_embed.weight[:source_vocab_size].copy_(
        llm_model.model.embed_tokens.weight[:source_vocab_size])
    torch.nn.init.normal_(
        new_embed.weight[source_vocab_size:],
        mean=0.0, std=hidden_size ** -0.5)
model.model.embed_tokens = new_embed

# 重建 lm_head: 同上
new_lm_head = nn.Linear(hidden_size, target_vocab_size, bias=False, dtype=torch_dtype)
with torch.no_grad():
    new_lm_head.weight[:source_vocab_size].copy_(
        llm_model.lm_head.weight[:source_vocab_size])
    torch.nn.init.normal_(
        new_lm_head.weight[source_vocab_size:],
        mean=0.0, std=hidden_size ** -0.5)
model.lm_head = new_lm_head

# 保持 base model 的 vocab_size，不要覆盖成新 LLM 的
# model.config.text_config.vocab_size 保持 158720
```

常见错误：
- 直接用 `llm_model.config.vocab_size` 作为 `source_vocab_size` → 会复制源 LLM 的 padding rows，覆盖 base model 的特殊 token（如 `<|EOT|>`、`<|BOT|>`、audio token），导致生成乱码。
- 直接把 `lm_head` 替换为新 LLM 的 `lm_head`，并把 `vocab_size` 改成新 LLM 的大小 → tokenizer 中大量特殊 token 无法编码/解码，导致乱码或重复。
- 保留 base model 的 `lm_head` 但替换 LLM backbone → 输出维度不匹配，训练/推理直接失败。

### 2. 训练时冻结参数依赖 `model_arch`

ms-swift 的 `--freeze_llm` / `--freeze_vit` / `--freeze_aligner` 等 CLI 参数依赖于 `model_arch` 来知道哪些模块属于哪个组件。如果注册了自定义模型却没有配置 `model_arch`，CLI 冻结参数不会生效，冻结逻辑会完全错误。

正确做法：

```python
from swift.llm import MultiModelKeys, register_model_arch

register_model_arch(
    MultiModelKeys(
        'step_audio2_mini_custom_arch',
        language_model=['model.embed_tokens', 'model.layers', 'model.norm'],
        vision_tower=['encoder'],
        aligner=['adapter'],
        generator=['lm_head'],
    ))

register_model(
    ModelMeta(
        'step_audio2_mini_custom',
        ...,
        model_arch='step_audio2_mini_custom_arch',
        ...,
    ),
    exist_ok=True,
)
```

配置后可以通过 CLI 精确控制训练参数：

```bash
swift sft \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --trainable_parameters_regex lm_head \
  ...
```

## 失败处理

- **源路径不存在**：检查路径，或改用 `"random"`。
- **生成脚本语法检查失败**：检查适配器/模板代码并修复。
- **缺少某模型的专用适配器**：在 `templates/model_register/` 下新增该模型的适配器，不要回退到 checkpoint assembly。

## 进入下一阶段

注册脚本生成并记录后，进入 `registered/03-dataset-register.md`。
