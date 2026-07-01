# ms-swift 模型适配 Skill

## 核心思想

不要试图用确定性 generator 覆盖所有模型。而是：

> **把实现思路、关键模式、验证流程写入 skill，让 agent 根据 skill 去阅读具体模型的论文和源码，再编写适配代码。**

这个 skill 是 agent 的操作手册。agent 的任务是：

1. 理解目标模型（读论文、源码、config）
2. 套用本 skill 的适配模式
3. 编写 `register_model` / `register_template` / `register_dataset`
4. 生成训练脚本
5. 运行验证脚本并修复问题

## ms-swift 三大注册机制

适配任何模型到 ms-swift，本质上是完成三件事：

### 1. register_model

告诉 ms-swift 如何加载你的模型。通常需要：

- 一个 `get_model_tokenizer_xxx` 函数
- 可选的模型包装类（如果原始模型接口和 ms-swift 不兼容）
- 权重初始化/复制逻辑
- 冻结策略

```python
register_model(
    'your_model_type',
    '/path/to/base/model',
    get_model_tokenizer_function=get_model_tokenizer_your_model,
    template='your_template',
)
```

### 2. register_template

告诉 ms-swift 如何把 `{messages, audios, images}` 转换成模型需要的输入格式。

关键方法：

- `_encode(self, inputs) -> Dict[str, Tensor]`：单条数据编码
- `data_collator(self, batch) -> Dict[str, Tensor]`：batch 填充

```python
register_template(
    TemplateMeta('your_template', prefix=[], prompt=['{{query}}'], chat_sep=[], suffix=[]),
    use_slots=True,
    lazy_tokenize=True,
)
```

### 3. register_dataset

告诉 ms-swift 如何加载和预处理数据集。

```python
register_dataset(
    DatasetMeta(
        dataset_path='/path/to/data.jsonl',
        dataset_name='your_dataset',
        preprocess_func=YourPreprocessor(),
    ),
    exist_ok=True,
)
```

数据流：

```text
raw jsonl row  --preprocessor-->  {messages, audios}  --template.encode-->  model inputs
```

## 通用适配流程

### Step 1: 分析模型

Agent 需要阅读模型源码，回答：

- 基础 LLM 是什么？（Qwen/Llama/...）
- 是否有 audio/vision encoder？
- 模型 forward 的签名是什么？
- 是否需要 special tokens？
- 权重是否需要从 base model 复制？

### Step 2: 设计 Template

根据模型 forward 的输入，设计 template 输出：

- `input_ids`
- `attention_mask`
- 多模态特征（如 `whisper_input_feature`, `pixel_values`）
- `labels` 和 `loss_mask`

关键原则：

- **labels 和 logits 对齐**：如果 forward 内部会 shift，template 里就不要 shift
- **padding 位置必须设为 -100 或在 loss_mask 中排除**
- **data_collator 要能处理变长序列**

### Step 3: 设计 Model Wrapper

如果原始模型不能直接接受 ms-swift 的输入格式，需要写一个 wrapper：

```python
class YourSFTModel(OriginalModel):
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # 转换输入格式
        # 调用原始 forward
        # 计算/返回 loss
```

### Step 4: 注册数据集

数据集 jsonl 格式建议：

```json
{"wav": "path/to/audio.wav", "txt": "transcription text", "prompt": "instruction"}
```

preprocessor 转成：

```python
{
    "messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": txt}
    ],
    "audios": [wav]
}
```

### Step 5: 生成训练脚本

训练脚本至少包含：

```bash
swift sft \
  --custom_register_path custom/auto_register.py \
  --model /path/to/base/model \
  --model_type your_model_type \
  --dataset your_dataset_name \
  --train_type full \
  --freeze_llm true/false \
  --freeze_vit true/false \
  --freeze_aligner true/false \
  --per_device_train_batch_size ... \
  --num_train_epochs ... \
  --learning_rate ...
```

### Step 6: 运行验证

按顺序跑验证脚本，每步通过后再进行下一步：

1. `validate_model_registration.py`
2. `validate_dataset_registration.py`
3. `validate_template_registration.py`
4. `validate_forward_pass.py`
5. `validate_loss_computation.py`
6. `validate_single_step_training.py`
7. `validate_inference.py`
8. `validate_distributed_launch.py`

## Kimi-Audio 案例参考

### 模型结构

- Base LLM: Qwen2.5-7B
- Audio encoder: Whisper-large-v3
- Connector: `vq_adaptor`
- Audio decoder: `mimo_layers` + `mimo_norm`
- Text head: `mimo_output`（从 Qwen lm_head 部分复制）

### 关键决策

| 问题 | 决策 | 原因 |
|------|------|------|
| 是否训练 Whisper？ | 冻结 | 训练会破坏预训练特征 |
| 哪些模块训练？ | vq_adaptor, mimo_layers, mimo_norm, mimo_output | 只有这些需要适配 |
| labels 是否 shift？ | 不 shift，在 forward 里 shift | 避免 shift 两次 |
| 怎么处理 padding？ | collator 里设 -100，forward 里用 text_loss_mask | 确保 loss 只算 response |

### 关键代码模式

**Template encode 返回：**

```python
{
    'input_ids': audio_input_ids[0],
    'text_input_ids': text_input_ids[0],
    'is_continuous_mask': is_continuous_mask[0],
    'whisper_input_feature': wav,
    'labels': text_input_ids[0],
    'text_loss_mask': text_loss_mask[0],
}
```

**Forward 里处理 list of waveforms：**

```python
whisper_feat_list = []
for wav_tensor in wav_tensors:
    feats = self.whisper_model(wav_tensor)
    feats = feats.reshape(...)
    whisper_feat_list.append(feats.squeeze(0))
# pad and stack
```

**Loss 计算：**

```python
text_labels = torch.cat((text_labels[:, 1:], pad), dim=1)
text_loss_mask = torch.cat((text_loss_mask[:, 1:], false_mask), dim=1)
loss = cross_entropy(logits_flat, labels_flat) * mask_flat
loss = loss.sum() / mask.sum()
```

## 常见陷阱

### 1. Labels 被 shift 两次

如果 template 里已经 shift 了 labels，model forward 又 shift 一次，会导致位置错位。

**检查方法**：跑 `validate_label_shift.py`。

### 2. Padding 位置参与 loss

data_collator 必须把所有 padding 位置设为 -100 或在 loss_mask 中排除。

**检查方法**：`validate_loss_computation.py`。

### 3. freeze_vit 不生效

确认 `vision_tower` 在 `register_model_arch` 中正确注册。

**检查方法**：`validate_freeze_unfreeze.py`。

### 4. Multi-GPU 设备 hardcode

不要在代码里写 `torch.cuda.current_device()` 或固定 device id。

### 5. batch_size > 1 时 whisper 输入处理

单条音频是 1D tensor，batch 时要能处理 list of tensors 并 padding。

### 6. dataloader_num_workers 导致 shared memory 不足

docker 启动时加 `--shm-size=64g`。

## 给 Agent 的 Prompt 模板

当要让 agent 适配一个新模型时，用这个 prompt：

```text
请根据 skills/kimi-audio-adapter-generator/SKILL.md 的指引，把以下模型适配到 ms-swift 框架。

模型信息：
- 模型路径：{model_path}
- 模型论文/源码链接：{paper_or_repo}
- 基础 LLM：{base_llm}
- 模态：{audio/image/text}
- 数据集路径：{dataset_jsonl}
- 数据集名称：{dataset_name}

请完成：
1. 阅读模型源码，分析 forward 输入输出
2. 编写 custom/{model_name}_swift_register.py，包含 register_model/register_template/register_dataset
3. 编写 run_{model_name}_{dataset_name}.sh 训练脚本
4. 按 SKILL.md 的顺序运行 validators/ 下的验证脚本
5. 如果验证失败，修复代码并重新验证

要求：
- 代码放在 SURE_train/custom/ 目录
- 训练脚本放在 SURE_train/ 目录
- 每次修改后运行 python3 -m py_compile 检查语法
```

## 目录结构

```text
skills/kimi-audio-adapter-generator/
├── SKILL.md                          # 本手册
├── generate_adapter.py               # 针对 Kimi-Audio 的旧版确定性生成器（参考用）
└── validators/                       # 验证脚本
    ├── core/                         # 通用验证
    │   ├── validate_model_registration.py
    │   ├── validate_template_registration.py
    │   ├── validate_dataset_registration.py
    │   ├── validate_forward_pass.py
    │   ├── validate_single_step_training.py
    │   ├── validate_checkpoint_save_load.py
    │   ├── validate_inference.py
    │   ├── validate_freeze_unfreeze.py
    │   ├── validate_distributed_launch.py
    │   └── validate_batch_size_scaling.py
    └── model_specific/kimi_audio/    # Kimi-Audio 特有验证
        ├── validate_weight_initialization.py
        ├── validate_loss_computation.py
        ├── validate_label_shift.py
        └── validate_sequence_length.py
```

## 使用流程

```bash
# 1. Agent 阅读 skill 和模型源码
# 2. Agent 编写适配代码
# 3. Agent 运行验证
python validators/core/validate_model_registration.py ...
python validators/core/validate_template_registration.py ...
# ... 依次运行
# 4. Agent 启动训练
bash run_xxx.sh
```
