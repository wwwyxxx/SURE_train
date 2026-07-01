# Kimi-Audio + ms-swift 训练实战指南

## 定位

本 skill 记录将 Kimi-Audio 适配到 ms-swift 框架下进行 ASR 微调的成功经验。

## 模型结构

- **Shared LLM Layer**: Qwen2.5-7B
- **Audio Encoder**: Whisper-large-v3
- **Adaptor**: `model.vq_adaptor`
- **MIMO 分支**: `model.mimo_layers` + `model.mimo_norm`（随机初始化，必须训练）
- **Text Head**: `mimo_output`
- **Audio Head**: `lm_head`（ASR 任务中忽略）

## ASR 任务训练范围

冻结：

- `model.embed_tokens`
- `model.layers`
- `model.norm`
- `whisper_model`
- `lm_head`（Audio Head，ASR 不需要）

训练：

- `model.vq_adaptor`
- `model.mimo_layers`
- `model.mimo_norm`
- `mimo_output`（Text Head）

可选训练：

- `whisper_model`（通过 `KIMI_AUDIO_TRAIN_WHISPER=1`，但通常不推荐）

## Labels/Mask 标准格式

最终采用**标准 CausalLM 格式**：

1. `Template._encode` 返回**未 shift** 的 `labels`（即 `text_input_ids`）
2. `data_collator` 对 batch 做 right padding，非预测位置设为 `-100`：
   ```python
   labels[~text_loss_mask.bool()] = -100
   ```
3. `model.forward` 内部对 `labels` 和 `text_loss_mask` 做 shift，再计算 loss
4. ms-swift 的 `compute_acc` 会自动对未 shift 的 labels 做 shift 并过滤 `-100`

## 关键教训

- **不要在 `_encode` 里 shift labels**，否则 `compute_acc` 会再 shift 一次，导致 `token_acc` 几乎为 0
- `pad_token_id` 可能为 `None`，需要 fallback 到 `eos_token_id` 或 `0`
- `mimo_layers` 是**随机初始化**的，必须解冻训练
- `vq_adaptor` 的 transpose 维度：**`transpose(0, 1)`**，不是 `transpose(1, 2)`

## 冻结策略代码

```python
train_prefixes = [
    'model.vq_adaptor.',
    'model.mimo_layers.',
    'model.mimo_norm.',
    'mimo_output.',
]

if os.environ.get('KIMI_AUDIO_TRAIN_WHISPER', '0') == '1':
    train_prefixes.append('whisper_model.')

if os.environ.get('KIMI_AUDIO_TEXTHEAD_ONLY', '0') == '1':
    train_prefixes = ['mimo_output.']
```

## 八卡 DDP 训练示例

```bash
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --custom_register_path custom/kimi_audio_swift_register.py \
  --model /workspace/model/Qwen2.5-7B \
  --model_type kimi_audio_text \
  --dataset combined_asr_aishell_1 \
  --train_type full \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir output/combined_asr_aishell1_8gpu_freeze_whisper_2ep \
  --report_to none
```

## Batch Size > 1 实现要点

### Data Collator

- 对 `input_ids`, `text_input_ids`, `is_continuous_mask`, `labels`, `text_loss_mask` 做 right padding
- 生成 `attention_mask`
- `whisper_input_feature` 保持为 **list of raw waveforms**

### Model Forward

1. 对 list 中每个 raw waveform 逐个过 `whisper_model`
2. reshape whisper feature（4:1 下采样）
3. padding 到 batch 内最大长度
4. 传给 shared layers

### 关键修改

- `expanded_whisper` 形状改为 `[batch, seq_len, dim]`
- `vq_adaptor` 输入用 `transpose(0, 1)` 得到 `[seq_len, batch, dim]`
- `position_ids` 基于 `attention_mask` 生成，支持 right padding

## 推理验证

```bash
python batch_infer_asr.py \
  --checkpoint output/combined_asr_aishell1_8gpu_freeze_whisper_2ep/vX-YYYYMMDD-HHMMSS/checkpoint-XXX \
  --dataset data/combined_asr_aishell-1.jsonl \
  --num-samples 100 \
  --start 0 \
  --max-new-tokens 128
```

成功标志：

- 训练 `loss` 持续下降
- `token_acc` 明显上升
- 推理结果与目标文本相关

## 常见错误及解决方案

| 错误 | 原因 | 解决 |
|------|------|------|
| `token_acc` 几乎为 0 | labels 被 shift 两次 | `_encode` 返回未 shift labels，forward 内部 shift |
| `TypeError: new_full(): fill_value must be Number, not NoneType` | `pad_token_id` 为 None | fallback 到 `eos_token_id` 或 `0` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | `vq_adaptor` transpose 维度错误 | 用 `transpose(0, 1)` |
| `Sizes of tensors must match except in dimension 1` | labels shift 时 batch 维度硬编码 | 用 `(text_labels.shape[0], 1)` 代替 `(1, 1)` |
| `Bus error. DataLoader worker is killed` | shared memory 不足 | docker 加 `--shm-size=64g` 或减少 workers |
| 输出全是"的" | `mimo_layers` 随机初始化且被冻结 | 解冻 `model.mimo_layers.` 和 `model.mimo_norm.` |
| 模型不学习音频信息 | whisper 被解冻且 lr 过高 | 冻结 whisper，只训练 adaptor/head |

## 性能参考

冻结 whisper + 2 epochs + lr=1e-4 + AISHELL-1 134k：

- train_loss: ~0.9
- token_acc: ~96.7%
- 单卡显存: ~57GB (bs=8, 8卡)
