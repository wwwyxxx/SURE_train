# Kimi Audio + ms-swift 训练实战指南

本 SKILL 记录将 Kimi Audio 适配到 ms-swift 框架下进行 ASR 微调的成功经验，包括八卡 DDP 训练、batch size > 1、labels/mask 处理、常见错误及解决方案。

## 1. 项目目标

- **目标模型**：Kimi Audio（基于 Shared LLM Layer + Adaptor + Whisper Encoder + Text/Audio Head）
- **训练框架**：ms-swift
- **冻结模块**：Shared LLM Layer（Qwen2.5-7B）
- **训练模块**：
  - `model.vq_adaptor`（Adaptor）
  - `model.mimo_layers` + `model.mimo_norm`（MIMO 分支，**必须训练**，因为随机初始化）
  - `mimo_output`（Text Head）
  - 可选 `whisper_model`（Whisper Encoder）
- **不训练模块**：`lm_head`（Audio Head）

## 2. 核心适配文件

### `custom/kimi_audio_swift_register.py`

该文件是接入 ms-swift 的核心：

- `KimiAudioTextSFTModel`：包装原始 `KimiAudioModel`，用于文本-only SFT。
- `KimiAudioTextTemplate`：自定义 Template，把 `messages + audios` 编码成模型输入。
- `KimiAudioASRPreprocessor`：把原始 `{wav, txt, prompt}` 转成 `{messages, audios}`。
- 注册函数：把模型、template、数据集挂到 ms-swift。

### `Kimi-Audio/finetune_codes/modeling_kimia.py`

原始模型实现，主要修改：

- 移除 `torch.cuda.current_device()` 硬编码，改为跟随输入 tensor device。
- 支持 batch size > 1 的 whisper feature 拼接。
- 支持 right padding 的 `attention_mask` 和 `position_ids`。
- 修正 `vq_adaptor` 的 transpose 维度（`transpose(0, 1)`，不是 `transpose(1, 2)`）。

## 3. Labels/Mask 标准格式

经过多次调试，最终采用**标准 CausalLM 格式**：

1. `Template._encode` 返回**未 shift** 的 `labels`（即 `text_input_ids`）。
2. `data_collator` 对 batch 做 right padding，并把非预测位置设为 `-100`：
   ```python
   labels[~text_loss_mask.bool()] = -100
   ```
3. `model.forward` 内部对 `labels` 和 `text_loss_mask` 做 shift，再计算 loss。
4. ms-swift 的 `compute_acc` 会自动对未 shift 的 labels 做 shift 并过滤 `-100`。

**关键教训**：

- 不要预先在 `_encode` 里 shift labels，否则 `compute_acc` 会再 shift 一次，导致 `token_acc` 几乎为 0。
- `pad_token_id` 可能为 `None`，需要 fallback 到 `eos_token_id` 或 `0`。
- `mimo_layers` 是**随机初始化**的，必须解冻训练，否则 text head 学到的是噪声。

## 4. 训练范围配置

通过环境变量控制训练模块（在 `custom/kimi_audio_swift_register.py` 的 `_freeze_for_text_overfit` 中）：

```python
# 默认：训练 adaptor + mimo_layers + mimo_norm + text head
train_prefixes = [
    'model.vq_adaptor.',
    'model.mimo_layers.',
    'model.mimo_norm.',
    'mimo_output.',
]

# 加上 whisper
if os.environ.get('KIMI_AUDIO_TRAIN_WHISPER', '0') == '1':
    train_prefixes.append('whisper_model.')

# 只训练 text head（调试用）
if os.environ.get('KIMI_AUDIO_TEXTHEAD_ONLY', '0') == '1':
    train_prefixes = ['mimo_output.']
```

## 5. 八卡 DDP 训练

### Docker 启动

```bash
docker run -it --rm --gpus all \
  --shm-size=64g \
  -v /aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train:/workspace \
  -w /workspace \
  docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0 \
  /bin/bash
```

**注意**：

- 必须用真实路径（`realpath` 查看），不能用 `/mnt/lustre/...` 这种不存在或符号链路错误的路径。
- `--shm-size=64g` 防止 DataLoader worker 因 shared memory 不足被杀。
- 如果数据音频在项目目录外，需要额外 `-v` 挂载。

### 训练脚本

```bash
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
  --custom_register_path custom/kimi_audio_swift_register.py \
  --model /workspace/model/Qwen2.5-7B \
  --model_type kimi_audio_text \
  --dataset reprodata_asr_zh_existing_aishell93 \
  --train_type full \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 100 \
  --learning_rate 1e-4 \
  --lr_scheduler_type constant \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 512 \
  --logging_steps 1 \
  --save_steps 1000 \
  --save_total_limit 1 \
  --save_only_model true \
  --output_dir output/reprodata_asr_zh_existing_aishell93_8gpu_smoke_bs4 \
  --report_to none
```

`NPROC_PER_NODE=8` 会触发 ms-swift 用 `torch.distributed.run` 启动 8 个进程。

## 6. Batch Size > 1 的实现要点

### Data Collator

- 对 `input_ids`、`text_input_ids`、`is_continuous_mask`、`labels`、`text_loss_mask` 做 right padding。
- 生成 `attention_mask`。
- `whisper_input_feature` 保持为 **list of raw waveforms**（长度不同，不能简单 padding）。

### Model Forward

在 `KimiAudioTextSFTModel.forward` 中：

1. 对 list 中每个 raw waveform 逐个过 `whisper_model`。
2. reshape whisper feature（4:1 下采样）。
3. padding 到 batch 内最大长度。
4. 传给 shared layers。

### MoonshotKimiaModel

- `expanded_whisper` 形状改为 `[batch, seq_len, dim]`。
- `vq_adaptor` 输入用 `transpose(0, 1)` 得到 `[seq_len, batch, dim]`，确保作用在 feature 维度上。
- `position_ids` 基于 `attention_mask` 生成，支持 right padding。

## 7. 数据准备

### 93 条验证集

从 `data/reprodata_asr_zh_existing.jsonl` 中筛选出能在 AISHELL-1 目录中找到对应音频的 93 条，音频复制到 `data/aishell_audio/`。

生成命令：

```bash
python3 - <<'PY'
import json
import os

jsonl_path = 'data/reprodata_asr_zh_existing.jsonl'
audio_dir = 'data/aishell_audio'
output_path = 'data/reprodata_asr_zh_existing_aishell93.jsonl'

available = set(os.listdir(audio_dir))

with open(jsonl_path, 'r', encoding='utf-8') as f:
    rows = [json.loads(line) for line in f]

new_rows = []
for row in rows:
    basename = os.path.basename(row['wav'])
    if basename not in available:
        continue
    new_path = os.path.join(audio_dir, basename)
    new_row = dict(row)
    new_row['wav'] = new_path
    new_rows.append(new_row)

with open(output_path, 'w', encoding='utf-8') as f:
    for row in new_rows:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print(f'Wrote {len(new_rows)} rows to {output_path}')
PY
```

### 完整数据集

原始 776 条中 683 条音频路径缺失，需要：

- 挂载 `/hpc_stor03` 到容器内：`-v /hpc_stor03:/hpc_stor03`
- 或者用 AISHELL-1 完整数据重新构建训练集

## 8. 推理验证

```bash
python batch_infer_asr.py \
  --checkpoint output/reprodata_asr_zh_existing_aishell93_8gpu_smoke_bs4/vX-YYYYMMDD-HHMMSS/checkpoint-XXX \
  --dataset data/reprodata_asr_zh_existing_aishell93.jsonl \
  --num-samples 93
```

成功标志：

- 训练 `loss` 降到 `1e-6` 量级
- `token_acc` 接近或达到 `1.0`
- 推理结果大部分 `OK`

## 9. 常见错误及解决方案

| 错误 | 原因 | 解决 |
|------|------|------|
| `Permission denied: /hpc_stor03/...` | 容器内没挂载数据目录 | 启动 docker 时加 `-v /hpc_stor03:/hpc_stor03` |
| `token_acc` 几乎为 0 | labels 被 shift 了两次 | 让 `_encode` 返回未 shift labels，forward 内部 shift |
| `TypeError: new_full(): fill_value must be Number, not NoneType` | `self.config.pad_token_id` 为 None | fallback 到 `eos_token_id` 或 `0` |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied (20480x152 and 5120x3584)` | `vq_adaptor` transpose 维度错误 | 用 `transpose(0, 1)`，不是 `transpose(1, 2)` |
| `Sizes of tensors must match except in dimension 1. Expected size 4 but got size 1` | labels shift 时 batch 维度硬编码 | 用 `(text_labels.shape[0], 1)` 代替 `(1, 1)` |
| `Bus error. DataLoader worker is killed` | shared memory 不足 | 启动 docker 加 `--shm-size=64g` 或 `--dataloader_num_workers 0` |
| 输出全是"的" | `mimo_layers` 随机初始化且被冻结 | 解冻 `model.mimo_layers.` 和 `model.mimo_norm.` |

## 10. 关键脚本清单

| 脚本 | 用途 |
|------|------|
| `run_reprodata_asr_8gpu_smoke_bs4.sh` | 八卡 bs=4 smoke test |
| `run_reprodata_asr_8gpu_smoke.sh` | 八卡 bs=1 smoke test |
| `run_reprodata_asr_8gpu.sh` | 八卡正式训练（可训练 whisper） |
| `batch_infer_asr.py` | 批量推理验证 |
| `debug_encode_sample.py` | 检查 labels/mask 编码 |

## 11. 成功配置总结

最终能成功 overfit 93 条数据的配置：

- 八卡 DDP
- per_device_train_batch_size=4
- learning_rate=1e-4
- num_train_epochs=100
- 训练模块：`model.vq_adaptor`、`model.mimo_layers`、`model.mimo_norm`、`mimo_output`
- 冻结模块：shared LLM、whisper_model、lm_head
- labels 标准格式：未 shift + `-100` mask
- 总训练时间：约 3 分钟
- 最终 loss：`~1e-6`
- 最终 token_acc：`1.0`
