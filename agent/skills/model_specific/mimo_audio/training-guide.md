# MiMo-Audio + ms-swift 训练实战指南

## 定位

本 skill 记录将 MiMo-Audio 适配到 ms-swift 框架下进行 ASR 微调的成功经验。

MiMo-Audio 与 Kimi-Audio 的最大区别：**audio tokenizer 在 LLM 外部**，必须先编码成离散 RVQ tokens 再喂给 LLM。这导致 lazy tokenize 不可行，必须做音频 token 缓存。

## 模型结构

- **Shared LLM Layer**: Qwen2.5-7B
- **Audio Tokenizer**: `MiMoAudioTokenizer`（独立模型，**不是 LLM 的一部分**）
- **Audio Input Embedder / Aligner**:
  - `speech_embeddings`
  - `input_local_transformer`
  - `speech_group_downcast`
- **Text Head**: `lm_head`

Audio tokenizer 输出 RVQ 离散 tokens：`[num_quantizers=8, T]`，按 `group_size=4` 分组后展开为 `[T * audio_channels]` 的 1D token 序列。

## ASR 任务训练范围

冻结：

- `model.embed_tokens`
- `model.layers`
- `model.norm`
- `MiMoAudioTokenizer`（独立 audio tokenizer，永远不参与训练）

训练：

- `speech_embeddings`
- `input_local_transformer`
- `speech_group_downcast`
- `lm_head`

## Labels/Mask 标准格式

最终采用**标准 CausalLM 格式**，和 Kimi-Audio 一致：

1. `Template._encode` 返回**未 shift** 的 `labels`（即 `text_input_ids`）
2. `labels[text_loss_mask == False] = -100`
3. `model.forward` 内部对 logits/labels 做 causal shift：
   ```python
   shift_logits = text_logits[..., :-1, :]
   shift_labels = labels[..., 1:]
   shift_mask = text_loss_mask[..., 1:]
   ```
4. ms-swift 的 `compute_acc` 会自动对未 shift 的 labels 再 shift 一次并过滤 `-100`

## 关键教训

- **不要在 `_encode` 里 shift labels**，否则 `token_acc` 会几乎为 0
- **MiMo-Audio 不能吃 raw waveform**，必须在 dataset preprocessor / template 里先把音频编码成 RVQ tokens
- **audio tokenizer 是独立 CUDA 模型**，训练时在线编码会导致：
  - dataset preprocess 阶段极慢
  - 某条音频触发 CUDA device-side assert 后，整个 CUDA context 永久损坏
- **必须预先缓存 audio tokens** 到 `.pt` 文件，训练时直接加载整数 token
- **缓存需要进程隔离**：chunk 级独立 docker 容器，避免一个坏样本污染所有 GPU
- **不要同时启动多个预处理任务**（旧版、batched、chunk-based 混跑），会共享 GPU、互相干扰
- **后台缓存任务必须 disable_timeout**，否则系统超时会 kill 整个进程组
- 小数据 overfit 需要 **20 epochs 以上**才能学会，2 epochs 不够

> 详细缓存流程见 [`preprocessing-guide.md`](./preprocessing-guide.md)

## 冻结策略代码

```python
def _freeze_for_asr(model):
    for _, p in model.named_parameters():
        p.requires_grad = False

    train_prefixes = (
        'speech_embeddings.',
        'input_local_transformer.',
        'speech_group_downcast.',
        'lm_head.',
    )
    for name, p in model.named_parameters():
        if name.startswith(train_prefixes):
            p.requires_grad = True
```

## 训练示例

```bash
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --custom_register_path custom/mimo_audio_swift_register.py \
  --model /workspace/model/MiMo-Audio-7B-Base-merged \
  --model_type mimo_audio \
  --dataset combined_asr_aishell_1_cached \
  --train_type full \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --lr_scheduler_type constant \
  --warmup_ratio 0 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir output/combined_asr_aishell1_8gpu \
  --report_to none
```

注意：

- `lr_scheduler_type=constant`，`warmup_ratio=0`（小数据/overfit 场景）
- 大数据训练再考虑 warmup + cosine
- 数据集用 `*_cached`（已预缓存 audio tokens）

## Batch Size > 1 实现要点

### Template._encode

1. `input_ids` 形状为 `[audio_channels+1, T]`：
   - channel 0: text tokens
   - channel 1~8: RVQ audio tokens
2. `text_ids = input_ids[0, ::group_size]` 提取 text 位置
3. 返回：
   - `input_ids`: `[audio_channels+1, T]`
   - `attention_mask`: `[T_groups]`
   - `position_ids`: `[T_groups]`
   - `labels`: `[T_groups]`，未 shift
   - `text_loss_mask`: `[T_groups]`

### Data Collator

- `input_ids` 是 3D `[B, audio_channels+1, T]`
- **audio channels 不能用 text pad_token_id 填充**，否则会超出 audio embedding 词表触发 CUDA assert
- 用 `speech_zeroemb_idx` 填充 audio channels：
  ```python
  for c in range(1, C):
      padded_input_ids[:, c, :] = self.speech_zeroemb_idx[c - 1]
  ```
- `labels`、`attention_mask`、`position_ids`、`text_loss_mask` 做 right padding

### Model Forward

- 接收 3D `input_ids`，调用 `_prepare_input_embeds` 把 audio tokens 转成 embeddings
- `hidden_states` 形状 `[B, T_groups, H]`
- `lm_head` 输出 `[B, T_groups, vocab_size]`
- 内部 causal shift 计算 loss

## Audio Token 缓存（必读）

### 为什么不能 lazy tokenize

MiMo-Audio 的 LLM 输入是**离散 RVQ tokens**，不是连续特征。这些 token 必须在 collator 之前得到，所以无法像 Kimi-Audio 那样在 model forward 里 lazily 提取 Whisper 特征。

### 缓存脚本设计

1. 预先运行 audio tokenizer，把每条音频编码成 `.pt` 文件
2. 生成对应的 `*_cached.jsonl`，其中 `wav` 字段指向 `.pt` 缓存文件
3. dataset preprocessor 检测到 `.pt` 路径就直接 `torch.load`，不再编码

### 关键：进程隔离

MiMoAudioTokenizer 编码时可能触发 CUDA device-side assert，一旦触发，**当前进程的 CUDA context 永久损坏**，即使重载模型也无效。

推荐方案：

- 每 1000 条一个 chunk
- 每个 chunk 跑一个独立的单卡 docker 容器
- 8 个 GPU 每轮并行处理 8 个 chunk
- 一个 chunk 失败只会损失该 chunk，其他 chunk 不受影响

示例脚本：`tools/precompute_mimo_audio_tokens_chunks.sh`

```bash
CHUNK_SIZE=1000
N_GPUS=8
split -l "${CHUNK_SIZE}" -d "${DATASET}" "${CHUNK_DIR}/chunk_"
# 每轮 8 个独立 docker 容器并行处理
```

### 缓存后数据集注册

```python
register_dataset(
    DatasetMeta(
        dataset_path=os.path.join(_ROOT, 'data/combined_asr_aishell-1_cached.jsonl'),
        dataset_name='combined_asr_aishell_1_cached',
        preprocess_func=MiMoAudioASRPreprocessor(),
    ),
    exist_ok=True,
)
```

## 推理验证

```bash
python batch_infer_asr.py \
  --checkpoint output/combined_asr_aishell1_8gpu/vX-YYYYMMDD-HHMMSS/checkpoint-XXX \
  --dataset data/aishell1-test_ASR_infer.jsonl \
  --num-samples 100 \
  --start 0 \
  --max-new-tokens 128
```

成功标志：

- 训练 `loss` 持续下降
- `token_acc` 明显上升（小数据 20 epochs 可达 ~90%）
- 推理结果与目标文本相关

## 常见错误及解决方案

| 错误 | 原因 | 解决 |
|------|------|------|
| `token_acc` 几乎为 0 | labels 被 shift 两次 | `_encode` 返回未 shift labels，forward 内部 shift |
| CUDA device-side assert / Embedding index out of range | audio channels 用 text pad_token_id 填充 | data_collator 用 `speech_zeroemb_idx` 填充 audio channels |
| Dataset preprocess 极慢 / 卡死 | 在线调用 MiMoAudioTokenizer | 预缓存 audio tokens 到 `.pt` |
| 训练几分钟后 CUDA assert，之后所有操作都失败 | audio tokenizer 出错污染了 CUDA context | 用 chunk 级独立 docker 进程做预处理 |
| 1000 条训 2 epochs decode 全错 | epoch 不足 | 小数据 overfit 至少 20 epochs，大数据按 loss 收敛调整 |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied` | MiMo-Audio 输入形状与 template 不匹配 | 确认 `input_ids` 是 `[B, audio_channels+1, T]` |
| 训练不收敛 / loss 不降 | 学习率太低或 scheduler 不对 | 尝试 constant lr=1e-4，无 warmup |
| 预处理速度极慢、GPU util 低 | 多个预处理任务共享 GPU | 清理重叠进程，只保留 chunk-based |
| 缓存任务莫名停止 | 后台任务超时 | 用 nohup 或设置 disable_timeout=true |

## 性能参考

MiMo-Audio-7B-Base + AISHELL-1 93 条 overfit：

- `num_train_epochs=20`
- `learning_rate=1e-4`
- `lr_scheduler_type=constant`
- `per_device_train_batch_size=8`
- 单卡显存：~40GB (A800 80GB)
- 结果：decode 83/93 = 89.25% 正确

MiMo-Audio-7B-Base + AISHELL-1 134k 完整训练（缓存后）：

- batch size 8，gradient accumulation 4
- frozen LLM，训练 aligner + lm_head
- 单卡显存：~40GB
