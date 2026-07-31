# MiMo-Audio 音频 Token 缓存指南

## 定位

本 skill 专门指导 agent 如何为 MiMo-Audio 预处理音频 token。

MiMo-Audio 的 audio tokenizer 是独立于 LLM 的 CUDA 模型，必须在训练前把音频编码成离散 RVQ tokens 并缓存。这是适配 MiMo-Audio 时**最容易出错、最耗时**的环节。

## 核心原则

1. **训练前必须先缓存**，不能 lazy tokenize
2. **进程隔离**：每个 chunk 用独立 docker 容器处理
3. **不要同时启动多个预处理任务**，避免 GPU 冲突
4. **后台运行时必须 disable_timeout**，否则会被系统 kill
5. **缓存完成后必须验证完整性**再启动训练

## 何时触发本 skill

在 `07-dataset-register` 和 `09-template-register` 完成后、`13-full-training` 之前，如果 dataset 还没缓存，必须先执行本 skill。

## 缓存脚本

使用 `tools/precompute_mimo_audio_tokens_chunks.sh`：

```bash
bash tools/precompute_mimo_audio_tokens_chunks.sh \
  data/combined_asr_aishell-1.jsonl \
  data/combined_asr_aishell-1_cached.jsonl \
  data/audio_tokens_cache/combined_asr_aishell-1 \
  1000 \
  8
```

参数说明：

| 位置 | 含义 | 示例 |
|------|------|------|
| 1 | 原始 jsonl 数据集 | `data/combined_asr_aishell-1.jsonl` |
| 2 | 缓存后输出 jsonl | `data/combined_asr_aishell-1_cached.jsonl` |
| 3 | `.pt` 缓存目录 | `data/audio_tokens_cache/combined_asr_aishell-1` |
| 4 | 每个 chunk 的样本数 | `1000` |
| 5 | GPU 数量（可选，默认 8） | `8` 或 `7` |

## 7 GPU vs 8 GPU

- 默认用 8 GPU，每轮处理 8 个 chunk
- 如果需要留一卡做其他事情，用 7 GPU
- 修改方式：第 5 个参数传 `7`

## 启动方式

推荐用 `nohup` + 后台任务，并设置 disable_timeout：

```bash
nohup bash tools/precompute_mimo_audio_tokens_chunks.sh \
  data/combined_asr_aishell-1.jsonl \
  data/combined_asr_aishell-1_cached.jsonl \
  data/audio_tokens_cache/combined_asr_aishell-1 \
  1000 \
  7 > output/precompute_audio_tokens_chunks.log 2>&1 &
```

如果用 Kimi Code 的 Bash 后台任务，必须：

```bash
bash tools/precompute_mimo_audio_tokens_chunks.sh ...
```

并设置 `disable_timeout=true`。

## 绝对不要做的事

### 1. 同时启动多个预处理任务

以下行为会导致 GPU 冲突和重复编码：

- 同时跑旧版 `precompute_mimo_audio_tokens.py --gpus 0 1 2 3 4 5 6`
- 同时跑 batched 版 `precompute_mimo_audio_tokens_batched.py`
- 同时跑多个 chunk-based 脚本实例

**后果**：同一个 GPU 上多个 CUDA context 切换，速度变慢；一个进程 CUDA assert 会污染同 GPU 上其他进程。

### 2. 在线编码音频 token

不要在 dataset preprocessor 里每条音频都调用 `MiMoAudioTokenizer`：

```python
# 错误：训练时在线编码
audio_tokens = _encode_audio_to_list(wav)
```

正确做法：dataset preprocessor 检测 `.pt` 路径直接加载：

```python
if resolved.endswith('.pt') and os.path.exists(resolved):
    audio_tokens = torch.load(resolved, map_location='cpu').tolist()
else:
    audio_tokens = _encode_audio_to_list(wav)
```

### 3. 把缓存任务作为有超时限制的后台任务

Kimi Code 的 Bash 后台任务默认有超时。缓存 134k 音频需要数小时，必须设置 `disable_timeout=true`。

## 监控进度

```bash
# 看完成了多少个 chunk
find data/audio_tokens_cache/combined_asr_aishell-1/ -maxdepth 1 -name 'chunk_out_*.jsonl' | wc -l

# 看生成了多少个 .pt 文件
find data/audio_tokens_cache/combined_asr_aishell-1/ -maxdepth 1 -name '*.pt' | wc -l

# 看最近完成的 chunk
ls -lt data/audio_tokens_cache/combined_asr_aishell-1/chunk_out_*.jsonl | head

# 看日志
tail -f output/precompute_audio_tokens_chunks.log
```

## 验证缓存完整性

缓存跑完后，合并前必须检查：

```bash
DATASET=data/combined_asr_aishell-1.jsonl
CACHE_DIR=data/audio_tokens_cache/combined_asr_aishell-1

TOTAL=$(wc -l < "$DATASET")
CHUNKS=$(find "$CACHE_DIR" -maxdepth 1 -name 'chunk_out_*.jsonl' | wc -l)
CACHED=$(find "$CACHE_DIR" -maxdepth 1 -name '*.pt' | wc -l)

echo "Total samples: $TOTAL"
echo "Completed chunks: $CHUNKS"
echo "Cached .pt files: $CACHED"

# 检查每个 chunk_out 是否有 1000 行（最后一个可能不足）
for f in "$CACHE_DIR"/chunk_out_*.jsonl; do
    lines=$(wc -l < "$f")
    echo "$f: $lines lines"
done
```

通过标准：

- `CHUNKS` 等于预期 chunk 数（ceil(TOTAL / CHUNK_SIZE)）
- `CACHED` 接近 `TOTAL`（允许少量失败）
- 大部分 `chunk_out_*.jsonl` 有 `CHUNK_SIZE` 行

## 合并缓存数据集

脚本会自动合并。如果手动合并：

```bash
OUTPUT=data/combined_asr_aishell-1_cached.jsonl
> "$OUTPUT"
for f in data/audio_tokens_cache/combined_asr_aishell-1/chunk_out_*.jsonl; do
    cat "$f" >> "$OUTPUT"
done
wc -l "$OUTPUT"
```

## 失败重跑

如果某个 chunk 失败：

1. 删除对应的 `chunk_out_X.jsonl`
2. 删除该 chunk 对应的部分 `.pt` 文件（可选，脚本会 skip-existing）
3. 重新跑整个脚本，已完成的 chunk 会自动 skip

## 注册缓存后数据集

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

注意：cached 数据集的名字要和原始数据集区分开，训练时用 `combined_asr_aishell_1_cached`。

## Agent Checklist

- [ ] 确认原始数据集存在
- [ ] 确认 `tools/precompute_mimo_audio_tokens_chunks.sh` 存在
- [ ] 检查是否有其他预处理任务在跑
- [ ] 选择 GPU 数量（默认 8，可选 7）
- [ ] 启动缓存脚本并设置 disable_timeout
- [ ] 监控 chunk 完成进度
- [ ] 验证缓存完整性
- [ ] 合并生成 `*_cached.jsonl`
- [ ] 注册 cached 数据集
- [ ] 训练时使用 cached 数据集名字
