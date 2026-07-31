# 06 - Download Weights

## 目标

根据 `download_plan.json`，确保所有需要的初始化权重文件都已下载到本地。

## 核心原则

1. **优先使用 `SURE_train/model/` 下已有的本地权重，不要重复下载。**
2. **只下载语音理解 / ASR 需要的权重。音频输出部件（audio decoder、Patch Decoder 等）不需要下载。**

## Agent Checklist

- [ ] 读取 `outputs/{run_id}/download_plan.json`
- [ ] 读取 `outputs/{run_id}/user_decision.json` 中的 `user_provided_links`
- [ ] **优先检查 `SURE_train/model/` 目录**：
  ```bash
  python .swift-adapter-agent/executors/find_local_model.py \
    --model-id qwen/Qwen2.5-7B \
    --sure-train-dir SURE_train \
    --output outputs/{run_id}/find_local_qwen.json
  ```
- [ ] 对 download_plan 中每个 `model_id`，先用 `find_local_model.py` 查找本地是否已有
- [ ] 本地已有的权重，更新 `download_plan.json` 中的 `local_dir`，不再下载
- [ ] 本地不存在的权重，调用 `batch_download.py`：
  ```bash
  python .swift-adapter-agent/executors/batch_download.py \
    --plan outputs/{run_id}/download_plan.json \
    --sure-train-dir SURE_train \
    --output outputs/{run_id}/download_report.json
  ```
- [ ] 读取 `download_report.json`
- [ ] 如果全部下载成功：
  - 更新 `pipeline_state.json`
  - 进入 dataset_register
- [ ] 如果有失败：
  - 生成 `missing_weights.json`
  - 向用户展示缺失列表，请求提供下载链接
  - 用户回复后，更新 `download_plan.json`
  - 重新调用 batch_download.py
  - 重试最多 3 次

## 查找逻辑

```text
for each model_id in download_plan:
    1. 检查 download_plan 中指定的 local_dir 是否已存在
    2. 检查 SURE_train/model/<model-name> 是否已存在
    3. 如果本地有，更新 local_dir 并跳过下载
    4. 如果本地没有，从 modelscope/huggingface 下载
```

## 下载前确认

在真正开始下载前，agent 应该向用户展示：

```text
以下权重已在 SURE_train/model/ 中找到，将直接使用：
- Qwen2.5-7B -> /aistor/.../SURE_train/model/Qwen2.5-7B

以下权重需要下载：
- xxx/xxx -> /workspace/model/xxx

是否开始下载？
```

用户确认后再开始下载。

## 支持的来源

| 来源 | 处理方式 |
|------|---------|
| `SURE_train/model/` 已存在 | 直接使用，不下载 |
| 指定的 `local_dir` 已存在 | 直接使用，不下载 |
| `modelscope` ID | 调用 batch_download.py |
| `huggingface` ID | 调用 batch_download.py --use-hf |
| `random` | 无需下载 |
| `copy_from_xxx` | 在 model_register 阶段处理 |
| 用户提供的 URL | 用 wget/git/git-lfs 下载 |

## 输出 JSON 格式

`download_report.json`：

```json
{
  "passed": true,
  "downloads": [
    {
      "component": "shared_llm",
      "model_id": "qwen/Qwen2.5-7B",
      "local_dir": "/aistor/.../SURE_train/model/Qwen2.5-7B",
      "passed": true,
      "report": "Found existing weights at ..."
    },
    {
      "component": "whisper_encoder",
      "model_id": "openai/whisper-large-v3",
      "local_dir": "/aistor/.../SURE_train/model/whisper-large-v3",
      "passed": true,
      "report": "Found existing weights at ..."
    }
  ],
  "missing": []
}
```

## 缺失权重处理

如果某些权重下载失败，agent 应该：

1. 生成 `outputs/{run_id}/missing_weights.json`
2. 向用户展示：
   ```text
   以下权重无法自动下载：
   - whisper_encoder: openai/whisper-large-v3

   请提供下载链接或本地路径。例如：
   - ModelScope: damo/speech_whisper-large-v3_demo
   - HuggingFace: openai/whisper-large-v3
   - 本地路径: /workspace/model/whisper-large-v3
   ```
3. 等待用户回复
4. 更新 `download_plan.json`
5. 重新下载

## 失败处理

- 网络问题：重试 3 次
- model_id 错误：询问用户正确 ID
- 磁盘空间不足：清理或更换路径
- 下载后文件损坏：删除重下

## 进入下一阶段

所有权重下载成功后，进入 `07-dataset-register` 阶段。
