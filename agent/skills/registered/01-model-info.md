# registered/01 - Model Info

## 目标

对于 ms-swift 已原生支持的模型，不需要读论文和源码，只需要从 ms-swift 源代码和示例中确认关键信息。

## Agent Checklist

- [ ] 读取 `outputs/{run_id}/swift_support_report.json`
- [ ] 确认 `model_type`（如 `qwen2_audio`、`qwen2_5_omni`）
- [ ] 确认 `model_path`：
  - 优先使用 input.json 中用户指定的 `model_path`
  - 如果未指定，根据 `model_family` 推断默认值
  - 检查该路径是否存在本地权重
- [ ] 搜索 `SURE_train/ms-swift/examples/` 中是否有该模型的训练/推理示例
- [ ] 搜索 `SURE_train/ms-swift/docs/` 中是否有该模型的文档
- [ ] 检查该模型是否需要特殊环境变量或参数，例如：
  - `MAX_PIXELS`（视觉模型）
  - `USE_HF`（需要从 HuggingFace 下载）
  - 特殊的 `task_type`（如 `seq_cls`）
- [ ] 记录 inference 方式：
  - 能否直接用 `swift infer`？
  - 是否需要 `PtEngine` Python API？
  - 是否不支持推理（需要后续确认）？
- [ ] 填写 `outputs/{run_id}/registered_model_info.json`
- [ ] 更新 `pipeline_state.json`

## 输出 JSON 格式

```json
{
  "model_family": "qwen2_audio",
  "model_type": "qwen2_audio",
  "model_path": "/workspace/model/Qwen2-Audio-7B",
  "model_path_exists": true,
  "category": "MLLM",
  "inference_engine": "PtEngine",
  "can_use_swift_infer": true,
  "special_args": {},
  "examples_found": {
    "training": false,
    "inference": false
  },
  "notes": "Native ms-swift audio model. Use --model_type qwen2_audio."
}
```

## 组件化初始化（可选）

如果用户想从多个独立来源初始化模型组件，而不是使用单一官方 checkpoint，需要额外收集以下信息：

1. **base_model_path**：用于复制 `config.json`、tokenizer、processor 等辅助文件的官方/base checkpoint 路径。
2. **component_paths**：每个组件的来源路径或 `"random"`。
   - 从 `executors/data/registered_model_arch_components.json` 查询该 `model_type` 的组件列表。
   - 对每个组件询问用户：提供 checkpoint 路径，或 `"random"` 随机初始化。
3. **component_trainable**（可选）：每个组件是否参与训练。
   - 组件名来自 `component_paths` 中的 key，或特殊伪组件 `text_head`。
   - `true` = 训练，`false` = 冻结。
   - `text_head` 由 harness 自动解析为实际的文本输出头参数名（见 `executors/data/registered_model_text_head.json`）。

示例（qwen2_audio）：

```json
{
  "base_model_path": "/workspace/model/Qwen2-Audio-7B",
  "component_paths": {
    "language_model": "/workspace/model/Qwen2.5-7B",
    "aligner": "random",
    "vision_tower": "/workspace/model/whisper-large-v3"
  },
  "component_trainable": {
    "language_model": false,
    "aligner": true,
    "vision_tower": false,
    "text_head": true
  }
}
```

如果提供了 `component_paths`，在 `registered_model_info.json` 中写入 `registered_model_register_path` 占位符（实际脚本在 `registered_model_register` 阶段生成）：

```json
{
  "model_family": "qwen2_audio",
  "model_type": "qwen2_audio",
  "model_path": "/workspace/model/Qwen2-Audio-7B",
  "base_model_path": "/workspace/model/Qwen2-Audio-7B",
  "registered_model_type": null,
  "registered_model_register_path": null,
  "component_paths": {
    "language_model": "/workspace/model/Qwen2.5-7B",
    "aligner": "random",
    "vision_tower": "/workspace/model/whisper-large-v3"
  },
  "component_trainable": {
    "language_model": false,
    "aligner": true,
    "vision_tower": false,
    "text_head": true
  }
}
```

## 需要确认的关键问题

1. **model_path 是否正确？**
   - 本地是否存在？
   - 如果不存在，需要从 ModelScope/HuggingFace 下载什么 model_id？

2. **是否需要下载权重？**
   - 对于已注册模型，通常只有一个 model repo 需要下载
   - 调用 `find_local_model.py` 检查本地是否存在
   - 如果不存在，向用户确认后下载

3. **是否有特殊训练参数？**
   - 查看 ms-swift examples 和 docs
   - 例如 `qwen2_5_omni` 需要设置 `MAX_PIXELS`
   - 某些模型需要用 `--use_chat_template true`

4. **推理方式？**
   - 大多数已注册音频模型可以用 `swift infer --val_dataset xxx.jsonl`
   - 如果 `swift infer` 不能处理音频，考虑用 `PtEngine` Python API

## 和未注册路径的区别

- 不需要分析模型组件
- 不需要写 model_register / template_register 代码
- 主要关注：model_type 对不对、model_path/组件路径 有没有、训练/冻结参数怎么设

## 失败处理

- `model_type` 不确定：重新跑 `check_swift_support.py` 或手动查 `constant.py`
- `model_path` 不存在：询问用户是否下载，或提供正确路径
- 找不到示例：不影响继续，但要在 notes 中记录
- 特殊参数不确定：查 ms-swift docs 或 issue

## 进入下一阶段

确认 `registered_model_info.json` 后，进入 `registered/02-model-register.md`。

如果用户没有提供 `component_paths`（使用单一官方 checkpoint），可以跳过 `registered_model_register`，直接进入 `registered/03-dataset-register.md`。
