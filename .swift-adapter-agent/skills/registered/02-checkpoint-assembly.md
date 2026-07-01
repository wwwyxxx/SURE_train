# registered/02 - Checkpoint Assembly

## 目标

本阶段用于**按组件组装初始化权重**。如果用户想从多个独立来源初始化模型组件（例如用 Qwen2.5-7B 初始化 `language_model`、用 Whisper 初始化 `vision_tower/audio_tower`、随机初始化 `aligner`），而不是使用单一官方 checkpoint，则执行本阶段。

如果用户已经提供了一个完整的官方 checkpoint 路径，可以**跳过**本阶段，直接进入 `registered/03-dataset-register.md`。

## 什么时候需要 checkpoint assembly？

- 用户只有各个组件的独立权重，没有完整的官方模型 checkpoint。
- 用户想替换某个组件的权重来源（例如换不同的 audio encoder）。
- 用户明确想随机初始化某些组件。

## Agent Checklist

- [ ] 读取 `outputs/{run_id}/registered_model_info.json`
- [ ] 确认 `base_model_path`（用于复制 config、tokenizer、processor 等辅助文件）
- [ ] 确认 `component_paths`：每个组件的来源路径或 `"random"`
- [ ] 从 `executors/data/registered_model_arch_components.json` 查询该 `model_type` 的组件列表
- [ ] 如果某个组件的目标前缀是列表，准备 `component_config.json`
- [ ] 跨架构组件（如 Whisper -> audio_tower）优先使用 harness 内置的自动 key 映射；自动映射不满足时再准备 `component_config.json`
- [ ] 调用 `assemble_registered_checkpoint.py` 组装 checkpoint
- [ ] 检查 `checkpoint_assembly_report.json`
- [ ] 更新 `registered_model_info.json` 中的 `assembled_model_path`
- [ ] 更新 `pipeline_state.json`

## 输入

从 `registered_model_info.json` 读取：

```json
{
  "model_family": "qwen2_audio",
  "model_type": "qwen2_audio",
  "base_model_path": "/workspace/model/Qwen2-Audio-7B",
  "component_paths": {
    "language_model": "/workspace/model/Qwen2.5-7B",
    "aligner": "random",
    "vision_tower": "/workspace/model/whisper-large-v3"
  }
}
```

## 查询组件映射

```bash
python3 - <<'PY'
import json
with open('.swift-adapter-agent/executors/data/registered_model_arch_components.json') as f:
    mapping = json.load(f)
print(json.dumps(mapping.get('qwen2_audio'), indent=2))
PY
```

`qwen2_audio` 示例：

```json
{
  "language_model": ["language_model"],
  "aligner": ["multi_modal_projector"],
  "vision_tower": ["audio_tower"]
}
```

## 默认 key 映射规则

对于每个组件，源 checkpoint 的所有 key 默认会被**加上目标前缀**。

例如 `language_model` 的目标前缀是 `language_model`：

- 源 `model.layers.0.self_attn.q_proj.weight`
- 目标 `language_model.model.layers.0.self_attn.q_proj.weight` ✅

## 自动跨架构映射

`assemble_registered_checkpoint.py` 会读取源 checkpoint 的 `config.json`，自动识别源模型类型。对于已知的跨架构组合（如 Whisper encoder → `qwen2_audio` 的 `audio_tower`），会应用内置映射：

- Whisper `model.encoder.*` → `audio_tower.*`

这由 `executors/data/registered_component_key_mappings.json` 维护，**不需要用户手写 `component_config.json`**。

如果自动映射失败，再回退到默认前缀追加，或让用户提供 `component_config.json`。

## 调用 assembly executor

```bash
python .swift-adapter-agent/executors/assemble_registered_checkpoint.py \
  --model-family qwen2_audio \
  --model-type qwen2_audio \
  --base-model-path /workspace/model/Qwen2-Audio-7B \
  --component-paths-json outputs/{run_id}/component_paths.json \
  --output-dir outputs/{run_id}/assembled_model \
  --output-report outputs/{run_id}/checkpoint_assembly_report.json
```

其中 `component_paths.json`：

```json
{
  "language_model": "/workspace/model/Qwen2.5-7B",
  "aligner": "random",
  "vision_tower": "/workspace/model/whisper-large-v3"
}
```

## 自定义 key 映射（可选）

如果组件目标前缀是列表，或源架构与目标架构不一致，提供 `component_config.json`：

```bash
python .swift-adapter-agent/executors/assemble_registered_checkpoint.py \
  ... \
  --component-config-json outputs/{run_id}/component_config.json
```

示例：把 Whisper encoder 直接映射到 `audio_tower`，并去掉源前缀 `model.encoder`：

```json
{
  "vision_tower": {
    "target_submodule": "audio_tower",
    "key_mapping": {
      "model.encoder.conv1.weight": "conv1.weight",
      "model.encoder.conv2.weight": "conv2.weight",
      "model.encoder.layers.0.self_attn.k_proj.weight": "layers.0.self_attn.k_proj.weight"
    }
  }
}
```

更简单的用法：只想指定目标子模块，仍用默认前缀追加：

```json
{
  "vision_tower": {
    "target_submodule": "audio_tower"
  }
}
```

> 注意：如果组件在架构映射中的目标前缀是列表（如 `qwen2_5_omni` 的 `language_model`），必须提供 `target_submodule` 或完整 `key_mapping`。

## 输出

- `outputs/{run_id}/assembled_model/model.safetensors`（如果 safetensors 可用）或 `pytorch_model.bin`
- `outputs/{run_id}/assembled_model/config.json`（从 `base_model_path` 复制）
- tokenizer / processor 文件（从 `base_model_path` 复制）
- `outputs/{run_id}/checkpoint_assembly_report.json`

报告示例：

```json
{
  "model_family": "qwen2_audio",
  "model_type": "qwen2_audio",
  "assembled_model_path": "/abs/path/outputs/{run_id}/assembled_model",
  "components": {
    "language_model": {
      "source_path": "/workspace/model/Qwen2.5-7B",
      "source_model_type": "qwen2",
      "source_keys": 296,
      "remapped_keys": 296,
      "target_prefixes": ["language_model"],
      "key_mapping_used": false,
      "auto_mapping_used": false
    },
    "vision_tower": {
      "source_path": "/workspace/model/whisper-large-v3",
      "source_model_type": "whisper",
      "source_keys": 264,
      "remapped_keys": 96,
      "target_prefixes": ["audio_tower"],
      "key_mapping_used": true,
      "auto_mapping_used": true
    }
  },
  "random_components": ["aligner"],
  "warnings": [],
  "base_model_files_copied": ["config.json", "tokenizer.json", "preprocessor_config.json"]
}
```

## 更新 model info

组装完成后，更新 `outputs/{run_id}/registered_model_info.json`：

```json
{
  "model_family": "qwen2_audio",
  "model_type": "qwen2_audio",
  "model_path": "/workspace/model/Qwen2-Audio-7B",
  "assembled_model_path": "outputs/{run_id}/assembled_model",
  "base_model_path": "/workspace/model/Qwen2-Audio-7B",
  "component_paths": {
    "language_model": "/workspace/model/Qwen2.5-7B",
    "aligner": "random",
    "vision_tower": "/workspace/model/whisper-large-v3"
  },
  ...
}
```

后续阶段（integration test / smoke test / training script）使用 `assembled_model_path` 作为 `--model`。

## 失败处理

- **源路径不存在**：检查路径，或改用 `"random"`
- **key 映射后形状不匹配**：在 `component_config.json` 中提供显式 `key_mapping`
- **base_model_path 不存在**：必须提供包含 config.json 和 tokenizer 文件的目录；否则组装出的 checkpoint 无法被 ms-swift 加载
- **组件目标前缀为列表**：提供 `target_submodule` 或 `key_mapping`

## 进入下一阶段

组装成功并通过报告检查后，进入 `registered/03-dataset-register.md`。
