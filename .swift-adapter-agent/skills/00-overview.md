# 00 - Pipeline 总览

## Agent 必读

本 skill 是 ms-swift 模型适配 pipeline 的总览。你在执行任何阶段前，必须先阅读本文件。

## Pipeline 输入

输入是一个 JSON 文件，最小版本只需要 3 个字段：

```json
{
  "model_family": "kimi_audio",
  "dataset_path": "data/combined_asr_aishell-1.jsonl",
  "dataset_name": "combined_asr_aishell_1"
}
```

## model_family 说明

`model_family` 是必填字段，用于告诉 agent 你要适配什么模型族。支持的值：

| model_family | 含义 | 自动推断的 model_path | ms-swift 是否原生支持 |
|-------------|------|----------------------|---------------------|
| `kimi_audio` | Kimi-Audio 音频模型 | `/workspace/model/Qwen2.5-7B` | ❌ 不支持（需自定义注册） |
| `mimo_audio` | MiMo-Audio 音频模型 | `/workspace/model/Qwen2.5-7B-Instruct` | ❌ 不支持（需自定义注册） |
| `qwen2_audio` | Qwen2-Audio | `/workspace/model/Qwen2-Audio-7B` | ✅ 支持 |
| `qwen_omni` | Qwen-Omni 系列 | `/workspace/model/Qwen2.5-Omni-3B` | ✅ 支持 |
| `qwen2_5_omni` | Qwen2.5-Omni | `/workspace/model/Qwen2.5-Omni-3B` | ✅ 支持 |
| `qwen3_omni` | Qwen3-Omni | `/workspace/model/Qwen3-Omni-3B` | ✅ 支持 |
| `step_audio` | Step-Audio | `/workspace/model/Step-Audio-Chat` | ✅ 支持 |
| `step_audio2_mini` | Step-Audio-2-mini | `/workspace/model/Step-Audio-2-mini` | ✅ 支持 |
| `qwen2` | 纯文本 Qwen2 | `/workspace/model/Qwen2.5-7B` | ✅ 支持 |
| `llama3` | 纯文本 Llama3 | `/workspace/model/Meta-Llama-3-8B` | ✅ 支持 |
| `other` | 其他模型 | 无，必须手动填 model_path | 动态判断 |

如果用户知道具体路径，可以直接在 input.json 里写 `model_path`。如果不知道，agent 根据 `model_family` 自动推断。

## 完整输入字段

```json
{
  "model_family": "qwen2_audio",
  "model_path": "/workspace/model/Qwen2-Audio-7B",
  "dataset_path": "data/combined_asr.jsonl",
  "dataset_name": "combined_asr",
  "source_code_url": "",
  "paper_url": "",
  "output_run_id": ""
}
```

对于已注册模型，`source_code_url` 和 `paper_url` 通常不需要。

## 本地 paper 处理

`paper_url` 和 `source_code_url` 都可以是本地路径：

```json
{
  "paper_url": "/workspace/papers/kimi-audio.pdf",
  "source_code_url": "/workspace/Kimi-Audio"
}
```

Agent 读取本地文件时：

- PDF：使用 pdfplumber 或 PyPDF2 提取文本
- Markdown/TXT：直接读取
- 源码目录：递归读取关键 `.py` 文件

注意：**已注册模型不需要读论文和源码**。

## Pipeline 路径

本 harness 有两条独立路径：

### 路径 A：未注册模型（Custom Path）

适用于 kimi_audio、mimo_audio 等 ms-swift 不原生支持的模型。

| 阶段 ID | 是否需要写代码 | Skill 文件 |
|---------|--------------|-----------|
| `hardware_probe` | 否 | `skills/01-hardware-probe.md` |
| `environment_setup` | 否（生成 Dockerfile） | `skills/02-environment-setup.md` |
| `swift_support_check` | 否 | `skills/03-swift-support-check.md` |
| `model_analysis` | 否（写分析报告） | `skills/04-model-analysis.md` |
| `user_decision` | 否（询问用户） | `skills/05-user-decision.md` |
| `download_weights` | 否（下载权重） | `skills/06-download-weights.md` |
| `dataset_register` | 是 | `skills/07-dataset-register.md` |
| `model_register` | 是 | `skills/08-model-register.md` |
| `template_register` | 是 | `skills/09-template-register.md` |
| `integration_test` | 否（跑验证） | `skills/10-integration-test.md` |
| `smoke_test` | 否（跑验证） | `skills/11-smoke-test.md` |
| `training_script` | 是（bash） | `skills/12-training-script.md` |
| `full_training` | 否（启动训练） | `skills/13-full-training.md` |

### 路径 B：已注册模型（Registered Path）

适用于 qwen2_audio、qwen_omni、Step-Audio-2-mini 等 ms-swift 原生支持的模型。

| 阶段 ID | 是否需要写代码 | Skill 文件 |
|---------|--------------|-----------|
| `hardware_probe` | 否 | `skills/01-hardware-probe.md` |
| `environment_setup` | 否 | `skills/02-environment-setup.md` |
| `swift_support_check` | 否 | `skills/03-swift-support-check.md` |
| `registered_model_info` | 否 | `skills/registered/01-model-info.md` |
| `registered_checkpoint_assembly` | 否/是 | `skills/registered/02-checkpoint-assembly.md`（可选） |
| `registered_dataset_register` | 是/否 | `skills/registered/03-dataset-register.md` |
| `registered_integration_test` | 否 | `skills/registered/04-integration-test.md` |
| `registered_smoke_test` | 否 | `skills/registered/05-smoke-test.md` |
| `registered_training_script` | 是（bash） | `skills/registered/06-training-script.md` |
| `registered_full_training` | 否 | `skills/registered/07-full-training.md` |

## 路径切换

`swift_support_check` 阶段会输出 `swift_support_report.json`：

- `supported: false` → 继续路径 A
- `supported: true` → 切换到路径 B

切换命令：

```bash
python .swift-adapter-agent/pipeline.py --switch-path registered --run-id {run_id}
```

## 与 ms-swift-internals 的关系

本目录（`skills/`）的 skill 告诉你**每个阶段要做什么、怎么调用工具**。

`skills/ms-swift-internals/` 告诉你**ms-swift 框架的内部原理**，在适配代码前必须先读。

完整阅读顺序：

1. `ms-swift-internals/00-overview.md`
2. `ms-swift-internals/01-register_model.md`
3. `ms-swift-internals/02-register_template.md`
4. `ms-swift-internals/03-register_dataset.md`
5. `ms-swift-internals/04-sft-training-flow.md`
6. `ms-swift-internals/05-extension-hooks.md`
7. `ms-swift-internals/06-data-flow.md`
8. `skills/00-overview.md`（本文件）
9. `skills/01-hardware-probe.md` ~ `skills/03-swift-support-check.md`
10. 根据 `swift_support_check` 结果选择路径：
    - 路径 A：`skills/04-model-analysis.md` ~ `skills/13-full-training.md`
    - 路径 B：`skills/registered/01-model-info.md` ~ `skills/registered/07-full-training.md`（02 为可选组件组装）
11. 遇到问题查 `ms-swift-internals/07-common-issues.md`

## Agent Checklist（通用）

每个阶段你都要做：

- [ ] 读取对应 skill
- [ ] 调用 `pipeline.py --start-stage {stage_id}`
- [ ] 按 skill 执行具体操作
- [ ] 调用对应 validator 或 executor
- [ ] 根据结果更新状态：
  - 成功：`pipeline.py --complete-stage {stage_id} --validation-report xxx.json`
  - 失败：`pipeline.py --fail-stage {stage_id} --error-file xxx.json`
- [ ] 如果失败，按 skill 修复，重试最多 3 次
- [ ] 成功后进入下一阶段

## 状态记录格式

每个阶段在 `pipeline_state.json` 中必须包含：

```json
{
  "id": "dataset_register",
  "name": "Dataset Register",
  "status": "completed",
  "started_at": "ISO8601",
  "completed_at": "ISO8601",
  "output": {},
  "validation": {
    "passed": true,
    "validator": "validators/core/validate_dataset_registration.py",
    "report": "..."
  },
  "code_path": "outputs/{run_id}/custom/xxx.py",
  "code_section": "register_dataset",
  "log_path": "outputs/{run_id}/logs/04-dataset-register.log",
  "error": null,
  "retry_count": 0
}
```

## 失败处理

1. 读错误日志，定位失败原因
2. 判断失败属于哪个阶段
3. 最小化修改代码
4. 重新运行该阶段验证
5. 重试 3 次仍失败则停止，向用户报告

## 禁止行为

- 跳过 validator
- 一次写多个阶段代码
- 修改已 completed 阶段（除非后续验证证明其有问题）
- 把代码写到 `outputs/{run_id}/custom/` 之外
- 在路径 A 和路径 B 之间混用 skill
