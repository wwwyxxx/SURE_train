# ms-swift 模型适配 Agent Harness

## 定位

本目录是为 **Kimi Code agent** 设计的一套 harness，用于把任意 Hugging Face 模型 + 数据集适配到 ms-swift 框架。

**核心原则**：

- 不自动调用 LLM API
- Kimi Code 是执行者
- harness 提供：操作手册、工具脚本、验证脚本、状态追踪

## 目录结构

```text
.swift-adapter-agent/
├── README.md                      # 本文件：给人看的总说明
├── AGENTS.md                      # 给 Kimi Code 看的总入口
├── pipeline.py                    # 状态管理工具（支持 custom / registered 两条路径）
├── skills/                        # 每个阶段的操作手册（检查清单）
│   ├── 00-overview.md
│   ├── 01-hardware-probe.md
│   ├── 02-environment-setup.md
│   ├── 03-swift-support-check.md  # 判断 ms-swift 是否原生支持该模型
│   ├── 04-model-analysis.md       # custom path
│   ├── 05-user-decision.md        # custom path
│   ├── 06-download-weights.md     # custom path
│   ├── 07-dataset-register.md     # custom path
│   ├── 08-model-register.md       # custom path
│   ├── 09-template-register.md    # custom path
│   ├── 10-integration-test.md     # custom path
│   ├── 11-smoke-test.md           # custom path
│   ├── 12-training-script.md      # custom path
│   ├── 13-full-training.md        # custom path
│   ├── 14-evaluation.md           # custom path
│   └── registered/                # registered path
│       ├── 00-overview.md
│       ├── 01-model-info.md
│       ├── 02-checkpoint-assembly.md  # 可选：按组件组装初始化权重
│       ├── 03-dataset-register.md
│       ├── 04-integration-test.md
│       ├── 05-smoke-test.md
│       ├── 06-training-script.md
│       ├── 07-full-training.md
│       └── 08-evaluation.md
├── executors/                     # 确定性工具脚本（Kimi Code 调用）
│   ├── hardware_probe.py
│   ├── check_swift_support.py     # 检查 ms-swift 支持情况
│   ├── build_docker.py
│   ├── run_validator.py
│   ├── run_integration_tests.py
│   ├── run_registered_integration_tests.py  # 已支持模型的集成测试
│   ├── assemble_registered_checkpoint.py    # 按组件组装已支持模型的 checkpoint
│   ├── generate_freeze_args.py              # 根据 component_trainable 生成冻结参数
│   └── start_training.sh
├── validators/                    # 验证脚本
│   ├── core/
│   └── model_specific/kimi_audio/
├── templates/                     # 代码模板
│   ├── Dockerfile.template
│   ├── training_script.template.sh           # custom path
│   └── training_script_registered.template.sh # registered path
├── schemas/
│   └── pipeline_state.schema.json
└── outputs/                       # 每次 run 的输出
    └── {run_id}/
        ├── pipeline_state.json
        ├── swift_support_report.json
        ├── custom/
        │   └── {model}_swift_register.py
        ├── run_{model}_{dataset}.sh
        ├── Dockerfile
        └── logs/
```

## 使用方式

### 1. 初始化

Kimi Code 读取输入后，初始化 pipeline：

```bash
python .swift-adapter-agent/pipeline.py \
  --init \
  --input input.json
```

### 2. 按阶段执行

Kimi Code 按顺序执行：

1. 读 `AGENTS.md`
2. 读对应 `skills/xx-xxx.md`
3. 调用 `executors/` 中的工具
4. 调用 `validators/` 验证
5. 用 `pipeline.py` 更新状态

### 3. 路径分支

`skills/03-swift-support-check.md` 会判断 ms-swift 是否原生支持该模型：

- **不支持**（如 kimi_audio, mimo_audio）：走 custom path（skills/04 ~ skills/14）
- **已支持**（如 qwen2_audio, qwen_omni, step_audio2_mini）：走 registered path（skills/registered/）

切换命令：

```bash
python .swift-adapter-agent/pipeline.py --switch-path registered --run-id {run_id}
```

### 4. 查看状态

```bash
python .swift-adapter-agent/pipeline.py --status --run-id 20260623-143052
```

## Pipeline 阶段

### Custom Path（未注册模型）

| 阶段 | 是否需要写代码 | 验证方式 |
|------|--------------|---------|
| hardware_probe | 否 | `validate_hardware.py` |
| environment_setup | 否（生成 Dockerfile） | `validate_docker_image.py` |
| swift_support_check | 否 | `check_swift_support.py` |
| model_analysis | 否（写分析报告） | 自检查 JSON schema |
| user_decision | 否（询问用户） | 用户确认 |
| download_weights | 否（下载权重） | 文件存在性检查 |
| dataset_register | 是 | `validate_dataset_registration.py` |
| model_register | 是 | `validate_model_registration.py` |
| template_register | 是 | `validate_template_registration.py` |
| integration_test | 否 | 多个 core validator |
| smoke_test | 否 | 模型特定验证 |
| training_script | 是（bash） | `validate_training_script.py` |
| full_training | 否 | 训练日志指标 |
| evaluation | 否（推理 + 算指标） | `evaluation/wenet_compute_cer.py`（CER/WER） |

### Registered Path（已注册模型）

| 阶段 | 是否需要写代码 | 验证方式 |
|------|--------------|---------|
| hardware_probe | 否 | `validate_hardware.py` |
| environment_setup | 否 | `validate_docker_image.py` |
| swift_support_check | 否 | `check_swift_support.py` |
| registered_model_info | 否 | 自检查 JSON |
| registered_model_register | 否/是 | `generate_registered_model_register.py`（可选） |
| registered_dataset_register | 是/否 | `validate_dataset_registration.py` |
| registered_integration_test | 否 | core validators |
| registered_smoke_test | 否 | `swift infer` / PtEngine |
| registered_training_script | 是（bash） | `validate_training_script.py` |
| registered_full_training | 否 | 训练日志指标 |
| registered_evaluation | 否（推理 + 算指标） | `evaluation/wenet_compute_cer.py`（CER/WER） |

## 输入 JSON 示例

### 未注册模型

```json
{
  "model_family": "kimi_audio",
  "dataset_path": "data/combined_asr_aishell-1.jsonl",
  "dataset_name": "combined_asr_aishell_1",
  "source_code_url": "/workspace/Kimi-Audio",
  "paper_url": "/workspace/papers/kimi-audio.pdf"
}
```

### 已注册模型

**使用单一官方 checkpoint：**

```json
{
  "model_family": "qwen2_audio",
  "model_path": "/workspace/model/Qwen2-Audio-7B",
  "dataset_path": "data/combined_asr.jsonl",
  "dataset_name": "combined_asr"
}
```

**按组件初始化（from scratch / 替换组件）：**

```json
{
  "model_family": "qwen2_audio",
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
  },
  "dataset_path": "data/combined_asr.jsonl",
  "dataset_name": "combined_asr"
}
```

跨架构组件（如 Whisper encoder → qwen2_audio 的 audio_tower）由 harness 自动识别 `config.json` 并应用 key 映射，一般无需手写 `component_config.json`。

## 输出

- `outputs/{run_id}/pipeline_state.json`
- `outputs/{run_id}/swift_support_report.json`
- `outputs/{run_id}/custom/{model}_swift_register.py`（custom path）
- `outputs/{run_id}/custom/{model}_dataset_register.py`（registered path，可选）
- `outputs/{run_id}/custom/{model_family}_registered_model_register.py`（registered path 按组件初始化时生成）
- `outputs/{run_id}/freeze_args_report.json`（registered path 按组件训练策略生成）
- `outputs/{run_id}/run_{model}_{dataset}.sh`
- `outputs/{run_id}/Dockerfile`

## 与旧 skill 的关系

旧的 `SURE_train/skills/kimi-audio-adapter-generator/` 只针对 Kimi-Audio。本框架是通用设计，Kimi-Audio 的特有验证脚本保留在 `validators/model_specific/kimi_audio/` 中。
