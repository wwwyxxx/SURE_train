# ms-swift 模型适配 Agent Harness 使用手册

**版本**：2026-06-29  
**适用目录**：`SURE_train/.swift-adapter-agent/`  
**目标读者**：使用 Kimi Code 将自定义语音/多模态模型适配到 ms-swift 框架的开发者。

---

## 1. 概述

本 harness 是 Kimi Code agent 的“操作手册 + 工具脚本 + 状态追踪”集合，用于把任意 Hugging Face 模型和自定义数据集适配到 [ms-swift](https://github.com/modelscope/ms-swift) 训练框架。

### 1.1 核心原则

- **不自动调用 LLM API**：所有决策和代码由 Kimi Code 在本地执行。
- **阶段化执行**：每个阶段必须完成并验证后才能进入下一阶段。
- **状态可追踪**：每次运行的状态保存在 `outputs/{run_id}/pipeline_state.json`。
- **两条独立路径**：
  - **Custom Path**：ms-swift 不原生支持的模型（如 `kimi_audio`、`mimo_audio`），需要自定义 `register_model` + `register_template`。
  - **Registered Path**：ms-swift 已原生支持的模型（如 `qwen2_audio`、`qwen_omni`、`step_audio2_mini`），只需准备数据集和训练脚本。

### 1.2 执行总览

```text
用户输入 → input.json → pipeline.py --init
        → skills/01 hardware_probe
        → skills/02 environment_setup
        → skills/03 swift_support_check
            ├─ supported=false → Custom Path (skills/04 ~ 13)
            └─ supported=true  → Registered Path (skills/registered/01 ~ 07；02 为可选组件组装)
        → 每个阶段：读 skill → 执行 → 验证 → 更新状态
        → 生成训练脚本并启动训练
```

---

## 2. 目录结构

```text
.swift-adapter-agent/
├── AGENTS.md                          # 给 Kimi Code 的总入口
├── README.md                          # 给人看的快速说明
├── pipeline.py                        # 状态管理 CLI（支持路径切换）
├── input.schema.json                  # 输入 JSON Schema
├── manual.md                          # 本手册源文件
├── swift-adapter-agent-manual.pdf     # 本手册 PDF
├── skills/                            # 阶段操作手册
│   ├── 00-overview.md
│   ├── 01-hardware-probe.md
│   ├── 02-environment-setup.md
│   ├── 03-swift-support-check.md      # 判断 ms-swift 是否原生支持
│   ├── 04-model-analysis.md           # Custom Path
│   ├── 05-user-decision.md
│   ├── 06-download-weights.md
│   ├── 07-dataset-register.md
│   ├── 08-model-register.md
│   ├── 09-template-register.md
│   ├── 10-integration-test.md
│   ├── 11-smoke-test.md
│   ├── 12-training-script.md
│   ├── 13-full-training.md
│   └── registered/                    # Registered Path
│       ├── 00-overview.md
│       ├── 01-model-info.md
│       ├── 02-checkpoint-assembly.md    # 可选：按组件组装初始化权重
│       ├── 03-dataset-register.md
│       ├── 04-integration-test.md
│       ├── 05-smoke-test.md
│       ├── 06-training-script.md
│       └── 07-full-training.md
├── executors/                         # 确定性工具脚本
│   ├── check_swift_support.py
│   ├── hardware_probe.py
│   ├── find_local_model.py
│   ├── batch_download.py
│   ├── run_validator.py
│   ├── run_integration_tests.py
│   ├── run_registered_integration_tests.py
│   ├── assemble_registered_checkpoint.py  # 按组件组装已支持模型的 checkpoint
│   ├── resolve_docker_image.py
│   └── build_docker.py
├── validators/                        # 验证脚本
│   ├── core/
│   └── model_specific/
├── templates/                         # 代码模板
│   ├── training_script.template.sh
│   └── training_script_registered.template.sh
├── schemas/
│   └── pipeline_state.schema.json
└── outputs/                           # 每次运行的产物
    └── {run_id}/
        ├── pipeline_state.json
        ├── swift_support_report.json
        ├── custom/
        ├── run_*.sh
        └── Dockerfile
```

---

## 3. 快速开始

### 3.1 准备输入

在与 `.swift-adapter-agent/` 同级的目录（即 `SURE_train/`）下创建 `input.json`：

```json
{
  "model_family": "qwen2_audio",
  "dataset_path": "data/combined_asr.jsonl",
  "dataset_name": "combined_asr"
}
```

### 3.2 初始化 pipeline

```bash
cd SURE_train
python .swift-adapter-agent/pipeline.py --init --input input.json
```

初始化成功后会输出 `run_id`，后续所有命令都基于该 ID。

### 3.3 查看状态

```bash
python .swift-adapter-agent/pipeline.py --status --run-id {run_id}
```

### 3.4 按阶段执行

Agent 会按顺序读取 `skills/` 中对应的 Markdown 文件，调用 `executors/` 中的脚本，并通过 `pipeline.py` 更新状态。用户只需要回答 agent 在 `user_decision` 等阶段的提问即可。

---

## 4. 输入规范（input.json）

### 4.1 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_family` | string | 是 | 模型族，见下表 |
| `model_path` | string | 否 | 单一模型权重路径；为空时根据 `model_family` 推断 |
| `base_model_path` | string/null | 否 | 官方/base checkpoint 路径，用于复制 config/tokenizer/processor |
| `component_paths` | object/null | 否 | 按组件初始化。键为组件名，值为路径或 `"random"` |
| `component_trainable` | object/null | 否 | 按组件训练/冻结策略。`text_head` 为伪组件 |
| `whisper_path` | string/null | 否 | 仅 custom audio 模型需要 |
| `dataset_path` | string | 是 | 数据集 jsonl 路径 |
| `dataset_name` | string | 是 | 注册后的数据集名称 |
| `source_code_url` | string/null | 否 | 模型源码本地路径或 URL（custom 路径建议填） |
| `paper_url` | string/null | 否 | 论文本地路径或 URL（custom 路径建议填） |
| `max_gpus` | integer | 否 | 训练最多使用 GPU 数，默认 7 |
| `output_run_id` | string/null | 否 | 自定义 run_id；为空则自动生成时间戳 |

### 4.2 model_family 与支持状态

| model_family | 默认 model_path | ms-swift 原生支持 |
|--------------|-----------------|-------------------|
| `kimi_audio` | `/workspace/model/Qwen2.5-7B` | ❌ Custom Path |
| `mimo_audio` | `/workspace/model/Qwen2.5-7B-Instruct` | ❌ Custom Path |
| `qwen2_audio` | `/workspace/model/Qwen2-Audio-7B` | ✅ Registered Path |
| `qwen_omni` | `/workspace/model/Qwen2.5-Omni-3B` | ✅ Registered Path |
| `qwen2_5_omni` | `/workspace/model/Qwen2.5-Omni-3B` | ✅ Registered Path |
| `qwen3_omni` | `/workspace/model/Qwen3-Omni-3B` | ✅ Registered Path |
| `step_audio` | `/workspace/model/Step-Audio-Chat` | ✅ Registered Path |
| `step_audio2_mini` | `/workspace/model/Step-Audio-2-mini` | ✅ Registered Path |
| `qwen2` | `/workspace/model/Qwen2.5-7B` | ✅ Registered Path |
| `llama3` | `/workspace/model/Meta-Llama-3-8B` | ✅ Registered Path |
| `other` | 无 | 动态判断 |

### 4.3 示例

**未注册模型（kimi_audio）**：

```json
{
  "model_family": "kimi_audio",
  "dataset_path": "data/combined_asr_aishell-1.jsonl",
  "dataset_name": "combined_asr_aishell_1",
  "source_code_url": "/workspace/Kimi-Audio",
  "paper_url": "/workspace/papers/kimi-audio.pdf"
}
```

**已注册模型（qwen2_audio）**：

```json
{
  "model_family": "qwen2_audio",
  "model_path": "/workspace/model/Qwen2-Audio-7B",
  "dataset_path": "data/combined_asr.jsonl",
  "dataset_name": "combined_asr"
}
```

---

## 5. 通用执行流程

每个阶段的执行模式相同：

```text
1. 读取 skill 文件（例如 skills/07-dataset-register.md）
2. pipeline.py --start-stage {stage_id}
3. 按 skill 执行具体操作（写代码 / 下载 / 配置）
4. 调用 validator 或 executor 验证
5. 成功：pipeline.py --complete-stage {stage_id} --validation-report xxx.json
   失败：pipeline.py --fail-stage {stage_id} --error-file xxx.json
6. 失败时按 skill 修复，最多重试 3 次
```

### 5.1 常用 pipeline.py 命令

| 命令 | 作用 |
|------|------|
| `--init --input input.json` | 初始化新 run |
| `--status --run-id {run_id}` | 查看当前状态 |
| `--start-stage {stage_id}` | 标记阶段开始 |
| `--complete-stage {stage_id} --validation-report {file}` | 标记阶段完成 |
| `--fail-stage {stage_id} --error-file {file}` | 标记阶段失败 |
| `--switch-path registered --run-id {run_id}` | 从 custom 切换到 registered path |

---

## 6. Swift 支持检查与路径选择

### 6.1 运行检查

```bash
python .swift-adapter-agent/executors/check_swift_support.py \
  --model-family {model_family} \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/swift_support_report.json
```

### 6.2 输出示例

```json
{
  "supported": true,
  "model_type": "qwen2_audio",
  "support_level": "native",
  "category": "MLLM",
  "inference_engine": "PtEngine",
  "has_training_examples": true,
  "has_inference_examples": false,
  "notes": "qwen2_audio is a native multimodal model type in ms-swift."
}
```

### 6.3 路径选择

- `supported: false` → 继续 Custom Path（`skills/04` ~ `13`）。
- `supported: true` → 切换到 Registered Path：

```bash
python .swift-adapter-agent/pipeline.py --switch-path registered --run-id {run_id}
```

---

## 7. Custom Path 详解

适用于 `kimi_audio`、`mimo_audio` 等 ms-swift 不原生支持的模型。

### 7.1 model_analysis（阶段 04）

- **目标**：深入理解模型结构，列出所有组件。
- **输入**：`config.json`、论文、源码（`modeling_*.py`）。
- **输出**：`outputs/{run_id}/model_analysis.json`。
- **必须回答的问题**：基础 LLM 是什么？`forward` 签名？音频编码器在 LLM 内部还是外部？输入是连续特征还是离散 token？是否需要扩展 special tokens？
- **ASR 范围注意**：只关注语音理解 / 文本输出，忽略 audio decoder / audio head 等音频合成部件。

### 7.2 user_decision（阶段 05）

- **目标**：向用户展示推荐训练策略并等待确认。
- **输出**：`outputs/{run_id}/user_decision.json`。
- **默认推荐（语音大模型 ASR）**：
  - `shared_llm`: freeze
  - `audio_encoder`: freeze
  - `adaptor`: train
  - `text_head`: train

### 7.3 download_weights（阶段 06）

- **目标**：根据决策下载所有初始化权重。
- **工具**：`find_local_model.py` 优先查找 `SURE_train/model/`，缺失则调用 `batch_download.py`。
- **输出**：`outputs/{run_id}/download_report.json`。

### 7.4 dataset_register（阶段 07）

- **目标**：编写 `register_dataset` 代码，让 ms-swift 能加载并预处理数据集。
- **代码位置**：`outputs/{run_id}/custom/{model}_swift_register.py`。
- **验证**：

```bash
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_dataset_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/validation_dataset.json
```

### 7.5 model_register（阶段 08）

- **目标**：编写 `register_model` 代码，告诉 ms-swift 如何构造模型。
- **验证**：

```bash
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_model_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} \
  --model-type {model_type} \
  --output outputs/{run_id}/validation_model.json
```

### 7.6 template_register（阶段 09）

- **目标**：编写 `register_template` 代码，把对话模板应用到该模型。
- **验证**：

```bash
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_template_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model-type {model_type} \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/validation_template.json
```

### 7.7 integration_test（阶段 10）

- **目标**：跑完整集成测试，验证 forward / loss / single-step / checkpoint / inference / freeze-unfreeze。
- **命令**：

```bash
python .swift-adapter-agent/executors/run_integration_tests.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} \
  --model-type {model_type} \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/integration_test_report.json
```

### 7.8 smoke_test（阶段 11）

- **目标**：验证模型真的能学会、能推理、batch size 不会 OOM。
- **三轮测试**：
  1. 单条 overfit（epoch=2, token_acc > 90%）
  2. 100 条 mini 训练（epoch=20）
  3. Batch size 上限测试
- **输出**：`outputs/{run_id}/smoke_test_report.json`。

### 7.9 training_script（阶段 12）

- **目标**：生成 `run_{model}_{dataset}.sh`。
- **模板**：`templates/training_script.template.sh`。
- **必须包含**：`--custom_register_path`、`--model`、`--model_type`、`--dataset`。
- **GPU 限制**：`NPROC_PER_NODE` 和 `CUDA_VISIBLE_DEVICES` 最多 7 个。
- **语法检查**：`bash -n outputs/{run_id}/run_*.sh`。

### 7.10 full_training（阶段 13）

- **目标**：启动正式训练。
- **命令**：直接执行生成的 bash 脚本，或调用 `start_training.sh`。
- **监控**：观察 loss、token_acc、保存 checkpoint。

---

## 8. Registered Path 详解

适用于 `qwen2_audio`、`qwen_omni`、`step_audio2_mini` 等 ms-swift 已原生支持的模型。

### 8.1 registered_model_info（阶段 registered/01）

- **目标**：确认 `model_type`、`model_path`、特殊参数（如 `MAX_PIXELS`）。
- **不需要读论文/源码**。
- **输出**：`outputs/{run_id}/registered_model_info.json`。
- **组件化初始化（可选）**：如果用户想按组件初始化而不是使用单一官方 checkpoint，收集 `base_model_path` 和 `component_paths`。
- **组件训练策略（可选）**：收集 `component_trainable`，`text_head` 为伪组件，harness 自动解析为实际参数名。

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

### 8.2 registered_checkpoint_assembly（阶段 registered/02，可选）

- **目标**：把各个组件的权重拼成一个 ms-swift 能直接加载的 checkpoint 目录。
- **触发条件**：`input.json` 中提供了 `component_paths`。
- **工具**：`executors/assemble_registered_checkpoint.py`。
- **输出**：`outputs/{run_id}/assembled_model/` 和 `outputs/{run_id}/checkpoint_assembly_report.json`。
- **默认 key 映射**：把源 checkpoint 的所有 key 加上目标组件前缀（如 `language_model.`、`audio_tower.`）。
- **自动跨架构映射**：读取源 checkpoint 的 `config.json`，对已知组合（如 Whisper encoder → `qwen2_audio` 的 `audio_tower`）自动应用 `model.encoder.*` → `audio_tower.*`。映射表在 `executors/data/registered_component_key_mappings.json`。
- **自定义映射**：如果自动映射不满足，提供 `component_config.json` 显式映射。

```bash
python .swift-adapter-agent/executors/assemble_registered_checkpoint.py \
  --model-family qwen2_audio \
  --model-type qwen2_audio \
  --base-model-path /workspace/model/Qwen2-Audio-7B \
  --component-paths-json outputs/{run_id}/component_paths.json \
  --output-dir outputs/{run_id}/assembled_model \
  --output-report outputs/{run_id}/checkpoint_assembly_report.json
```

组装完成后，更新 `registered_model_info.json` 中的 `assembled_model_path`，后续阶段用它作为 `--model`。

### 8.3 registered_dataset_register（阶段 registered/03）

数据集有三种接入方式：

**方案 A：数据已是标准格式（推荐）**

```json
{"messages": [...], "audios": ["xxx.wav"]}
```

训练时直接传路径：

```bash
swift sft --dataset data/combined_asr.jsonl ...
```

**方案 B：使用 dataset_info.json**

```json
[{
  "dataset_name": "combined_asr_custom",
  "dataset_path": "data/combined_asr.jsonl",
  "columns": {"wav": "audios", "txt": "response", "prompt": "query"}
}]
```

训练时：

```bash
swift sft --dataset combined_asr_custom --custom_dataset_info dataset_info.json ...
```

**方案 C：Python register_dataset**

当预处理复杂时使用。代码写在 `outputs/{run_id}/custom/{model}_dataset_register.py`，训练时：

```bash
swift sft \
  --dataset combined_asr_custom \
  --custom_register_path outputs/{run_id}/custom/{model}_dataset_register.py ...
```

注意：这里的 `--custom_register_path` **只包含 dataset register**，不包含 model/template register。

### 8.4 registered_integration_test（阶段 registered/04）

```bash
python .swift-adapter-agent/executors/run_registered_integration_tests.py \
  --model-type {model_type} \
  --model {model_path} \
  --dataset-name {dataset_name} \
  --custom-register-path outputs/{run_id}/custom/{model}_dataset_register.py \
  --output outputs/{run_id}/integration_test_report.json
```

如果数据集是方案 A/B（未用 Python 注册），可省略 `--custom-register-path`。

### 8.5 registered_smoke_test（阶段 registered/05）

- **优先使用** `swift infer` 或 `PtEngine` 做推理验证。
- 如果原生推理接口不可用，再写模型特定推理脚本。
- **输出**：`outputs/{run_id}/smoke_test_report.json`。

### 8.6 registered_training_script（阶段 registered/06）

- **模板**：`templates/training_script_registered.template.sh`。
- **关键区别**：**不需要 `--custom_register_path`**。
- `--model` 优先使用 `assembled_model_path`（如果存在），否则使用 `model_path`。
- 如果 `registered_model_info.json` 中有 `component_trainable`，调用 `generate_freeze_args.py` 生成 `{{freeze_args}}`。
- 示例（7 卡 LoRA）：

```bash
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --model_type qwen2_audio \
  --model {{model_path}} \
  --dataset combined_asr_custom \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 8 \
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
  --output_dir output/qwen2_audio_combined_asr \
  --report_to none
```

### 8.7 registered_full_training（阶段 registered/07）

- 执行生成的 `run_*.sh`。
- 监控训练日志与 checkpoint。

---

## 9. Pipeline 状态管理

### 9.1 状态文件

`outputs/{run_id}/pipeline_state.json` 记录了：

```json
{
  "run_id": "20260629-143052",
  "input": { ... },
  "path": "custom",
  "environment": { "docker_image": "..." },
  "current_stage": "dataset_register",
  "overall_status": "in_progress",
  "stages": [
    { "id": "hardware_probe", "name": "Hardware Probe", "status": "completed", ... }
  ]
}
```

### 9.2 状态变更命令

```bash
# 标记开始
python .swift-adapter-agent/pipeline.py --start-stage dataset_register --run-id {run_id}

# 标记完成
python .swift-adapter-agent/pipeline.py --complete-stage dataset_register \
  --run-id {run_id} \
  --validation-report outputs/{run_id}/validation_dataset.json

# 标记失败
python .swift-adapter-agent/pipeline.py --fail-stage dataset_register \
  --run-id {run_id} \
  --error-file outputs/{run_id}/error_dataset.json
```

---

## 10. 调试指南

### 10.1 分阶段单独调试

每个阶段都可以独立运行，无需重跑整个 pipeline。

**Swift 支持检查**：

```bash
python .swift-adapter-agent/executors/check_swift_support.py \
  --model-family {model_family} \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/swift_support_report.json
```

**Dataset Register 验证**：

```bash
python3 -m py_compile outputs/{run_id}/custom/{model}_swift_register.py

python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_dataset_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/validation_dataset.json
```

**Integration Test**：

```bash
# Custom path
python .swift-adapter-agent/executors/run_integration_tests.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} --model-type {model_type} --dataset-name {dataset_name} \
  --output outputs/{run_id}/integration_test_report.json

# Registered path
python .swift-adapter-agent/executors/run_registered_integration_tests.py \
  --model-type {model_type} --model {model_path} --dataset-name {dataset_name} \
  --output outputs/{run_id}/integration_test_report.json
```

**Training Script 语法检查**：

```bash
bash -n outputs/{run_id}/run_{model}_{dataset}.sh
```

### 10.2 手动覆盖状态

调试时如果确定某个阶段已经通过，可以手动标记：

```bash
python .swift-adapter-agent/pipeline.py --complete-stage dataset_register \
  --run-id {run_id} \
  --validation-report outputs/{run_id}/validation_dataset.json
```

> 注意：不要跳过真实验证，除非你已经确认该阶段没问题。

### 10.3 Registered Path 快速验证

```bash
python3 - <<'PY'
from swift.llm import get_model_tokenizer, get_template
model, processor = get_model_tokenizer('/workspace/model/Qwen2-Audio-7B', model_type='qwen2_audio')
template = get_template('qwen2_audio', processor)
print('Model and template loaded successfully')
PY
```

---

## 11. 输出产物说明

| 文件 | 说明 |
|------|------|
| `outputs/{run_id}/pipeline_state.json` | 完整 pipeline 状态 |
| `outputs/{run_id}/swift_support_report.json` | ms-swift 支持检查结果 |
| `outputs/{run_id}/custom/{model}_swift_register.py` | Custom Path 的 model/template/dataset 注册代码 |
| `outputs/{run_id}/custom/{model}_dataset_register.py` | Registered Path 可选的 dataset 注册代码 |
| `outputs/{run_id}/assembled_model/` | Registered Path 按组件初始化时生成的 checkpoint |
| `outputs/{run_id}/checkpoint_assembly_report.json` | 组件组装报告 |
| `outputs/{run_id}/freeze_args_report.json` | 按组件训练策略生成的冻结参数 |
| `outputs/{run_id}/run_{model}_{dataset}.sh` | 训练脚本 |
| `outputs/{run_id}/Dockerfile` | 环境 Dockerfile |
| `outputs/{run_id}/model_analysis.json` | Custom Path 模型分析报告 |
| `outputs/{run_id}/user_decision.json` | 用户确认的训练策略 |
| `outputs/{run_id}/download_report.json` | 权重下载报告 |
| `outputs/{run_id}/integration_test_report.json` | 集成测试报告 |
| `outputs/{run_id}/smoke_test_report.json` | 冒烟测试报告 |

---

## 12. 最佳实践与禁止行为

### 12.1 最佳实践

- 先在 `swift_support_check` 判断路径，再读对应 skill。
- Custom Path 必须在 `user_decision` 阶段让用户确认每个组件策略。
- 下载权重前先用 `find_local_model.py` 检查本地是否存在，避免重复下载。
- 所有 Python 代码写完后先用 `python3 -m py_compile` 检查语法。
- 所有 shell 脚本写完后先用 `bash -n` 检查语法。
- 训练脚本中 GPU 数量不要超过 7。
- 每个阶段失败后最多重试 3 次，仍失败则停止并向用户报告 blocker。

### 12.2 禁止行为

- 跳过 validator。
- 跳过 `swift_support_check`。
- 在 Custom Path 跳过 `user_decision`。
- 不下载权重就进入 `model_register`。
- 一次写多个阶段代码再一起验证。
- 把代码写到 `outputs/{run_id}/custom/` 之外。
- 在 Custom Path 和 Registered Path 之间混用 skill。

---

## 13. 常见问题（FAQ）

**Q1：如何判断我的模型是否被 ms-swift 原生支持？**  
运行 `executors/check_swift_support.py`。它会解析 `ms-swift/swift/llm/model/constant.py` 并给出结论。

**Q2：如果 `check_swift_support.py` 返回 `supported: false`，但我确定 ms-swift 支持，怎么办？**  
检查 `model_family` 拼写、ms-swift 版本，或手动查看 `constant.py`。必要时可手动切换路径。

**Q3：已注册模型还需要写 register_model / register_template 吗？**  
不需要。只需要处理数据集，并生成不带 `--custom_register_path` 的训练脚本。

**Q4：数据集已经是标准格式，是否还需要写 register_dataset？**  
不需要。直接通过 `--dataset data/xxx.jsonl` 传入即可。

**Q5：训练脚本里的 batch size 怎么定？**  
参考 smoke test 阶段的 batch size 上限测试结果，结合可用 GPU 数量设定。

**Q6：我能否只调试某个失败阶段，而不用重跑整个 pipeline？**  
可以。所有 executor/validator 都可以独立运行，详见第 10 章。

---

## 附录 A：命令速查表

| 用途 | 命令 |
|------|------|
| 初始化 | `python .swift-adapter-agent/pipeline.py --init --input input.json` |
| 查看状态 | `python .swift-adapter-agent/pipeline.py --status --run-id {run_id}` |
| Swift 支持检查 | `python .swift-adapter-agent/executors/check_swift_support.py --model-family {family} --sure-train-dir SURE_train --output outputs/{run_id}/swift_support_report.json` |
| 切换路径 | `python .swift-adapter-agent/pipeline.py --switch-path registered --run-id {run_id}` |
| 组件组装 checkpoint | `python .swift-adapter-agent/executors/assemble_registered_checkpoint.py --model-family {family} --model-type {type} --base-model-path {base} --component-paths-json {json} --output-dir outputs/{run_id}/assembled_model` |
| 生成冻结参数 | `python .swift-adapter-agent/executors/generate_freeze_args.py --model-family {family} --model-type {type} --component-trainable-json {json} --output outputs/{run_id}/freeze_args_report.json` |
| 标记完成 | `python .swift-adapter-agent/pipeline.py --complete-stage {stage_id} --run-id {run_id} --validation-report {file}` |
| Python 语法检查 | `python3 -m py_compile outputs/{run_id}/custom/*.py` |
| Bash 语法检查 | `bash -n outputs/{run_id}/run_*.sh` |

## 附录 B：模型默认值表

见 4.2 节 `model_family` 表。
