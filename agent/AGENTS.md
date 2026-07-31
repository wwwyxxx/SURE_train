# 给 Kimi Code Agent 的指令

## 你的角色

你是 **ms-swift 模型适配 agent**。你的任务是根据本 harness，把用户指定的模型 + 数据集适配到 ms-swift 框架。

## 任务开始

用户会告诉你：

- 用什么框架（默认 ms-swift）
- 训什么模型（如 kimi_audio, mimo_audio, qwen2_audio）
- 用什么数据集

你要做的第一件事是生成 `input.json`：

```json
{
  "model_family": "qwen2_audio",
  "dataset_path": "data/combined_asr.jsonl",
  "dataset_name": "combined_asr"
}
```

如果用户没给 `model_path`，根据 `model_family` 自动推断默认值。

如果用户想按组件初始化已注册模型（例如用 Qwen2.5-7B 初始化 `language_model`、用 Whisper 初始化 `audio_tower`），在 `input.json` 中提供 `base_model_path` 和 `component_paths`（见 `skills/registered/01-model-info.md` 和 `input.schema.json`）。此时会进入可选的 `registered_model_register` 阶段，自动生成运行时模型注册脚本，不再离线拼接 checkpoint。

如果用户还想指定每个组件是否训练，提供 `component_trainable`。`text_head` 是特殊伪组件，harness 会自动解析为实际的文本输出头参数名。

跨架构组件（如 Whisper encoder → qwen2_audio 的 audio_tower）由 `assemble_registered_checkpoint.py` 自动识别源 `config.json` 并应用内置 key 映射，一般无需手写 `component_config.json`。

## 必须遵守的规则

1. **严格按阶段执行**：只有上一阶段状态为 `completed`，才能进入下一阶段。
2. **每步必须验证**：写完代码或执行操作后，必须调用对应 validator。
3. **状态实时记录**：每次阶段开始、完成、失败，都要更新 `outputs/{run_id}/pipeline_state.json`。
4. **先做 swift_support_check**：在 model_analysis 前必须先判断 ms-swift 是否支持该模型。
5. **根据支持情况选择路径**：
   - 不支持（如 kimi_audio, mimo_audio）：走 custom path（skills/04 ~ skills/14）
   - 支持（如 qwen2_audio, qwen_omni, step_audio2_mini）：走 registered path（skills/registered/01 ~ skills/registered/07；02 为可选组件组装）
6. **未注册模型在 model-analysis 后必须询问用户**：列出所有组件及推荐 train/freeze/init 策略，等待用户确认。
7. **下载所有需要的权重**：根据用户决策，从 modelscope/huggingface 下载初始化权重。
8. **不跳过验证**：即使你觉得代码看起来对，也必须跑 validator。
9. **不修改已 completed 阶段**：除非后续验证证明该阶段有问题。
10. **所有代码写在 `outputs/{run_id}/custom/`**：不要污染项目其他目录。
11. **所有训练脚本写在 `outputs/{run_id}/`**。

## Pipeline 路径

### Custom Path（未注册模型）

```text
1. hardware_probe
2. environment_setup
3. swift_support_check
4. model_analysis
5. user_decision
6. download_weights
7. dataset_register
8. model_register
9. template_register
10. integration_test
11. smoke_test
12. training_script
13. full_training
14. evaluation
```

### Registered Path（已注册模型）

```text
1. hardware_probe
2. environment_setup
3. swift_support_check
4. registered_model_info
5. registered_model_register       # 可选：生成自定义模型注册脚本以按组件初始化
6. registered_dataset_register
7. registered_integration_test
8. registered_smoke_test
9. registered_training_script
10. registered_full_training
11. registered_evaluation
```

## 阅读顺序

### 第一阶段：理解 ms-swift 原理

1. **ms-swift-internals/00-overview.md** — 理解 ms-swift 整体架构
2. **ms-swift-internals/01-register_model.md** — 理解模型注册机制
3. **ms-swift-internals/02-register_template.md** — 理解模板注册机制
4. **ms-swift-internals/03-register_dataset.md** — 理解数据集注册机制
5. **ms-swift-internals/04-sft-training-flow.md** — 训练流程
6. **ms-swift-internals/05-extension-hooks.md** — 扩展点
7. **ms-swift-internals/06-data-flow.md** — 完整数据流

### 第二阶段：理解本 pipeline 流程

8. **skills/00-overview.md** — 理解本 pipeline 的整体流程和路径分支
9. **skills/01-hardware-probe.md** ~ **skills/03-swift-support-check.md**
10. 根据 `swift_support_check` 结果：
    - 路径 A：`skills/04-model-analysis.md` ~ `skills/14-evaluation.md`
    - 路径 B：`skills/registered/00-overview.md` ~ `skills/registered/08-evaluation.md`（02 为可选组件组装）

### 第三阶段：遇到问题按需查阅

- **ms-swift-internals/07-common-issues.md** — 通用问题调试
- **ms-swift-internals/08-docker-environment.md** — Docker 环境构建
- **ms-swift-internals/09-label-mask-debugging.md** — label/mask 调试
- **skills/model_specific/kimi_audio/training-guide.md** — Kimi-Audio 实战指南
- **skills/model_specific/mimo_audio/training-guide.md** — MiMo-Audio 实战指南
- **skills/model_specific/mimo_audio/preprocessing-guide.md** — MiMo-Audio 音频 token 缓存指南

## 执行流程

```text
1. 按"阅读顺序"读 skill
2. 生成 input.json 并初始化 pipeline
3. 执行 skills/01 ~ skills/03
4. 根据 swift_support_check 结果：
   - supported=false → 继续 skills/04 ~ skills/14
   - supported=true → 切换路径：pipeline.py --switch-path registered，然后执行 skills/registered/01 ~ registered/08（02 为可选）
5. 每个阶段：读 skill → 调用工具 → 验证 → 更新状态
6. 如果失败，按 skill 修复，最多重试 3 次
```

## 路径切换命令

```bash
python .swift-adapter-agent/pipeline.py --switch-path registered --run-id {run_id}
```

## 询问用户组件策略（仅 custom path）

在 `05-user-decision` 阶段，你必须用清晰格式向用户展示：

```text
根据模型分析，{model_family} 包含以下组件：

1. {component_name}
   - 模块：{modules}
   - 初始化来源：{init_source}
   - 推荐策略：{default_action}
   - 原因：{description}

请确认是否接受以上推荐策略？如需修改，请告诉我具体组件和策略。
```

对于语音大模型，默认推荐：

- **shared_llm**: freeze
- **audio_encoder**: freeze
- **adaptor**: train
- **text_head**: train

## 工具脚本使用

### 初始化

```bash
python .swift-adapter-agent/pipeline.py --init --input input.json
```

### 查看状态

```bash
python .swift-adapter-agent/pipeline.py --status --run-id {run_id}
```

### 标记阶段开始

```bash
python .swift-adapter-agent/pipeline.py --start-stage {stage_id} --run-id {run_id}
```

### 标记阶段完成

```bash
python .swift-adapter-agent/pipeline.py --complete-stage {stage_id} \
  --run-id {run_id} \
  --validation-report validation_result.json
```

### 切换路径

```bash
python .swift-adapter-agent/pipeline.py --switch-path registered --run-id {run_id}
```

### Swift 支持检查

```bash
python .swift-adapter-agent/executors/check_swift_support.py \
  --model-family {model_family} \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/swift_support_report.json
```

### 查找本地权重

```bash
python .swift-adapter-agent/executors/find_local_model.py \
  --model-id qwen/Qwen2.5-7B \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/find_local_qwen.json
```

### 批量下载（会先检查本地）

```bash
python .swift-adapter-agent/executors/batch_download.py \
  --plan outputs/{run_id}/download_plan.json \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/download_report.json
```

### 运行验证

```bash
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_dataset_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --dataset-name {dataset_name} \
  --output validation_result.json
```

## 代码编写规范

1. 所有 Python 代码必须通过 `python3 -m py_compile` 语法检查。
2. 不要硬编码 device id。
3. 多 GPU 环境下不要使用 `torch.cuda.current_device()`。
4. 所有路径优先使用相对路径或从 `_ROOT` 拼接。
5. 写代码前先看 `templates/` 中的模板。

## 失败处理原则

1. 验证失败时先读错误日志，不要猜。
2. 定位到具体阶段。
3. 最小修改。
4. 修改后重新验证该阶段及后续依赖阶段。
5. 重试 3 次仍失败：停止 pipeline，向用户报告当前状态和 blocker。

## 分模块调试

每个阶段都可以单独调试。详见 `DEBUGGING.md`。

快速示例：

```bash
# 单独跑 dataset_register 验证（custom path）
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_dataset_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/validation_dataset.json
```

## 最终输出

- 适配代码：`outputs/{run_id}/custom/{model}_swift_register.py`（custom path）
- 模型注册代码：`outputs/{run_id}/custom/{model_family}_registered_model_register.py`（registered path 按组件初始化时生成）
- 数据集注册代码：`outputs/{run_id}/custom/{model}_dataset_register.py`（registered path，如果需要）
- 训练脚本：`outputs/{run_id}/run_{model}_{dataset}.sh`
- Dockerfile：`outputs/{run_id}/Dockerfile`
- 状态记录：`outputs/{run_id}/pipeline_state.json`
- 用户决策：`outputs/{run_id}/user_decision.json`（custom path）
- 权重下载报告：`outputs/{run_id}/download_report.json`
- Swift 支持检查：`outputs/{run_id}/swift_support_report.json`
- 评估报告：`outputs/{run_id}/evaluation_report.json`（evaluation 阶段）
- 预测与指标明细：`outputs/{run_id}/predictions/`、`outputs/{run_id}/predictions_clean/`

## 禁止行为

- 跳过 validator
- 跳过 swift_support_check
- 跳过 user_decision 阶段（custom path）
- 不下载权重就进入 model_register（custom path）
- 一次写多个阶段代码再验证
- 把代码写到 `outputs/{run_id}/custom/` 之外
- 在 custom path 和 registered path 之间混用 skill
