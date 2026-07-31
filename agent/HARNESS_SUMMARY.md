# ms-swift 模型适配 Agent Harness 详细总结

> 面向论文写作的系统总结，覆盖：(1) Pipeline 整体流程；(2) 每阶段 Validation 机制；(3) 最终结果产物。
> 内容均与 `.swift-adapter-agent/` 下的实际文件（`skills/`、`executors/`、`validators/`、`pipeline.py`、`schemas/`、`outputs/` 真实运行记录）逐一核对。

---

## 一、系统定位与设计理念

### 1.1 定位

`.swift-adapter-agent/` 是一套为 **LLM 编码 Agent（Kimi Code）** 设计的 **harness（执行脚手架）**，目标是把**任意 Hugging Face 语音/多模态模型 + 自定义数据集**适配到 [ms-swift](https://github.com/modelscope/ms-swift) 训练框架，完成从环境探测到正式训练、测试集评估（CER/WER）的端到端流程。

### 1.2 核心设计原则

| 原则 | 含义 |
|------|------|
| **Harness 不调用 LLM API** | Harness 本身是纯确定性的工具集合（Python 脚本 + Markdown 手册 + JSON Schema）；所有理解、决策、代码编写均由 Agent 在本地完成。 |
| **Agent 是执行者** | Agent 按手册（skills）执行，Harness 提供：操作手册、工具脚本（executors）、验证脚本（validators）、状态追踪（pipeline.py）。 |
| **阶段化 + 强制验证** | Pipeline 被切分为有序阶段；每阶段必须通过对应 validator 才能标记 `completed`，禁止跳过验证。 |
| **状态可追踪、可恢复** | 每次运行的全部状态持久化在 `outputs/{run_id}/pipeline_state.json`，是唯一的"事实来源"（single source of truth）。 |
| **双路径自适应** | 根据 ms-swift 是否原生支持目标模型，自动在 **Custom Path（自定义注册）** 与 **Registered Path（原生支持）** 之间切换。 |
| **人在环路（Human-in-the-loop）** | Custom Path 设有强制性的 `user_decision` 阶段：组件训练/冻结策略必须由用户确认，Agent 不允许自行决定。 |

### 1.3 四大组件

```text
.swift-adapter-agent/
├── AGENTS.md / README.md / manual.md     # 给 Agent 的入口指令 + 给人看的文档
├── pipeline.py                           # 状态机 CLI（初始化/开始/完成/失败/切换路径/查询）
├── input.schema.json                     # 输入 JSON Schema（draft-07）
├── schemas/pipeline_state.schema.json    # 状态文件 JSON Schema（draft-07）
├── skills/                               # 阶段操作手册（Agent 的"检查清单"）
│   ├── 00-overview.md ~ 14-evaluation.md # Custom Path 手册（14 阶段）
│   ├── registered/00 ~ 08                # Registered Path 手册（11 阶段）
│   ├── ms-swift-internals/               # ms-swift 框架内部原理参考（10 篇）
│   └── model_specific/                   # 模型实战指南（kimi_audio/mimo_audio/qwen2_audio）
├── executors/                            # 确定性工具脚本（Agent 调用）
├── validators/                           # 验证脚本（core/ 通用 + model_specific/ 模型专属）
├── templates/                            # 代码模板（Dockerfile、训练脚本、模型注册适配器）
└── outputs/{run_id}/                     # 每次运行的全部产物
```

**三层知识架构**：

1. `skills/ms-swift-internals/`（00~09）：ms-swift 框架原理——三大注册机制（`register_model` / `register_template` / `register_dataset`）、SFT 训练流、扩展 hook、数据流、常见问题、Docker 环境、label/mask 调试。Agent 在写适配代码前必须先读。
2. `skills/`（00~14）与 `skills/registered/`（00~08）：告诉 Agent **每个阶段做什么、调什么工具、怎么验证**。
3. `skills/model_specific/`：已适配模型的实战经验（Kimi-Audio 训练指南、MiMo-Audio 训练指南 + 音频 token 缓存指南、Qwen2-Audio 训练指南），用于把踩过的坑固化成知识。

---

## 二、Pipeline 整体流程

### 2.1 输入规范

输入是一个 `input.json`（schema 见 `input.schema.json`），**最小只需 3 个字段**：

```json
{
  "model_family": "kimi_audio",
  "dataset_path": "data/combined_asr_aishell-1.jsonl",
  "dataset_name": "combined_asr_aishell_1"
}
```

完整字段（均为 draft-07 强类型约束）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_family` | enum(string) | ✅ | 模型族：`kimi_audio` / `mimo_audio` / `qwen2_audio` / `qwen_audio` / `qwen_omni` / `qwen2_5_omni` / `qwen3_omni` / `step_audio` / `step_audio2_mini` / `qwen2` / `llama3` / `other` |
| `model_path` | string | ❌ | 单一权重路径；为空时由 `pipeline.py` 按 `model_family` 查表自动推断（`DEFAULT_MODEL_PATHS`） |
| `base_model_path` | string/null | ❌ | Registered Path 组件化初始化时：官方 checkpoint，用于复制 config/tokenizer/processor |
| `component_paths` | object/null | ❌ | 按组件初始化来源：`{组件名: 路径 或 "random"}`，如 `{language_model, vision_tower, aligner, generator}` |
| `component_trainable` | object/null | ❌ | 按组件训练策略（bool）；`text_head` 为伪组件，harness 自动解析为真实参数名 |
| `additional_trainable_parameters` | array/null | ❌ | 自动 text_head 解析不足时，额外指定可训练参数前缀 |
| `whisper_path` | string/null | ❌ | 仅 custom audio 模型（kimi/mimo）需要 |
| `dataset_path` / `dataset_name` | string | ✅ | 数据集 jsonl 路径与注册名 |
| `source_code_url` / `paper_url` | string/null | ❌ | 模型源码/论文（本地路径或 URL）；Custom Path 建议填，Registered Path 不需要 |
| `max_gpus` | integer | ❌ | 训练最多使用的 GPU 数，**默认 7** |
| `output_run_id` | string/null | ❌ | 自定义 run_id，否则自动生成 UTC 时间戳 `YYYYMMDD-HHMMSS` |

`pipeline.py --init` 会：推断默认路径 → 创建 `outputs/{run_id}/` → 生成初始 `pipeline_state.json`（所有阶段 `pending`）。

### 2.2 双路径总览

Pipeline 共 **3 个公共阶段 + 路径专属阶段**。第 3 阶段 `swift_support_check` 是分叉点：

```text
input.json
   │
   ▼
① hardware_probe          （公共）
② environment_setup       （公共）
③ swift_support_check     （公共）── 解析 ms-swift 源码 constant.py + examples/
   │
   ├─ supported=false ──► Custom Path（14 阶段，适用于 kimi_audio / mimo_audio / slam_llm / tasu 等）
   │   ④ model_analysis → ⑤ user_decision → ⑥ download_weights →
   │   ⑦ dataset_register → ⑧ model_register → ⑨ template_register →
   │   ⑩ integration_test → ⑪ smoke_test → ⑫ training_script →
   │   ⑬ full_training → ⑭ evaluation
   │
   └─ supported=true ───► Registered Path（11 阶段，适用于 qwen2_audio / qwen2_5_omni / step_audio2_mini 等）
       ④ registered_model_info → ⑤ registered_model_register（可选）→
       ⑥ registered_dataset_register → ⑦ registered_integration_test →
       ⑧ registered_smoke_test → ⑨ registered_training_script →
       ⑩ registered_full_training → ⑪ registered_evaluation
```

路径切换命令：`python pipeline.py --switch-path registered --run-id {run_id}`。
状态机在切换时**保留公共阶段状态**、重建路径专属阶段列表；一旦任何路径专属阶段已开始，切换被拒绝（防止混用）。

### 2.3 公共阶段（3 个，两条路径共用）

| # | 阶段 ID | 目标 | 关键执行器 | 产物 |
|---|---------|------|-----------|------|
| 1 | `hardware_probe` | 探测 GPU 资源，**硬性约束最多使用 7 张 GPU** | `hardware_probe.py --max-gpus 7` | `hardware.json`（gpu_count、usable_gpu_count、型号/显存/算力、CUDA 版本） |
| 2 | `environment_setup` | 为模型找到或构建 Docker 镜像 | `resolve_docker_image.py`（三层查找）→ `build_docker.py` | `docker_resolution.json`、`docker_build_report.json`、（必要时）`SURE_train/Dockerfile/<model>_dockerfile/Dockerfile` |
| 3 | `swift_support_check` | **不依赖论文/源码**，纯解析 ms-swift 源码判断原生支持 | `check_swift_support.py`（静态解析 `swift/llm/model/constant.py` + `examples/`，不 import swift） | `swift_support_report.json`（supported、model_type、support_level、category、inference_engine、是否有训练/推理示例） |

**Docker 镜像命名规范**：`docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-<模型名>:<版本>`（模型名去 `_`/`-` 全小写，如 `kimi_audio→kimiaudio`）。

**三层镜像查找逻辑**：① `docker images` 本地已有 → `use_existing`；② Dockerfile 已存在 → `build_existing`；③ 都没有 → 按模板新建（需显式 `--create-if-missing`）→ `build_new`。

**已知支持映射**（写死在 `check_swift_support.py` 的规则表中）：

| harness model_family | ms-swift model_type | 原生支持 |
|---|---|---|
| qwen2_audio / qwen_audio | qwen2_audio / qwen_audio | ✅ |
| qwen_omni / qwen2_5_omni / qwen3_omni | qwen2_5_omni / qwen3_omni | ✅ |
| step_audio / step_audio2_mini | step_audio / step_audio2_mini | ✅ |
| kimi_audio / mimo_audio | 无 | ❌（走 Custom Path） |
| other | 动态判断 | 视解析结果 |

### 2.4 Custom Path 详解（14 阶段）

适用对象：ms-swift 不原生支持的模型（kimi_audio、mimo_audio、slam_llm、tasu 等）。核心工作量是**手写三个注册**（dataset / model / template），产物统一在 `outputs/{run_id}/custom/{model}_swift_register.py` 一个文件中。

| # | 阶段 ID | 是否写代码 | 目标与关键动作 |
|---|---------|-----------|----------------|
| 4 | `model_analysis` | 否（写分析报告） | 读 `config.json` + 论文 + 源码；列出全部**组件**（可独立决定 train/freeze/init 的模块）；必须回答 11 个问题：基础 LLM、forward 签名、**音频 tokenizer 在 LLM 内部还是外部**（决定是否需要预缓存 audio tokens）、输入是连续特征还是离散 token、adaptor、text head、需忽略的音频输出部件、是否需要 wrapper、special tokens 扩展。**ASR 场景下 audio decoder/head 一律标记 ignore。** |
| 5 | `user_decision` | 否（询问用户） | **强制人在环路**：向用户展示每个组件的推荐策略（语音大模型默认：shared_llm=freeze、audio_encoder=freeze、adaptor=train、text_head=train）+ 下载清单；用户确认后写 `user_decision.json` + `download_plan.json`。Agent 被明确禁止跳过或自行决定。 |
| 6 | `download_weights` | 否 | `find_local_model.py` 先查 `SURE_train/model/` 本地权重（优先复用、不重复下载）；缺失的走 `batch_download.py`（ModelScope 优先，`--use-hf` 回退 HuggingFace）；失败生成 `missing_weights.json` 请用户提供链接，最多重试 3 次。 |
| 7 | `dataset_register` | ✅ | 读数据集前 5 行识别字段（wav/txt/prompt）；在 `{model}_swift_register.py` 中写 `ResponsePreprocessor`（输出 `{messages, audios}`）+ `register_dataset`；`py_compile` 语法检查。 |
| 8 | `model_register` | ✅ | 按 `model_analysis` 决策：是否需要 wrapper（如 Kimi-Audio 的 Whisper 在 forward 内部）、是否需要权重复制（`_copy_weights`，如 lm_head 部分行复制）、冻结策略（`_freeze`）；实现 `get_model_tokenizer_xxx` + `register_model(ModelMeta)`。 |
| 9 | `template_register` | ✅ | 按 forward 签名设计 `_encode` 输出 keys；实现 `data_collator`（padding + loss mask）。**关键原则（两大模型都踩过的坑）：labels 不预 shift——model forward 内部做 causal shift，ms-swift 的 compute_acc 再 shift 一次；若 token_acc≈0 首先查是否双重 shift。** padding 处 labels=-100；special tokens（`<|sosp|>` 等）动态加入 tokenizer。 |
| 10 | `integration_test` | 否（跑验证） | `run_integration_tests.py` 顺序跑 **6 个 core validator + 动态发现的 model_specific validator**；失败按类型回退（forward→model/template register；loss→template register；freeze→model register）。 |
| 11 | `smoke_test` | 否（跑验证） | **正确性最终守门员**，三轮测试（详见 §3.4）；Agent 必须先写模型专属推理脚本 `executors/model_specific/{family}/infer_{family}.py`（harness 不提供通用推理脚本，因为各模型 forward 签名差异巨大）。 |
| 12 | `training_script` | ✅（bash） | 按 `templates/training_script.template.sh` 生成 `run_{model}_{dataset}.sh`；`NPROC_PER_NODE = min(usable_gpu_count, max_gpus) ≤ 7`；含 `--custom_register_path`；scheduler 按场景选（小数据 overfit 用 constant+warmup 0，大数据用 cosine+warmup 0.03）；`bash -n` 语法检查。 |
| 13 | `full_training` | 否 | `start_training.sh`（nohup 后台启动 + 写 PID）；监控 loss/token_acc/OOM/NaN；确认 checkpoint 保存。 |
| 14 | `evaluation` | 否 | 用模型自己的推理脚本在标准测试集推理 → 文本归一化 → CER/WER → 评估报告（详见 §3.5 与 §4.4）。 |

### 2.5 Registered Path 详解（11 阶段）

适用对象：ms-swift 原生支持的模型（qwen2_audio、qwen2_5_omni、qwen3_omni、step_audio2_mini 等）。**不读论文/源码、不做组件分析、通常不写 model/template register**，重点是把数据集和训练参数配对。

| # | 阶段 ID | 是否写代码 | 目标与关键动作 |
|---|---------|-----------|----------------|
| 4 | `registered_model_info` | 否 | 确认 model_type / model_path（本地是否存在）、搜 ms-swift examples/docs、确认特殊参数（如 qwen2_5_omni 需 `MAX_PIXELS=1003520`、`ENABLE_AUDIO_OUTPUT=0`）、推理方式（`swift infer` 还是 `PtEngine`）；写 `registered_model_info.json`。**可选**：收集 `component_paths` + `component_trainable` 进入组件化初始化。 |
| 5 | `registered_model_register` | 可选（自动生成） | **仅当提供 `component_paths` 时触发**。`generate_registered_model_register.py` 根据 `templates/model_register/{model_type}.adapter.py`（专用适配器：qwen2_5_omni / qwen2_audio / step_audio2_mini；否则回退 `default.adapter.py` 整子模块替换）生成运行时注册脚本 `{family}_registered_model_register.py`：注册 derived `model_type`（如 `qwen2_5_omni_custom`），在加载时动态替换组件（如 LLM←Qwen2.5-7B、audio encoder←Whisper），**取代旧的离线 checkpoint 拼接**（`assemble_registered_checkpoint.py` 已弃用保留）。 |
| 6 | `registered_dataset_register` | 是/否 | 三档方案：**A** 数据已是标准格式（`{messages, audios}`）→ 直接传 jsonl 路径，零代码；**B** 简单字段映射 → `dataset_info.json` + `--custom_dataset_info`；**C** 复杂预处理 → Python `register_dataset`（`{model}_dataset_register.py` + `--custom_register_path`）。 |
| 7 | `registered_integration_test` | 否 | `run_registered_integration_tests.py`（轻量版）：模型可加载 → template 可获取 → 数据集可加载 → forward 可跑 → loss 可计算 → single-step 训练可跑。无需 model-specific validator。 |
| 8 | `registered_smoke_test` | 否 | 同样三轮 smoke test，但**推理优先用 `swift infer` / `PtEngine`**，不必手写推理脚本；组件化初始化时推理必须带 `--external_plugins`（重建基座组件后**只加载训练过的参数**，不能整体 load_state_dict）。 |
| 9 | `registered_training_script` | ✅（bash） | 按 `templates/training_script_registered.template.sh` 生成；组件化时：`--model_type {derived}` + `--model {base_model_path}` + `--external_plugins {register}`；`component_trainable` 经 `generate_freeze_args.py` 转成 CLI 冻结参数（注册脚本默认冻结全部 → `--freeze_parameters_regex '.*'` + `--trainable_parameters ...` 选择性解冻）。 |
| 10 | `registered_full_training` | 否 | 同 Custom Path：启动 + 监控。 |
| 11 | `registered_evaluation` | 否 | 同 Custom Path 的评估流程，推理优先 `swift infer` / `PtEngine`。 |

**组件化初始化的两个关键技术点**（手册中固化的高频坑）：

1. **词表与 embedding/lm_head 对齐**：替换 LLM 时**保持 base model 的 vocab_size**；源 LLM 的真实词表大小用 `len(tokenizer)` 而非 `config.vocab_size`（后者含 padding 行）；公共 token 复用源权重，base 独有 token 随机初始化（`std = hidden_size^-0.5`）。典型例：Step-Audio-2-mini（158720）+ Qwen2.5-7B（151665，前 151665 token 完全一致，剩 7055 随机初始化）。
2. **`--freeze_llm/--freeze_vit/--freeze_aligner` 依赖 `model_arch`**：自定义注册必须用 `register_model_arch(MultiModelKeys(...))` 声明组件归属，否则 CLI 冻结参数静默失效。

组件元数据沉淀在 `executors/data/`：
- `registered_model_arch_components.json`：48 种 model_type 的组件→模块前缀映射（language_model / vision_tower / aligner / generator）。
- `registered_model_text_head.json`：各 model_type 的 text head 参数名（如 `qwen2_5_omni→thinker.lm_head`、`step_audio2_mini→model.lm_head`）+ 回退规则。
- `registered_component_key_mappings.json`：跨架构权重 key 映射（如 Whisper encoder → qwen2_audio 的 audio_tower：`model.encoder. → audio_tower.`，`drop_unmapped=true`）。

### 2.6 状态机与执行纪律

**状态机（`pipeline.py`，437 行，纯本地 CLI，不调用 LLM）**：

- 6 类命令：`--init` / `--status` / `--start-stage` / `--complete-stage`（挂 `--validation-report`）/ `--fail-stage`（挂 `--error-file`，自动 `retry_count+1`）/ `--switch-path`。
- **顺序强制**：`check_stage_order` 要求前一阶段 `completed` 才允许开始下一阶段。
- **失败处理纪律**（写进 AGENTS.md/skills）：① 读日志不靠猜；② 定位到具体阶段；③ 最小修改；④ 重跑该阶段及下游依赖验证；⑤ **单阶段最多重试 3 次**，仍失败则停止 pipeline 并向用户报告 blocker。
- **禁止行为**（AGENTS.md 明列）：跳过 validator、跳过 swift_support_check、Custom Path 跳过 user_decision、不下载权重就进 model_register、一次写多阶段代码再验证、代码写到 `outputs/{run_id}/custom/` 之外、两条路径混用 skill。
- **代码规范**：Python 必过 `py_compile`；bash 必过 `bash -n`；不硬编码 device id；多卡下禁用 `torch.cuda.current_device()`；路径优先相对路径。

---

## 三、每阶段 Validation 机制

### 3.1 验证器总体设计

**两层验证器体系**：

```text
validators/
├── core/                    # 14 个模型无关验证器：检查任何 ms-swift 集成都必须满足的不变量
└── model_specific/{family}/ # 模型专属验证器：编码特定架构的已知失败模式
    ├── kimi_audio/  (4 个)
    ├── mimo_audio/  (4 个)
    └── tasu/        (1 个 e2e)
```

**统一执行契约**：
- 每个 validator 是独立 Python 脚本，接受 `--custom-register-path` + 阶段相关参数（`--model` / `--model-type` / `--dataset-name` 等）；
- 动态 import 生成的注册模块，调用 ms-swift API（`load_dataset` / `get_model_tokenizer` / `get_template` / `data_collator`）；
- **exit code 0 = 全部断言通过，非 0 = 失败**；
- 新增模型专属验证器零配置：`validators/model_specific/{family}/validate_*.py` 会被 `run_integration_tests.py` 按 glob 自动发现（按字母序执行）。

**执行包装器 `executors/run_validator.py`**：子进程运行 validator，**默认 600s 超时**（大模型加载/CUDA 编译可能 hang），产出结构化 JSON 报告：

```json
{
  "passed": true,
  "validator": "validators/core/validate_dataset_registration.py",
  "report": "...stdout 末尾 2000 字符...",
  "stdout": "...完整 stdout...",
  "stderr": "...完整 stderr...",
  "metrics": {}
}
```

脚本缺失、超时、非零退出码 → `passed=false`，输出完整保留供诊断。**该报告会被嵌入 `pipeline_state.json` 对应阶段的 `validation` 字段**，实现"状态文件自包含、可审计"。

### 3.2 Core 验证器清单（14 个，模型无关）

| 验证器 | 检查内容 |
|---|---|
| `validate_hardware.py` | 至少 1 张 GPU 可用、显存可读、CUDA 版本可获取 |
| `validate_docker_image.py` | 镜像能 `docker run --gpus` 启动并成功 `import torch; import swift; import transformers` |
| `validate_dataset_registration.py` | `load_dataset` 可加载；preprocessor 产出 `{messages, audios}`；原始 jsonl 必填字段（wav/txt/prompt）存在；引用音频文件可访问（抽样 N 条） |
| `validate_model_registration.py` | `register_model` 已按 model_type 注册；`get_model_tokenizer` 可调用；权重在预期设备上；**至少有一个可训练参数** |
| `validate_template_registration.py` | template 可按 model_type 获取；`encode` 返回预期 keys；tensor 形状/dtype 正确；`data_collator` 可组 batch |
| `validate_forward_pass.py` | template 可编码样本 → collator 可出 batch → `model(**batch)` 无错 → 输出含预期形状的 loss/logits |
| `validate_loss_computation.py` | loss 为标量且有限；**对 label 扰动敏感**（破坏 label → loss 变化）；**padding(-100) 位置不贡献 loss** |
| `validate_single_step_training.py` | 单步 forward+backward+optimizer.step 可跑通 |
| `validate_checkpoint_save_load.py` | 保存 trainable-only checkpoint → 重新加载 → **训练参数前后完全一致** |
| `validate_inference.py` | greedy 生成可行；输出非空且含文本 token；**不退化为单 token 重复** |
| `validate_freeze_unfreeze.py` | 预期冻结前缀 `requires_grad=False`、预期训练前缀 `requires_grad=True` |
| `validate_batch_size_scaling.py` | 对一组候选 batch size 逐个跑 forward+backward，报告峰值显存与是否 OOM（找最大可用 batch size） |
| `validate_distributed_launch.py` | DDP 多卡：各 rank 初始化进程组、模型被 DDP 包装、各 rank 上跑通一步 forward/backward |
| `validate_training_script.py` | 训练脚本静态检查（必要参数齐全、路径存在、GPU 数 ≤7） |

### 3.3 模型专属验证器

| 模型 | 验证器 | 针对性检查 |
|---|---|---|
| **kimi_audio** | `validate_label_shift.py` | **高危 bug 区**：`text_input_ids`/`labels`/`text_loss_mask` 形状一致；mask 恰好标记 assistant 响应 token；forward shift 后 loss 位置与预测 token 对齐；padding 为 -100 |
| | `validate_loss_mask.py` | batch 含 `text_loss_mask` 字段；loss 只来自 mask=True 位置 |
| | `validate_sequence_length.py` | 不同时长音频（5/10/20/30/60/120s）可编码、长度在 max_length 内、forward 不 OOM |
| | `validate_weight_initialization.py` | shared LLM 权重与基座 Qwen 一致；`mimo_output` 前 N 行与 Qwen `lm_head` 一致；可训练模块（vq_adaptor/mimo_layers/mimo_norm）非全零；Whisper 已加载且默认冻结 |
| **mimo_audio** | `validate_mimo_audio_model.py` | 模型注册正确性（8B 总参数、可训练参数子集） |
| | `validate_mimo_audio_template.py` | template 产出 `input_ids/attention_mask/labels/text_loss_mask`，3D input_ids 与 RVQ token 交织格式正确 |
| | `validate_mimo_audio_forward.py` | 真实样本上 forward + loss 正确 |
| | `validate_mimo_audio_integration.py` | 综合集成：forward、loss、single-step、freeze/unfreeze、推理 |
| **tasu** | `validate_tasu_e2e.py` | 端到端：dataset/model/template/forward/loss/single-step/checkpoint/freeze/inference 全链路，任一失败即非零退出 |

### 3.4 分阶段验证矩阵

**公共阶段**：

| 阶段 | 验证方式 | 通过标准 |
|---|---|---|
| hardware_probe | `validate_hardware.py` + `hardware_probe.py` 输出自洽 | `usable_gpu_count ≥ 1` 且 ≤ `max_gpus`(7)；无 GPU → 停止 pipeline |
| environment_setup | `validate_docker_image.py` | 镜像内 `import torch/swift/transformers` 成功；镜像名符合命名规范 |
| swift_support_check | `check_swift_support.py`（静态解析，确定性） | 产出 `swift_support_report.json`，`supported` 字段决定路径分叉 |

**Custom Path**：

| 阶段 | 验证方式 | 通过标准（关键项） |
|---|---|---|
| model_analysis | 自检查 JSON schema（必填字段齐全） | 含 components 列表、forward_signature、modalities、11 个必答问题结论；音频输出部件标记 ignore |
| user_decision | **用户确认**（唯一的人工 gate） | `user_decision.json` 中 `confirmed: true` + `download_plan_confirmed: true` |
| download_weights | 文件存在性检查（`verify_weights.py` / `batch_download.py` 报告） | 所有 component 权重 `exists: true`，`missing: []` |
| dataset_register | `py_compile` + `validate_dataset_registration.py` | 数据集可加载、字段齐备、音频可访问、缺失音频数 0 |
| model_register | `py_compile` + `validate_model_registration.py` | 模型可加载、**可训练参数 > 0**（过度冻结会被抓出） |
| template_register | `py_compile` + `validate_template_registration.py` | encode keys 匹配 forward 签名、collator 可组 batch、loss mask 正确 |
| integration_test | `run_integration_tests.py`：**6 core validator 顺序执行**（forward_pass → loss_computation → single_step_training → checkpoint_save_load → inference → freeze_unfreeze）**+ model_specific 全部通过** | 全部 `passed: true` |
| smoke_test | `run_smoke_test.py` 编排 + 模型专属推理脚本 | **三轮全过**（见下方详述） |
| training_script | `bash -n` + `validate_training_script.py` | 必要参数齐全、GPU ≤ 7 |
| full_training | 训练日志指标 | loss 有限非 NaN、token_acc > 0、checkpoint 文件存在 |
| evaluation | 指标 sanity check | 测试集全样本覆盖；CER/WER 已计算；指标不低于 smoke test 预期量级（如 CER > 50% 触发排查而非放行） |

**Registered Path**（差异项）：

| 阶段 | 验证方式 | 通过标准 |
|---|---|---|
| registered_model_info | 自检查 JSON | model_type/model_path 确认、权重本地存在、特殊参数记录 |
| registered_model_register（可选） | `py_compile` + 生成器内置规则 | 注册脚本可编译；词表对齐逻辑正确；`model_arch` 已注册 |
| registered_dataset_register | 方案 A/B 免代码；方案 C 用 `validate_dataset_registration.py` | 同 Custom Path 数据集标准 |
| registered_integration_test | `run_registered_integration_tests.py`（load_model / load_dataset / forward_pass / single_step_training） | 全部 `passed: true` |
| registered_smoke_test | `run_smoke_test.py --mode registered`（推理用 `swift infer`/`PtEngine`） | 三轮全过，标准同 Custom Path |
| registered_training_script | `bash -n` + `validate_training_script.py` | 同 Custom Path；组件化时 `--external_plugins` 不可缺 |
| registered_full_training / registered_evaluation | 同 Custom Path | 同 Custom Path |

### 3.5 Smoke Test：正确性最终守门员（重点）

Smoke test 是**训练脚本生成前的强制闸门**（`smoke_test_report.json` 中 `passed: true` 才允许进入 training_script 阶段）。三轮递进验证"能学会、能推理、不 OOM"：

| 测试 | 数据 | 参考训练参数 | 通过标准 |
|---|---|---|---|
| **Test 1：单条 overfit** | 1 条音频复制 100 遍（`overfit1.jsonl`） | 单卡；epoch=2；lr=1e-4；constant scheduler；warmup=0 | 训练 `token_acc > 90%`；**对该条音频推理输出必须与 ground truth 完全一致**（exact match） |
| **Test 2：100 条 mini 训练** | 100 条音频（`mini100.jsonl`） | 7 卡；epoch=20；lr=1e-4；bs=8×grad_accum 4 | `token_acc > 65%`；100 条推理**无乱码、无重复、无不停止**（允许部分样本错，但整体可读） |
| **Test 3：batch size 上限** | 统一的 30s 音频数据集 `test_audio_30s_x100` | 从 bs=8 起训练，OOM 则依次降 8→6→4→2→1 | 记录最大可行 batch size，供正式训练脚本采用 |

**参数弹性原则**：手册明确 smoke test 训练参数只是**参考默认值**——不同模型 projector 容量/初始化差异大，未达标时 Agent 可自行增大 epoch（2→10、20→100）、提高 lr（1e-4→1e-3）或调整 batch；**最终判定以推理验证结果为准**（overfit1 必须完全正确、mini100 必须可读且 token_acc 达标）。若训练日志无 `token_acc`（部分自定义模型），以推理指标为通过依据。

**失败回退路径**：overfit1 失败 → 回 template/model register（重点查 label shift、loss mask、special tokens）；mini100 失败 → 回 model register/training script（查冻结策略、lr、epoch）；bs 全 OOM → 回 model register（查 gradient checkpointing、max_length、设备放置）。

### 3.6 Evaluation 验证（收尾阶段）

```text
checkpoint ──► [1. 模型专属推理] predictions/{dataset}.jsonl（每行 {labels, response}）
            ──► [2. 文本归一化]  中文 evaluation/aispeech_norm / 英文 evaluation/whisper_norm
            ──► [3. 指标计算]    evaluation/wenet_compute_cer.py（tochar=True→CER / False→WER）
            ──► evaluation_report.json + pipeline_state.json 更新
```

一键封装：`tools/postprocess_predictions.py --pred-dir ... --clean-dir ... --datasets {name}.jsonl:zh`，产出 `predictions_clean/{name}.metrics.json` 与 `summary.json`。
**纪律**：推理 prompt 必须与训练一致（防指标虚高/虚低）；指标异常先排查推理脚本与 checkpoint，不直接放行；禁止为评估另写"通用推理脚本"绕过模型专属脚本。

---

## 四、最终结果产物

### 4.1 产物总清单（`outputs/{run_id}/`）

| 产物 | 阶段 | 说明 |
|---|---|---|
| `pipeline_state.json` | 全程 | **核心产物**：完整运行状态（schema 见 `schemas/pipeline_state.schema.json`），可恢复、可审计 |
| `hardware.json` | ① | GPU 探测结果 |
| `docker_resolution.json` / `docker_build_report.json` | ② | 镜像解析动作（use_existing/build_existing/build_new）与构建结果 |
| `SURE_train/Dockerfile/<model>_dockerfile/Dockerfile` | ② | 环境 Dockerfile（新建时） |
| `swift_support_report.json` | ③ | 原生支持判定与 model_type |
| `model_analysis.json` | ④ Custom | 模型组件分析报告（组件、来源、推荐策略、forward 签名、integration warnings） |
| `user_decision.json` / `download_plan.json` | ⑤ Custom | 用户确认的组件策略与下载清单 |
| `download_report.json` / `verify_weights_report.json` | ⑥ Custom | 权重获取结果与本地校验 |
| `custom/{model}_swift_register.py` | ⑦⑧⑨ Custom | **核心代码产物**：dataset + model + template 三注册合一文件 |
| `registered_model_info.json` | ④ Registered | 模型信息与组件化初始化配置 |
| `custom/{family}_registered_model_register.py` | ⑤ Registered 可选 | 组件化初始化的运行时模型注册脚本 |
| `custom/{model}_dataset_register.py` / `dataset_info.json` | ⑥ Registered | 数据集注册代码（方案 C）或字段映射（方案 B） |
| `component_paths.json` / `component_trainable.json` / `freeze_args_report.json` | ⑤⑨ Registered | 组件来源、训练策略及生成的 CLI 冻结参数 |
| `validation_dataset.json` / `validation_model.json` / `validation_template.json` / `validation_docker.json` / `validation_training_script.json` | 各阶段 | run_validator.py 结构化报告 |
| `integration_test_report.json` | ⑩/⑦ | 集成测试逐项结果 + loss/logits 等细节 |
| `smoke_test/` 与 `smoke_test_report.json` | ⑪/⑧ | overfit1/mini100/bs_test 数据集、训练输出、推理脚本与三轮结果 |
| `run_{model}_{dataset}.sh` | ⑫/⑨ | **可直接执行的正式训练脚本**（≤7 卡） |
| `training/train.log` + `train.pid` | ⑬/⑩ | 正式训练日志与进程 PID |
| `checkpoint-xxx/` | ⑬/⑩ | 训练好的模型权重（`--save_only_model true` 保证完整可加载） |
| `predictions/{dataset}.jsonl` | ⑭/⑪ | 测试集原始预测（`labels` + `response`） |
| `predictions_clean/` | ⑭/⑪ | 归一化后 ref/hyp 文本 + `{name}.metrics.json` + `summary.json` |
| `evaluation_report.json` | ⑭/⑪ | **最终评估报告**：checkpoint 路径、各数据集语言/指标/样本数/CER 或 WER 百分比 |

### 4.2 `pipeline_state.json` 结构（核心状态产物）

draft-07 强约束 schema，顶层含：

- **运行元数据**：`run_id`、`created_at`、`updated_at`、`overall_status`（pending/in_progress/completed/failed）、`error_summary`；
- **输入快照**：完整 `input.json` 内容；
- **执行路径**：`path`（custom / registered）；
- **环境摘要**：docker_image、dockerfile、import_test_passed；
- **stages 数组**：每阶段记录 `id`、`status`（pending/in_progress/completed/failed/skipped）、`retry_count`、`started_at`、`completed_at`、`validation`（**内嵌 validator 完整报告**）、`code_path`（指向产出的代码文件，实现结果→源码可追溯）、`code_section`、`log_path`、`error`（{type, message, traceback}）。

真实片段（MiMo-Audio custom run `20260627-160445`）：

```json
{
  "id": "dataset_register", "status": "completed",
  "code_path": "custom/mimo_audio_swift_register.py",
  "validation": {
    "passed": true, "dataset_name": "combined_asr_aishell_1",
    "dataset_size": 134423, "samples_checked": 3,
    "output_keys": ["messages", "audios"], "missing_audio": 0
  }
},
{
  "id": "model_register", "status": "completed",
  "validation": {
    "passed": true, "model_type": "mimo_audio",
    "total_parameters": 8018600000, "trainable_parameters": 741600000,
    "trainable_tensors": 83
  }
},
{
  "id": "integration_test", "status": "completed",
  "validation": {
    "passed": true,
    "checks": ["freeze_unfreeze", "forward_loss", "single_step_training", "inference_forward"],
    "details": {"loss_before": 9.2857, "loss_after": 9.0357, "logits_shape": [1, 53, 151680]}
  }
}
```

### 4.3 真实运行实例

| run_id | 路径 | 模型 | 关键结果 |
|---|---|---|---|
| `outputs/20260627-160445` | Custom | **MiMo-Audio-7B-Base** | 全流程通过：8.02B 总参数 / 0.74B 可训练（83 tensors）；integration loss 9.29→9.04 一步下降；overfit100 smoke loss 8.71→0.005；数据集 134,423 条零缺失音频；进入 full_training |
| `outputs/20260629-183012` | Registered | **Qwen2-Audio-7B**（组件化：LLM←Qwen1.5-7B、audio_tower←whisper-large-v3、aligner 随机） | 训练完成后评估：**AISHELL-1 test CER 6.08%**（7176 条）、**LibriSpeech test-clean WER 11.21%**（2619 条）、**test-other WER 16.43%**（2939 条） |
| `outputs/20260708-203711` ~ `20260709-144622` | Custom | **SLAM-LLM**（family=other/slam_llm） | 已推进至 full_training / training_script 阶段，验证 `other` 族动态判定能力 |

评估产物实例（`predictions_clean/summary.json`）：

```json
{
  "aishell1-test_ASR_infer": {"language": "zh", "metric": "cer",  "num_samples": 7176, "cer_percent": 6.0782},
  "librispeech_test-clean_ASR": {"language": "en", "metric": "wer", "num_samples": 2619, "wer_percent": 11.2078},
  "librispeech_test-other_ASR": {"language": "en", "metric": "wer", "num_samples": 2939, "wer_percent": 16.4347}
}
```

### 4.4 产物依赖链（可追溯性）

```text
input.json ─► hardware.json ─► docker image ─► swift_support_report.json
   ─► (Custom) model_analysis.json ─► user_decision.json ─► download_report.json
   ─► custom/{model}_swift_register.py ─► validation_{dataset,model,template}.json
   ─► integration_test_report.json ─► smoke_test_report.json
   ─► run_{model}_{dataset}.sh ─► training/train.log ─► checkpoint-xxx/
   ─► predictions/ ─► predictions_clean/ ─► evaluation_report.json
   ─► 全部汇入 pipeline_state.json
```

---

## 五、论文写作要点提示（可直接引用的事实）

1. **方法学定位**：本 harness 属于 "LLM agent + 确定性工具脚手架" 范式——LLM 负责需要理解力的部分（读论文/源码、写注册代码、debug），确定性脚本负责可验证的部分（环境探测、下载、验证、状态管理），二者通过 **JSON 产物 + exit code 契约** 解耦。
2. **规模量化**：14 阶段（Custom：3 公共 + 11 专属）/ 11 阶段（Registered：3 公共 + 8 专属）pipeline；14 个通用验证器 + 9 个模型专属验证器；48 种已支持 model_type 的组件映射元数据；6 类状态机命令；单阶段 ≤3 次重试预算。
3. **可靠性机制**：顺序强制（前一阶段 completed 才解锁）、强制验证（validator 不过不能 complete）、超时隔离（600s 子进程包装）、状态持久化（draft-07 schema + 内嵌验证报告）、路径切换守卫（防混用）、人在环路（user_decision 强制 gate）。
4. **知识沉淀**：两大高频 bug 被固化为手册条款与专属验证器——(a) labels 双重 shift（token_acc≈0 的第一嫌疑）；(b) 组件化初始化时的词表对齐（`len(tokenizer)` vs `config.vocab_size`、padding 行覆盖特殊 token）。
5. **实证**：MiMo-Audio（Custom Path，8B 模型/0.74B 可训练）与 Qwen2-Audio（Registered Path 组件化初始化）均端到端跑通，后者取得 AISHELL-1 CER 6.08%、LibriSpeech clean/other WER 11.21%/16.43%。
6. **约束条件**：训练脚本硬性限制最多使用 7 张 GPU（`NPROC_PER_NODE ≤ 7`、`CUDA_VISIBLE_DEVICES` 最多 7 个 id，即使机器有 8 卡）；ASR 场景忽略音频生成部件（audio decoder/head 不下载、不训练）。

---

*文档生成依据：`.swift-adapter-agent/` 全部源文件及 `outputs/` 真实运行记录，逐文件核对。*
