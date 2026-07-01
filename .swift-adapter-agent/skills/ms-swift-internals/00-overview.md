# ms-swift 内部机制总览

## 定位

本系列 skill 记录 ms-swift 框架的内部实现细节。适配新模型时，agent 应优先查阅本 skill，而不是每次都重新阅读源码。

## 源码位置

```text
SURE_train/ms-swift/swift/
├── cli/sft.py              # CLI 入口
├── llm/
│   ├── model/              # 模型注册与加载
│   ├── template/           # 模板注册与编码
│   ├── dataset/            # 数据集注册与加载
│   ├── train/sft.py        # SFT pipeline
│   ├── base.py             # pipeline 基类
│   ├── argument/           # 参数解析
│   └── train/tuner.py      # LoRA/full/adapter 等 tuner
├── trainers/               # Trainer 实现
├── hub/                    # ModelScope/HuggingFace 下载
└── plugin/                 # 扩展插件
```

## 核心映射表

| 映射 | 文件 | 作用 |
|------|------|------|
| `MODEL_MAPPING` | `llm/model/register.py` | model_type → ModelMeta |
| `MODEL_ARCH_MAPPING` | `llm/model/model_arch.py` | arch_name → ModelKeys |
| `TEMPLATE_MAPPING` | `llm/template/register.py` | template_type → TemplateMeta |
| `DATASET_MAPPING` | `llm/dataset/register.py` | dataset id/path → DatasetMeta |
| `TRAINER_MAPPING` | `trainers/trainer_factory.py` | task_type → trainer class |

## 适配三大注册机制

所有模型适配最终都是完成三件事：

1. **register_model**：告诉 ms-swift 如何加载模型
2. **register_template**：告诉 ms-swift 如何把 messages 编码成模型输入
3. **register_dataset**：告诉 ms-swift 如何加载和预处理数据集

本 skill 系列分别深入讲解这三个机制。

## 阅读顺序

### 第一阶段：理解 ms-swift 原理

1. `ms-swift-internals/00-overview.md`
2. `ms-swift-internals/01-register_model.md`
3. `ms-swift-internals/02-register_template.md`
4. `ms-swift-internals/03-register_dataset.md`
5. `ms-swift-internals/04-sft-training-flow.md`
6. `ms-swift-internals/05-extension-hooks.md`
7. `ms-swift-internals/06-data-flow.md`

### 第二阶段：理解本 pipeline 流程

8. `skills/00-overview.md`
9. `skills/01-hardware-probe.md` ~ `skills/09-full-training.md`

### 第三阶段：遇到问题按需查阅

10. `ms-swift-internals/07-common-issues.md` — 通用问题
11. `ms-swift-internals/08-docker-environment.md` — Docker 环境
12. `ms-swift-internals/09-label-mask-debugging.md` — label/mask 调试
13. `skills/model_specific/kimi_audio/training-guide.md` — Kimi-Audio 实战
