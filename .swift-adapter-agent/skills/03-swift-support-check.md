# 03 - Swift Support Check

## 目标

在深入分析模型之前，先判断 **ms-swift 框架是否已经原生支持** 该模型。

这个判断通过读取 `SURE_train/ms-swift/swift/llm/model/constant.py` 和 `SURE_train/ms-swift/examples/` 完成，不依赖论文或模型源码。

## Agent Checklist

- [ ] 调用 executor：
  ```bash
  python .swift-adapter-agent/executors/check_swift_support.py \
    --model-family {model_family} \
    --sure-train-dir SURE_train \
    --output outputs/{run_id}/swift_support_report.json
  ```
- [ ] 读取 `swift_support_report.json`
- [ ] 根据 `supported` 字段决定后续路径：
  - `supported: false` → 走**未注册模型路径**（skills/04 ~ skills/13）
  - `supported: true` → 走**已注册模型路径**（skills/registered/01 ~ skills/registered/07；02 为可选组件组装）
- [ ] 把结果写入 `pipeline_state.json`，并记录 `model_path` / `model_type` 等关键信息

## 输出 JSON 格式

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

## 判断规则

`check_swift_support.py` 会检查 `constant.py` 中的以下类别：

- `LLMModelType`：纯文本模型
- `MLLMModelType`：多模态模型（含音频、视觉模型）
- `BertModelType`、RMModelType、RerankerModelType 等

当前 harness 已知的映射关系：

| harness model_family | ms-swift model_type | 是否原生支持 |
|---------------------|---------------------|------------|
| `qwen2_audio` | `qwen2_audio` | ✅ 支持 |
| `qwen_audio` | `qwen_audio` / `qwen2_audio` | ✅ 支持 |
| `qwen_omni` | `qwen2_5_omni` / `qwen3_omni` | ✅ 支持 |
| `qwen2_5_omni` | `qwen2_5_omni` | ✅ 支持 |
| `qwen3_omni` | `qwen3_omni` | ✅ 支持 |
| `step_audio` | `step_audio` | ✅ 支持 |
| `step_audio2_mini` | `step_audio2_mini` / `step_audio` | ✅ 支持 |
| `kimi_audio` | 无 | ❌ 不支持 |
| `mimo_audio` | 无 | ❌ 不支持 |
| `other` | 视输入而定 | 动态判断 |

## 两条路径的对比

### 路径 A：未注册模型（如 kimi_audio, mimo_audio）

需要：
- `04-model-analysis.md`：读论文/源码，分析组件
- `05-user-decision.md`：询问 train/freeze/init 策略
- `06-download-weights.md`：下载各组件权重
- `07-dataset-register.md`：注册数据集
- `08-model-register.md`：自定义注册模型
- `09-template-register.md`：自定义注册 template
- `10-integration-test.md`：跑完整集成测试
- `11-smoke-test.md`：需要写模型特定的推理脚本
- `12-training-script.md`：使用 `--custom_register_path`

### 路径 B：已注册模型（如 qwen2_audio, qwen_omni, Step-Audio-2-mini）

跳过：
- 读论文源码
- 组件分析
- model register
- template register

需要：
- `registered/01-model-info.md`：确认 model_type、model_path、特殊参数
- `registered/02-checkpoint-assembly.md`（可选）：按组件组装初始化权重
- `registered/03-dataset-register.md`：注册自定义数据集（如果是标准格式可更简单）
- `registered/04-integration-test.md`：验证 model + dataset 能加载
- `registered/05-smoke-test.md`：使用 `swift infer` 或 `PtEngine` 做推理
- `registered/06-training-script.md`：直接用 `--model_type` / `--model`，不需要 `--custom_register_path`
- `registered/07-full-training.md`：启动训练

## 失败处理

- `constant.py` 找不到：检查 `sure-train-dir` 是否正确
- 输出 `supported: false` 但你认为是支持的：
  - 检查 `model_family` 拼写
  - 检查 ms-swift 版本是否过旧
  - 可手动查看 `SURE_train/ms-swift/swift/llm/model/constant.py`
- 判断错误导致进入错误路径：用户可以手动要求 agent 切换路径

## 进入下一阶段

- 如果 `supported: false` → 进入 `04-model-analysis.md`
- 如果 `supported: true` → 进入 `skills/registered/01-model-info.md`
