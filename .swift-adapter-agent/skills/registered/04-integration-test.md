# registered/04 - Integration Test

## 目标

对于 ms-swift 已支持的模型，验证 model + dataset 能被 ms-swift 正确加载，并跑一次前向/单步训练。

因为模型和 template 已经由 ms-swift 原生提供，集成测试比未注册模型路径简单很多。

## Agent Checklist

- [ ] 确认 `registered_model_info.json` 和 dataset register 都已完成
- [ ] 调用 executor：
  ```bash
  python .swift-adapter-agent/executors/run_registered_integration_tests.py \
    --model-type {model_type} \
    --model {model_path} \
    --dataset-name {dataset_name} \
    --custom-register-path outputs/{run_id}/custom/{model}_dataset_register.py \
    --output outputs/{run_id}/integration_test_report.json
  ```
  - 如果数据集是标准格式、没有用 Python 注册，`--custom-register-path` 可以省略
- [ ] 读取 `integration_test_report.json`，检查顶层 `passed` 和 `results` 列表
- [ ] 如果通过，更新 `pipeline_state.json`
- [ ] 如果失败，根据错误类型修复：
  - 模型加载失败 → 检查 `model_path` 和 `model_type`
  - 数据集加载失败 → 回到 `registered/03-dataset-register.md`
  - forward 失败 → 检查数据格式是否与 template 匹配
  - loss 计算失败 → 检查 labels 格式

## 验证内容

对于已注册模型，集成测试至少包括：

1. **模型能加载**：`get_model_tokenizer(model_path, model_type=...)` 成功
2. **Template 能获取**：`get_template(...)` 成功
3. **数据集能加载**：`load_dataset` + preprocessor 成功
4. **Forward 能跑**：一个 batch 能过 `model.forward`
5. **Loss 能计算**：`loss.backward()` 不报错
6. **Single step 训练能跑**：optimizer step 不报错

## 与未注册路径的区别

- 不需要验证 `register_model` 代码
- 不需要验证 `register_template` 代码
- 不需要跑 model-family-specific validators
- 不需要检查 freeze/unfreeze（由 ms-swift 默认处理，或用户在 training_script 阶段指定）

## 输出报告

`integration_test_report.json` 示例：

```json
{
  "passed": true,
  "model_type": "qwen2_audio",
  "model": "/workspace/model/Qwen2-Audio-7B",
  "dataset_name": "combined_asr_custom",
  "results": [
    {
      "name": "load_model",
      "passed": true,
      "message": "Model loaded"
    },
    {
      "name": "load_dataset",
      "passed": true,
      "message": "Dataset loaded: 1200 samples"
    },
    {
      "name": "forward_pass",
      "passed": true,
      "message": "loss=2.3456"
    },
    {
      "name": "single_step_training",
      "passed": true,
      "message": "loss=2.3456"
    }
  ]
}
```

## 失败处理

- **模型加载失败**：
  - 检查 `model_path` 是否存在
  - 检查 `model_type` 是否写对
  - 检查是否需要 `trust_remote_code=True`（ms-swift 默认会处理）
- **数据集加载失败**：
  - 回到 `registered/03-dataset-register.md`
  - 检查 jsonl 路径和格式
- **Forward 失败**：
  - 通常是数据格式与 template 不匹配
  - 检查 `messages` 中 `<audio>` placeholder 是否正确
  - 检查 `audios` 字段是否为列表
- **Loss 为 NaN 或异常大**：
  - 可能是 labels 格式不对
  - 检查 assistant content 是否正确

## 进入下一阶段

集成测试通过后，进入 `registered/05-smoke-test.md`。
