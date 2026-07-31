# Registered Model Path - 总览

本目录下的 skill 用于 **ms-swift 已原生支持的模型**。

## 什么时候走这条路径？

在 `skills/03-swift-support-check.md` 阶段，如果 executor 输出：

```json
{
  "supported": true,
  "model_type": "qwen2_audio",
  ...
}
```

则走本路径。

## 已支持模型示例

- `qwen2_audio`
- `qwen2_5_omni`
- `qwen3_omni`
- `step_audio`
- `step_audio2_mini`

## 这条路径和未注册模型路径的区别

| 步骤 | 已注册模型路径 | 未注册模型路径 |
|------|--------------|--------------|
| 读论文/源码 | ❌ 不需要 | ✅ 需要 |
| 组件分析 | ❌ 不需要 | ✅ 需要 |
| 用户决策 train/freeze | ❌ 不需要 | ✅ 需要 |
| model register | 仅当使用 `component_paths` 时需要 | ✅ 需要 |
| template register | 仅当使用 `component_paths` 时需要 | ✅ 需要 |
| dataset register | ✅ 仍需要（自定义数据集） | ✅ 需要 |
| 训练脚本 | 使用 `component_paths` 时带 `--external_plugins` | 有 `--custom_register_path` |
| 推理 | `swift infer` / `PtEngine` | 模型特定推理脚本 |

## 本路径阶段

| 阶段 ID | Skill 文件 | 说明 |
|---------|-----------|------|
| `registered_model_info` | `registered/01-model-info.md` | 确认 model_type、model_path、特殊参数 |
| `registered_model_register` | `registered/02-model-register.md` | （可选）生成自定义模型注册脚本以按组件初始化 |
| `registered_dataset_register` | `registered/03-dataset-register.md` | 注册自定义数据集 |
| `registered_integration_test` | `registered/04-integration-test.md` | 验证 model + dataset 能加载 |
| `registered_smoke_test` | `registered/05-smoke-test.md` | 冒烟测试 |
| `registered_training_script` | `registered/06-training-script.md` | 生成训练脚本 |
| `registered_full_training` | `registered/07-full-training.md` | 启动训练 |
| `registered_evaluation` | `registered/08-evaluation.md` | 测试集评估（CER/WER） |

## Agent 通用 Checklist

每个阶段你都要做：

- [ ] 读取对应 skill
- [ ] 调用 `pipeline.py --start-stage {stage_id}`
- [ ] 按 skill 执行操作
- [ ] 调用 validator 或 executor 验证
- [ ] 更新状态：
  - 成功：`pipeline.py --complete-stage {stage_id} --validation-report xxx.json`
  - 失败：`pipeline.py --fail-stage {stage_id} --error-file xxx.json`
- [ ] 如果失败，按 skill 修复，重试最多 3 次
- [ ] 成功后进入下一阶段
