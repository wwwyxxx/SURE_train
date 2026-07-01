# 05 - Model Register

## 目标

编写 `register_model` 代码，让 ms-swift 能加载目标模型。

## Agent Checklist

- [ ] 读取 `model_analysis.json`
- [ ] 判断是否需要 model wrapper
- [ ] 在 `outputs/{run_id}/custom/{model}_swift_register.py` 中编写 `register_model`
- [ ] 如果需要 wrapper，实现 wrapper class 和 `get_model_tokenizer_xxx`
- [ ] 如果需要权重复制，实现 `_copy_weights`
- [ ] 如果需要冻结策略，实现 `_freeze`
- [ ] 运行语法检查
- [ ] 调用验证：
  ```bash
  python .swift-adapter-agent/executors/run_validator.py \
    --validator .swift-adapter-agent/validators/core/validate_model_registration.py \
    --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
    --model {model_path} \
    --model-type {model_type} \
    --output outputs/{run_id}/validation_model.json
  ```
- [ ] 如果失败，修复代码，重试最多 3 次
- [ ] 更新 `pipeline_state.json`

## 关键决策

| 问题 | 决策依据 |
|------|---------|
| 是否需要 wrapper？ | model_analysis.requires_wrapper |
| 是否复制权重？ | model_analysis.requires_weight_copy |
| 冻结哪些层？ | model_analysis.recommended_freeze_strategy |

## 失败处理

- 模型加载失败：检查 model_path 和 config
- 没有 trainable 参数：检查 freeze 策略是否过度冻结
- 权重复制失败：检查 base model 和目标模型形状是否匹配

## 进入下一阶段

模型注册验证通过后，进入 `09-template-register` 阶段。
