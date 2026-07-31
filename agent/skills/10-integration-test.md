# 07 - Integration Test

## 目标

在数据集、模型、模板都注册成功后，做端到端集成验证。

## Agent Checklist

- [ ] 确认 dataset_register、model_register、template_register 都 completed
- [ ] 调用 executor：
  ```bash
  python .swift-adapter-agent/executors/run_integration_tests.py \
    --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
    --model {model_path} \
    --model-type {model_type} \
    --model-family {model_family} \
    --dataset-name {dataset_name} \
    --output outputs/{run_id}/integration_test_report.json
  ```
- [ ] 读取 `integration_test_report.json`
- [ ] 如果全部通过，更新 `pipeline_state.json`
- [ ] 如果有失败，根据失败类型回退到对应阶段修复：
  - forward 失败 → 检查 model_register / template_register
  - loss 不对 → 检查 template_register
  - inference 失败 → 检查 model_register / template_register
  - freeze 不对 → 检查 model_register

## 验证顺序

Integration test 由两部分组成：

### 1. Core validators（所有模型都跑）

按顺序执行：

1. `validators/core/validate_forward_pass.py`
2. `validators/core/validate_loss_computation.py`
3. `validators/core/validate_single_step_training.py`
4. `validators/core/validate_checkpoint_save_load.py`
5. `validators/core/validate_inference.py`
6. `validators/core/validate_freeze_unfreeze.py`

Core validators 是模型无关的，只通过 ms-swift 的 `get_model_tokenizer`、`get_template`、`load_dataset` 接口验证基本能力。

### 2. Model-family-specific validators

根据 `--model-family` 动态发现 `validators/model_specific/{model_family}/validate_*.py` 并顺序执行。

例如 Kimi-Audio 会额外跑：

- `validators/model_specific/kimi_audio/validate_loss_mask.py`
- `validators/model_specific/kimi_audio/validate_label_shift.py`
- `validators/model_specific/kimi_audio/validate_sequence_length.py`
- `validators/model_specific/kimi_audio/validate_weight_initialization.py`

MiMo-Audio 会额外跑：

- `validators/model_specific/mimo_audio/validate_mimo_audio_forward.py`
- `validators/model_specific/mimo_audio/validate_mimo_audio_integration.py`
- `validators/model_specific/mimo_audio/validate_mimo_audio_model.py`
- `validators/model_specific/mimo_audio/validate_mimo_audio_template.py`

## 添加新的模型特定验证

如果某个模型需要额外的集成检查：

1. 在 `validators/model_specific/{model_family}/` 下新建 `validate_xxx.py`
2. 脚本必须接受 `--custom-register-path`、`--model`、`--model-type`、`--dataset-name` 参数
3. 返回 0 表示通过，非 0 表示失败
4. 不需要修改 `run_integration_tests.py`，会自动发现

## 失败处理

- 第一次失败后，先回到最可能相关的阶段修复
- 修复后重新跑该阶段验证 + integration test
- 不要只跑 integration test 而不验证修复阶段

## 进入下一阶段

所有 core validators 和 model-specific validators 通过后，进入 `11-smoke-test` 阶段。
