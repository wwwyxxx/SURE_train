# 06 - Template Register

## 目标

编写 `register_template` 代码，把 `{messages, audios/images}` 编码成模型输入。

## Agent Checklist

- [ ] 读取 `model_analysis.json` 中的 forward_signature
- [ ] 设计 `_encode` 输出 keys
- [ ] 在 `outputs/{run_id}/custom/{model}_swift_register.py` 中编写 template
- [ ] 实现 `data_collator`，正确处理 padding 和 loss mask
- [ ] 运行语法检查
- [ ] 调用验证：
  ```bash
  python .swift-adapter-agent/executors/run_validator.py \
    --validator .swift-adapter-agent/validators/core/validate_template_registration.py \
    --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
    --model-type {model_type} \
    --dataset-name {dataset_name} \
    --output outputs/{run_id}/validation_template.json
  ```
- [ ] 如果失败，修复代码，重试最多 3 次
- [ ] 更新 `pipeline_state.json`

## 关键原则

1. **labels 和 logits 对齐**：如果 model forward 内部 shift，template 不要 shift
   - 这是 Kimi-Audio 和 MiMo-Audio 都踩过的坑。Template 返回未 shift 的 labels，model forward 内部做 causal shift，ms-swift 的 `compute_acc` 也会再 shift 一次。
   - 判断标准：如果 `token_acc` 几乎为 0，首先检查 labels 是否被 shift 了两次。
2. **padding 不参与 loss**：labels padding 设为 -100 或用 loss_mask 排除
3. **batch 处理正确**：data_collator 能处理变长序列
4. **检查 special tokens**：如果模型定义了特殊 token（如 `<|sosp|>`、`<|eosp|>`、`<|im_start|>` 等），确保 tokenizer 已经包含它们。必要时在 template 初始化时动态添加。

## 输出

- `outputs/{run_id}/custom/{model}_swift_register.py` 中的 template 部分

## 失败处理

- encode 输出 keys 不对：对照 model forward 签名修改
- collator 报错：检查 padding 值和维度
- 验证显示 mask 不对：检查 assistant tokens 是否正确标记

## 进入下一阶段

Template 注册验证通过后，进入 `10-integration-test` 阶段。
