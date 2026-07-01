# ms-swift 常见问题与调试

## 1. 模型注册找不到

**现象**：`Model type xxx not found`

**原因**：

- `--custom_register_path` 没传或路径错误
- `register_model` 没被调用
- `model_type` 写错

**调试**：

```python
from swift.llm.model.register import MODEL_MAPPING
print(MODEL_MAPPING.keys())
```

## 2. Template 找不到

**现象**：`Template xxx not found`

**原因**：

- `register_template` 没被调用
- `ModelMeta.template` 和 `TemplateMeta.template_type` 不匹配

**调试**：

```python
from swift.llm.template.register import TEMPLATE_MAPPING
print(TEMPLATE_MAPPING.keys())
```

## 3. Dataset 找不到

**现象**：`Dataset xxx not found`

**原因**：

- `register_dataset` 没被调用
- `dataset_path` 不存在
- `--dataset` 参数写错

**调试**：

```python
from swift.llm.dataset.register import DATASET_MAPPING
print(list(DATASET_MAPPING.keys())[:20])
```

## 4. labels shift 两次

**现象**：训练 loss 下降但预测错位

**原因**：

- template 里 shift 了 labels
- model forward 里又 shift 了一次

**解决**：

- 只在其中一个地方 shift
- 跑 `validators/model_specific/kimi_audio/validate_label_shift.py`

## 5. padding 位置参与 loss

**现象**：loss 异常高，token_acc 异常

**原因**：

- padding 位置没有被设为 -100
- 或 loss_mask 没排除 padding

**解决**：

- data_collator 里把 padding 位置 labels 设为 -100
- 跑 `validators/core/validate_loss_computation.py`

## 6. freeze 不生效

**现象**：`--freeze_llm true` 但 LLM 仍在训练

**原因**：

- `register_model_arch` 没注册对 `language_model` 路径
- 或自定义 freeze 逻辑覆盖了 ms-swift 的 freeze

**解决**：

- 检查 `MODEL_ARCH_MAPPING`
- 跑 `validators/core/validate_freeze_unfreeze.py`

## 7. OOM

**现象**：`torch.OutOfMemoryError`

**原因**：

- batch size 太大
- 序列太长
- gradient checkpointing 没开

**解决**：

- 减小 `per_device_train_batch_size`
- 减小 `max_length`
- 开 `--gradient_checkpointing true`
- 跑 `validators/core/validate_batch_size_scaling.py`

## 8. Multi-GPU 设备 hardcode

**现象**：多卡训练报错 `Expected all tensors to be on the same device`

**原因**：

- 代码里写了 `torch.cuda.current_device()` 或固定 `cuda:0`

**解决**：

- 用 `input_ids.device` 或 `next(model.parameters()).device`
- 不要 hardcode device id

## 9. 自定义注册文件没导入

**现象**：`MODEL_MAPPING` 里没有自定义 model_type

**原因**：

- `--custom_register_path` 传入后 ms-swift 会 import，但文件可能有语法错误

**解决**：

```bash
python3 -m py_compile custom/xxx_swift_register.py
```

## 10. 权重初始化错误

**现象**：loss 不下降或输出乱码

**原因**：

- 从 base LLM 复制权重时形状不匹配
- 或某些模块没初始化

**解决**：

- 检查 `_copy_weights` 逻辑
- 跑 `validators/model_specific/kimi_audio/validate_weight_initialization.py`
