# 04 - Dataset Register

## 目标

编写 `register_dataset` 代码，让 ms-swift 能加载并预处理数据集。

## Agent Checklist

- [ ] 读取数据集前 5 行，识别字段：`wav/audio`、`txt/text`、`prompt/instruction`
- [ ] 在 `outputs/{run_id}/custom/{model}_swift_register.py` 中编写 preprocessor
- [ ] 添加 `register_dataset` 调用
- [ ] 运行语法检查：`python3 -m py_compile outputs/{run_id}/custom/{model}_swift_register.py`
- [ ] 调用验证：
  ```bash
  python .swift-adapter-agent/executors/run_validator.py \
    --validator .swift-adapter-agent/validators/core/validate_dataset_registration.py \
    --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
    --dataset-name {dataset_name} \
    --output outputs/{run_id}/validation_dataset.json
  ```
- [ ] 如果失败，修复 preprocessor 或数据路径，重试最多 3 次
- [ ] 更新 `pipeline_state.json`

## Preprocessor 模板

```python
class MyDatasetPreprocessor:
    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": row.get("prompt", "")},
                {"role": "assistant", "content": row["txt"]},
            ],
            "audios": [row["wav"]],  # 或 images
        }
```

## 输出

- `outputs/{run_id}/custom/{model}_swift_register.py` 中的 `register_dataset` 部分

## 失败处理

- 数据集加载失败：检查 jsonl 格式和路径
- 音频文件找不到：检查路径是绝对路径还是相对路径
- preprocessor 输出格式不对：确保包含 `messages` 和对应模态字段
- 多模态输入形状复杂（如 MiMo-Audio 的 3D `input_ids`）：参考 `skills/model_specific/{model_family}/training-guide.md`

## 进入下一阶段

数据集注册验证通过后，进入 `08-model-register` 阶段。
