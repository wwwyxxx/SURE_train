# 08 - Training Script

## 目标

生成训练脚本 `run_{model}_{dataset}.sh`，**最多使用 7 张 GPU**。

## Agent Checklist

- [ ] 读取 `hardware.json` 中的 `usable_gpu_count` 和 `max_gpus`
- [ ] 读取 `model_analysis.json` 确定冻结策略
- [ ] 根据显存估算 batch size（可参考 `validate_batch_size_scaling.py`）
- [ ] 生成 `outputs/{run_id}/run_{model}_{dataset}.sh`
- [ ] 确保：
  - `NPROC_PER_NODE = min(usable_gpu_count, max_gpus)`，最多 7
  - `CUDA_VISIBLE_DEVICES` 最多 7 个 id
- [ ] 运行语法检查：`bash -n outputs/{run_id}/run_{model}_{dataset}.sh`
- [ ] 调用验证：
  ```bash
  python .swift-adapter-agent/executors/run_validator.py \
    --validator .swift-adapter-agent/validators/core/validate_training_script.py \
    --script outputs/{run_id}/run_{model}_{dataset}.sh \
    --output outputs/{run_id}/validation_training_script.json
  ```
- [ ] 更新 `pipeline_state.json`

## 脚本模板

参考 `templates/training_script.template.sh`。

示例（7 卡）：

```bash
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --custom_register_path custom/{model}_swift_register.py \
  --model {model_path} \
  --model_type {model_type} \
  --dataset {dataset_name} \
  ...
```

如果机器只有 4 张 GPU：

```bash
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
  ...
```

## 关键参数

- `--custom_register_path`
- `--model`
- `--model_type`
- `--dataset`
- `--freeze_llm`, `--freeze_vit`, `--freeze_aligner`
- `--per_device_train_batch_size`
- `--output_dir`
- `--learning_rate`
- `--lr_scheduler_type`
- `--warmup_ratio`

## Scheduler 选择

模板默认用 `cosine` + `warmup_ratio=0.03`，但以下场景需要调整：

- **小数据 overfit / 快速验证**：用 `constant` + `warmup_ratio=0`
  - 例如 MiMo-Audio 93 条 / 1000 条实验：`lr=1e-4`，`lr_scheduler_type=constant`，`warmup_ratio=0`
- **大数据正式训练**：可以用 `cosine` + 少量 warmup

## 推理验证参数

训练脚本本身不控制推理，但生成推理脚本时需要关注：

- **ASR 任务**：`max_new_tokens=128` 通常足够
- **短音频 ASR**：`max_new_tokens=64` 也可以
- **翻译 / 更长文本任务**：根据目标长度调整

建议在训练完成后，立即用一个独立脚本解码 20–100 条样本验证效果。

## 失败处理

- 缺少必要参数：对照模板补充
- 路径不存在：检查 outputs/{run_id}/custom/ 下文件
- GPU 数量超过 7：立即修正为 7

## 进入下一阶段

训练脚本验证通过后，进入 `13-full-training` 阶段。
