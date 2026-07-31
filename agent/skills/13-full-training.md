# 09 - Full Training

## 目标

启动训练并监控过程。

## Agent Checklist

- [ ] 调用 executor：
  ```bash
  bash .swift-adapter-agent/executors/start_training.sh \
    outputs/{run_id}/run_{model}_{dataset}.sh \
    outputs/{run_id}/training
  ```
- [ ] 监控训练日志：`tail -f outputs/{run_id}/training/train.log`
- [ ] 检查关键指标：
  - loss 是否下降
  - token_acc 是否上升
  - 是否有 OOM
  - 是否有 NaN
- [ ] 确认 checkpoint 已保存
- [ ] 更新 `pipeline_state.json`

## 通过标准

- loss 为有限值，不是 NaN
- token_acc > 0
- checkpoint 文件存在

## 失败处理

- OOM：减小 batch size 或 max_length，回到 training_script 阶段
- NaN：降低 learning rate，检查 loss 计算
- 不收敛：检查数据、冻结策略、学习率

## 训练完成后

进入 `evaluation` 阶段：用模型自己的推理脚本在标准测试集上对 checkpoint 做完整推理，并计算 CER（中文）/ WER（英文）指标，详见 `skills/14-evaluation.md`。
