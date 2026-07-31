# 11 - Smoke Test

## 目标

在 integration test 通过之后、生成正式训练脚本之前，做三轮 smoke test，验证模型真的能学会、能推理、且 batch size 不会 OOM。

Smoke test 是**模型适配正确性的最终守门员**。如果 smoke test 失败，不能进入 `12-training-script` 阶段。

## 重要说明：训练参数只是参考值

本 skill 中给出的训练参数（epoch、learning_rate、batch_size 等）**只是参考默认值，不是硬性要求**。不同模型的 projector/adapter 容量、初始化方式、数据难度差异很大，参考值可能对具体模型不适用。

如果你发现参考值训练后未达到通过标准（如 token_acc 偏低、推理不正确），**可以根据具体模型自行调整参数**并重新训练，常见调整方向：

- 增大 epoch（如 overfit1: 2 → 10；mini100: 20 → 100）
- 提高 learning_rate（如 1e-4 → 1e-3）
- 调整 batch_size / gradient_accumulation_steps
- 若训练日志中没有 `token_acc` 指标（部分自定义模型），以**推理验证指标**作为通过依据

判断 smoke test 是否通过，以**推理验证结果**为最终标准（overfit1 推理必须完全正确；mini100 推理可读且整体 token_acc 达标）。

## Agent Checklist

- [ ] 确认 `10-integration-test` 已通过
- [ ] 准备 smoke test 数据集：
  - `overfit1.jsonl`：从原始数据集选 1 条音频，复制 100 遍
  - `mini100.jsonl`：从原始数据集选 100 条音频
- [ ] 运行 **Test 1: 单条 overfit**
  - 参考参数：epoch=2, lr=1e-4, lr_scheduler_type=constant, warmup_ratio=0（可根据模型调整）
  - 训练后 `token_acc > 90%`
  - 对该单条音频做推理，输出必须完全正确
- [ ] 运行 **Test 2: 100 条 mini 训练**
  - 参考参数：epoch=20, lr=1e-4, lr_scheduler_type=constant, warmup_ratio=0（可根据模型调整）
  - 训练后 `token_acc > 65%`
  - 对这 100 条音频做推理，输出不能乱码、不能重复、不能不停止
- [ ] 运行 **Test 3: batch size 上限测试**
  - 使用 30s 音频数据集
  - 从 batch_size=8 开始训练
  - 如果 OOM，依次降为 6、4、2、1
  - 记录最终能跑的最大 batch size
- [ ] 把 smoke test 结果写入 `outputs/{run_id}/smoke_test_report.json`
- [ ] 更新 `pipeline_state.json`

## 调用方式

```bash
python .swift-adapter-agent/executors/run_smoke_test.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} \
  --model-type {model_type} \
  --model-family {model_family} \
  --dataset-name {dataset_name} \
  --dataset-path data/combined_asr_aishell-1.jsonl \
  --batch-size-test-dataset test_audio_30s_x100 \
  --output-dir outputs/{run_id}/smoke_test \
  --output outputs/{run_id}/smoke_test_report.json
```

说明：

- `--dataset-name`：源数据集在 ms-swift 中注册的名字（用于训练脚本）
- `--dataset-path`：源 jsonl 文件路径（用于生成 smoke 数据集）
- `--batch-size-test-dataset`：batch size 上限测试用的数据集名字，所有模型复用 `test_audio_30s_x100`
- `--inference-script`：可选，默认根据 `--model-family` 自动选择 harness 内的模型特定推理脚本

## 推理脚本准备（必须由 Agent 自己实现）

在运行 smoke test 之前，agent 必须为当前 `model_family` 编写一个**模型特定的推理脚本**。

### 为什么必须自己写

不同模型的输入格式、tokenizer、生成方式差异很大：

- Kimi-Audio 需要 `whisper_input_feature` 和 `is_continuous_mask`
- MiMo-Audio 需要 3D `input_ids` 和 `text_loss_mask`
- 其他模型可能有完全不同的 forward 签名

因此 harness 无法提供一个通用推理脚本。agent 必须根据 `model_analysis` 和 `09-template-register` 的结果自己写。

### 推理脚本要求

1. 文件路径：
   ```
   .swift-adapter-agent/executors/model_specific/{model_family}/infer_{model_family}.py
   ```
   例如 MiMo-Audio：`.swift-adapter-agent/executors/model_specific/mimo_audio/infer_mimo.py`

2. 必须接受以下参数：
   - `--checkpoint`：训练好的 checkpoint 路径
   - `--dataset`：待推理的 jsonl 路径
   - `--num-samples`：推理样本数
   - `--max-new-tokens`：最大生成 token 数
   - `--output`（可选）：输出 jsonl 路径

3. 必须做的事情：
   - 加载模型和 tokenizer
   - 按该模型的 template 编码输入
   - 调用 `model.generate` 或等价方法
   - 解码并输出文本结果
   - 如果输出为空、乱码、重复或不停止，应明确报错

4. 返回码：
   - `0`：推理成功完成
   - 非 `0`：推理失败

### 示例

MiMo-Audio 的推理脚本参考：
`.swift-adapter-agent/executors/model_specific/mimo_audio/infer_mimo.py`

### run_smoke_test.py 的行为

`run_smoke_test.py` 会根据 `--model-family` 自动查找上述路径：

```python
INFER_SCRIPTS = {
    'mimo_audio': '.swift-adapter-agent/executors/model_specific/mimo_audio/infer_mimo.py',
}
```

如果找不到对应 model_family 的推理脚本，必须传入 `--inference-script` 显式指定，或者 agent 必须先写好推理脚本再进入 smoke test。

## Test 1: 单条 Overfit

### 数据集准备

```bash
python .swift-adapter-agent/executors/prepare_smoke_datasets.py \
  --dataset-path data/combined_asr_aishell-1.jsonl \
  --output-dir outputs/{run_id}/smoke_test \
  --overfit-index 0
```

生成：

- `outputs/{run_id}/smoke_test/overfit1.jsonl`（100 条相同样本）

### 训练

```bash
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --custom_register_path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} \
  --model_type {model_type} \
  --dataset {smoke_overfit1_dataset_name} \
  --train_type full \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lr_scheduler_type constant \
  --warmup_ratio 0 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --logging_steps 1 \
  --save_steps 1000 \
  --save_total_limit 1 \
  --save_only_model true \
  --output_dir outputs/{run_id}/smoke_test/overfit1 \
  --report_to none
```

> 以上为参考参数。若该模型的过拟合收敛慢（如 projector 容量大、需要更多步数），可增大 `--num_train_epochs` 或 `--learning_rate`，直到推理完全正确。

### 通过标准

- 训练 log 最后几行的 `token_acc` 必须 **> 90%**
- loss 迅速收敛（通常从 10+ 降到 1 以下）

### 推理验证

```bash
bash outputs/{run_id}/smoke_test/infer_overfit1.sh
```

其中 `infer_overfit1.sh` 由 `run_smoke_test.py` 根据 `--model-family` 自动生成，调用对应的模型特定推理脚本。

**必须完全正确**，即推理输出 == 该条样本的 ground truth。

## Test 2: 100 条 Mini 训练

### 数据集准备

```bash
python .swift-adapter-agent/executors/prepare_smoke_datasets.py \
  --dataset data/combined_asr_aishell-1.jsonl \
  --output-dir outputs/{run_id}/smoke_test \
  --mini-size 100
```

生成：

- `outputs/{run_id}/smoke_test/mini100.jsonl`

### 训练

```bash
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --custom_register_path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} \
  --model_type {model_type} \
  --dataset {smoke_mini100_dataset_name} \
  --train_type full \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 20 \
  --learning_rate 1e-4 \
  --lr_scheduler_type constant \
  --warmup_ratio 0 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 1 \
  --save_only_model true \
  --output_dir outputs/{run_id}/smoke_test/mini100 \
  --report_to none
```

> 以上为参考参数。若 100 条样本上 token_acc 未达 65%，可增大 `--num_train_epochs`（如 50~100）或 `--learning_rate`（如 1e-3）后重训。

### 通过标准

- 训练 log 最后几行的 `token_acc` 必须 **> 65%**
- loss 明显下降并收敛

### 推理验证

```bash
bash outputs/{run_id}/smoke_test/infer_mini100.sh
```

其中 `infer_mini100.sh` 由 `run_smoke_test.py` 根据 `--model-family` 自动生成。

**不允许出现**：

- 乱码（大量 `<|empty|>`、`<|pad|>` 或其他无意义 token）
- 重复同一个字/词不停止
- 生成超长不停止的序列

允许部分样本预测错误，但整体必须有可读文本输出。

## Test 3: Batch Size 上限测试

### 数据集

所有模型统一使用 30s 音频数据集做 batch size 上限测试：

- 默认：`test_audio_30s_x100`

这个数据集包含 100 条约 30 秒的音频，用于测试 GPU 显存是否能承受不同 batch size。

### 测试流程

从 batch_size=8 开始，依次尝试 8 → 6 → 4 → 2 → 1。

```bash
for BS in 8 6 4 2 1; do
  rm -rf outputs/{run_id}/smoke_test/bs_test_${BS}
  NPROC_PER_NODE=7 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
  swift sft \
    --custom_register_path outputs/{run_id}/custom/{model}_swift_register.py \
    --model {model_path} \
    --model_type {model_type} \
    --dataset test_audio_30s_x100 \
    --train_type full \
    --split_dataset_ratio 0 \
    --per_device_train_batch_size ${BS} \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type constant \
    --warmup_ratio 0 \
    --max_grad_norm 1.0 \
    --bf16 true \
    --gradient_checkpointing true \
    --max_length 2048 \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --save_only_model true \
    --output_dir outputs/{run_id}/smoke_test/bs_test_${BS} \
    --report_to none \
    2>&1 | tee outputs/{run_id}/smoke_test/bs_test_${BS}.log
  
  if ! grep -q "OutOfMemoryError\|CUDA out of memory" outputs/{run_id}/smoke_test/bs_test_${BS}.log; then
    echo "Max workable batch size: ${BS}"
    break
  fi
done
```

### 通过标准

- 记录最终能跑的最大 batch size
- 该 batch size 将用于后续 `12-training-script` 阶段

## 输出报告

`smoke_test_report.json` 示例：

```json
{
  "passed": true,
  "tests": {
    "overfit1": {
      "passed": true,
      "token_acc": 0.96,
      "final_loss": 0.23,
      "inference_exact_match": true,
      "output_dir": "outputs/{run_id}/smoke_test/overfit1"
    },
    "mini100": {
      "passed": true,
      "token_acc": 0.72,
      "final_loss": 1.12,
      "inference_readable": true,
      "output_dir": "outputs/{run_id}/smoke_test/mini100"
    },
    "batch_size": {
      "passed": true,
      "max_batch_size": 8,
      "output_dir": "outputs/{run_id}/smoke_test/bs_test_8"
    }
  }
}
```

## 失败处理

- **overfit1 失败**（token_acc <= 90% 或推理不正确）
  - 回退到 `09-template-register` 或 `08-model-register`
  - 重点检查 labels shift、loss mask、special tokens
- **mini100 失败**（token_acc <= 65% 或推理异常）
  - 回退到 `08-model-register` 或 `12-training-script`
  - 检查冻结策略、学习率、epoch 数
- **batch_size 测试全部 OOM**（连 bs=1 都 OOM）
  - 回退到 `08-model-register`
  - 检查 gradient checkpointing、max_length、模型是否加载到正确设备
- **任何阶段失败**：修复后重新跑对应 smoke test，不要跳过

## 进入下一阶段

只有 smoke_test_report.json 中 `passed: true`，才能进入 `12-training-script` 阶段。
