# registered/05 - Smoke Test

## 目标

对于 ms-swift 已支持的模型，做三轮 smoke test：
1. 单条 overfit
2. 100 条 mini 训练
3. batch size 上限测试

和未注册模型路径的主要区别：**推理可以使用 `swift infer` 或 `PtEngine`，不需要写模型特定的推理脚本。**

## Agent Checklist

- [ ] 确认 `registered_integration_test` 已通过
- [ ] 准备 smoke test 数据集：
  - `overfit1.jsonl`：从原始数据集选 1 条音频，复制 100 遍
  - `mini100.jsonl`：从原始数据集选 100 条音频
- [ ] 运行 **Test 1: 单条 overfit**
  - epoch=2, lr=1e-4, lr_scheduler_type=constant, warmup_ratio=0
  - 训练后 `token_acc > 90%`
  - 用 `swift infer` 或 `PtEngine` 对该条音频做推理，输出必须完全正确
- [ ] 运行 **Test 2: 100 条 mini 训练**
  - epoch=20, lr=1e-4, lr_scheduler_type=constant, warmup_ratio=0
  - 训练后 `token_acc > 65%`
  - 对这 100 条音频做推理，输出不能乱码、不能重复、不能不停止
- [ ] 运行 **Test 3: batch size 上限测试**
  - 使用 30s 音频数据集
  - 从 batch_size=8 开始训练
  - 如果 OOM，依次降为 6、4、2、1
  - 记录最终能跑的最大 batch size
- [ ] 把 smoke test 结果写入 `outputs/{run_id}/smoke_test_report.json`
- [ ] 更新 `pipeline_state.json`

## 推理方式

### 方式 1：swift infer（推荐）

对于已注册模型，可以直接用：

```bash
swift infer \
  --model_type {model_type} \
  --model {checkpoint_path} \
  --val_dataset {test_jsonl} \
  --max_new_tokens 128 \
  --max_batch_size 1
```

注意：`--val_dataset` 接受的是 ms-swift 能识别的数据集格式。如果是自定义格式，需要同时传 `--custom_register_path`。

### 方式 2：PtEngine Python API

如果 `swift infer` 对音频支持不好，用 Python：

```python
from swift.llm import BaseArguments, InferRequest, PtEngine, get_template

args = BaseArguments.from_pretrained(checkpoint_path)
engine = PtEngine(args.model, adapters=[checkpoint_path])
template = get_template(args.template, engine.processor, args.system, use_chat_template=args.use_chat_template)
engine.default_template = template

infer_request = InferRequest(
    messages=[{'role': 'user', 'content': 'Transcribe the speech to text.'}],
    audios=['xxx.wav']
)
resp_list = engine.infer([infer_request])
print(resp_list[0].choices[0].message.content)
```

### 方式 3：自定义推理脚本

如果 `swift infer` 和 `PtEngine` 都跑不通，才需要写自定义推理脚本。

对于已注册模型，这种情况应该很少见。如果发生，参考 `skills/11-smoke-test.md` 的自定义推理脚本要求。

## 调用方式

```bash
python .swift-adapter-agent/executors/run_smoke_test.py \
  --mode registered \
  --model-type {model_type} \
  --model {model_path} \
  --dataset-name {dataset_name} \
  --dataset-path data/combined_asr.jsonl \
  --batch-size-test-dataset test_audio_30s_x100 \
  --output-dir outputs/{run_id}/smoke_test \
  --output outputs/{run_id}/smoke_test_report.json
```

参数说明：

- `--mode registered`：表示已注册模型，使用 `swift infer` / `PtEngine` 做推理
- `--dataset-name`：训练用的数据集名称
- `--dataset-path`：源 jsonl 文件路径（用于生成 smoke 数据集）
- `--batch-size-test-dataset`：batch size 上限测试数据集名称

## Test 1: 单条 Overfit

### 数据集准备

```bash
python .swift-adapter-agent/executors/prepare_smoke_datasets.py \
  --dataset-path data/combined_asr.jsonl \
  --output-dir outputs/{run_id}/smoke_test \
  --overfit-index 0
```

### 训练

```bash
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
  --model_type {model_type} \
  --model {model_path} \
  --dataset {smoke_overfit1_dataset_name} \
  --train_type lora \
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
  --output_dir outputs/{run_id}/smoke_test/overfit1 \
  --report_to none
```

### 通过标准

- 训练 log 最后几行的 `token_acc` 必须 **> 90%**
- 对该单条音频做推理，输出必须完全正确

## Test 2: 100 条 Mini 训练

### 训练

```bash
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --model_type {model_type} \
  --model {model_path} \
  --dataset {smoke_mini100_dataset_name} \
  --train_type lora \
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
  --output_dir outputs/{run_id}/smoke_test/mini100 \
  --report_to none
```

### 通过标准

- 训练 log 最后几行的 `token_acc` 必须 **> 65%**
- 推理输出不能乱码、不能重复、不能不停止

## Test 3: Batch Size 上限测试

和未注册模型路径相同，参考 `skills/11-smoke-test.md` 的 batch size 测试部分。

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

- **overfit1 失败**：
  - 回退到 `registered/03-dataset-register.md`
  - 检查数据格式、labels、loss mask
- **mini100 失败**：
  - 检查学习率、epoch 数、冻结策略
- **batch_size 全部 OOM**：
  - 检查 gradient checkpointing、max_length
- **`swift infer` 推理失败**：
  - 改用 `PtEngine` Python API
  - 如果还不行，才考虑写自定义推理脚本

## 进入下一阶段

只有 `smoke_test_report.json` 中 `passed: true`，才能进入 `registered/06-training-script.md`。
