# registered/05 - Smoke Test

## 目标

对于 ms-swift 已支持的模型，做三轮 smoke test：
1. 单条 overfit
2. 100 条 mini 训练
3. batch size 上限测试

和未注册模型路径的主要区别：**推理可以使用 `swift infer` 或 `PtEngine`，不需要写模型特定的推理脚本。**

## 重要说明：训练参数只是参考值

本 skill 中给出的训练参数（epoch、learning_rate、batch_size 等）**只是参考默认值，不是硬性要求**。不同模型收敛速度差异很大，若参考值下未达到通过标准，**可根据具体模型自行调整参数**（如增大 epoch、提高 learning_rate）并重新训练，详见 `skills/11-smoke-test.md` 开头的说明。

## Agent Checklist

- [ ] 确认 `registered_integration_test` 已通过
- [ ] 准备 smoke test 数据集：
  - `overfit1.jsonl`：从原始数据集选 1 条音频，复制 100 遍
  - `mini100.jsonl`：从原始数据集选 100 条音频
- [ ] 运行 **Test 1: 单条 overfit**
  - 参考参数：epoch=2, lr=1e-4, lr_scheduler_type=constant, warmup_ratio=0（可根据模型调整）
  - 训练后 `token_acc > 90%`
  - 用 `swift infer` 或 `PtEngine` 对该条音频做推理，**输出必须完全正确**
  - 如果没有通过单条overfit测试，你必须要去debug直到能通过单条overfit测试，才能进入Test2
- [ ] 运行 **Test 2: 100 条 mini 训练**
  - 参考参数：epoch=20, lr=1e-4, lr_scheduler_type=constant, warmup_ratio=0（可根据模型调整）
  - 训练后 `token_acc > 65%`
  - 对这 100 条音频做推理，输出不能乱码、不能重复、不能不停止
  - 如果没有通过 100 条 mini 训练测试，你必须要去debug直到能通过该测试，才能进入Test3
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
  --external-plugins outputs/{run_id}/custom/{model_family}_registered_model_register.py \
  --custom-register-path outputs/{run_id}/custom/{model_family}_dataset_register.py \
  --val_dataset {test_jsonl} \
  --max_new_tokens 128 \
  --max_batch_size 1
```

注意：`--val_dataset` 接受的是 ms-swift 能识别的数据集格式。如果是自定义格式，需要同时传 `--custom_register_path`。如果训练时使用了组件化初始化（`registered_model_register_path` 存在），推理也必须传 `--external_plugins`。

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
  --save_only_model true \
  --output_dir outputs/{run_id}/smoke_test/mini100 \
  --report_to none
```

### 保存与加载注意事项

**训练时必须加 `--save_only_model true`**
- ms-swift 默认保存的 checkpoint 只包含 LoRA/训练相关状态，对于 `train_type=full` 且基座经过组件化替换的场景，必须保存完整模型权重以便后续推理。
- 如果不加，训练结束后可能没有可加载的 checkpoint，或者 checkpoint 缺少用于推理的完整状态。

**推理时只能加载训练过的参数**
- 当使用组件化初始化时，推理脚本的基座模型必须重新执行组件替换（从 Qwen2.5-7B / Whisper 等重新初始化 LLM/encoder）。
- 训练保存的 checkpoint 虽然包含全模型，但 **不能整体加载**，否则会覆盖掉正确的基座组件。
- 必须只加载实际训练过的参数，例如：

```python
# 1. 重新构建组件化基座模型
model = AutoModelForCausalLM.from_pretrained(base_path, ...)
replace_llm(model, 'model/Qwen2.5-7B')
replace_encoder(model, 'model/whisper-large-v3')
reinit_or_load_adapter(model)

# 2. 只加载训练过的参数（adapter + lm_head）
ckpt = load_file('checkpoint-xxx/model-xxx.safetensors')
filtered_ckpt = {
    k: v for k, v in ckpt.items()
    if k.startswith('adapter.') or k.startswith('lm_head.')
}
model.load_state_dict(filtered_ckpt, strict=False)
```

常见错误：
- 直接 `model.load_state_dict(ckpt)` 加载全模型 → LLM/encoder 被训练时的快照覆盖，导致推理错误。
- 不加 `--save_only_model true` → 没有完整权重可加载，无法做离线推理验证。
- 词表大小设置错误 → 如果注册脚本把 `vocab_size` 改成了新 LLM 的大小，tokenizer 中 base model 独有的特殊 token 会失效，表现为推理乱码、重复或无法停止。词表处理应参考 `registered/02-model-register.md` 的「词表与 embedding / lm_head」章节。

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
