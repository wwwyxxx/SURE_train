# registered/06 - Training Script

## 目标

为 ms-swift 已支持的模型生成训练脚本 `run_{model}_{dataset}.sh`。

和未注册模型路径的关键区别：**不需要 `--custom_register_path`**，因为模型和 template 已经由 ms-swift 原生支持。

## Agent Checklist

- [ ] 读取 `hardware.json` 中的 `usable_gpu_count` 和 `max_gpus`
- [ ] 读取 `registered_model_info.json` 确定 `model_type` 和 `model_path`
  - 如果存在 `assembled_model_path`，优先使用它作为 `--model`
  - 否则使用 `model_path`（单一官方 checkpoint 路径）
- [ ] 如果 `registered_model_info.json` 里有 `component_trainable`：
  - 调用 `generate_freeze_args.py` 生成冻结/训练参数
  - 把返回的 freeze args 填入 `{{freeze_args}}`
- [ ] 读取 `smoke_test_report.json` 确定最大 batch size
- [ ] 选择训练类型：
  - `lora`（推荐用于快速实验和防止灾难性遗忘）
  - `full`（如果数据量大、算力充足）
- [ ] 生成 `outputs/{run_id}/run_{model}_{dataset}.sh`
- [ ] 确保：
  - `NPROC_PER_NODE = min(usable_gpu_count, max_gpus)`，最多 7
  - `CUDA_VISIBLE_DEVICES` 最多 7 个 id
  - 不包含 `--custom_register_path`
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

参考 `templates/training_script_registered.template.sh`。

示例（7 卡，LoRA）：

```bash
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --model_type qwen2_audio \
  --model {{model_path}} \
  --dataset combined_asr_custom \
  --train_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --output_dir output/qwen2_audio_combined_asr \
  --report_to none
```

### 如果数据集是标准格式（方案 A）

```bash
--dataset data/combined_asr.jsonl
```

### 如果数据集用 dataset_info.json 注册（方案 B）

```bash
--dataset combined_asr_custom \
--custom_dataset_info dataset_info.json
```

### 如果数据集用 Python 注册（方案 C）

```bash
--dataset combined_asr_custom \
--custom_register_path outputs/{run_id}/custom/{model}_dataset_register.py
```

注意：这里的 `--custom_register_path` **只包含 dataset register**，不包含 model/template register。

## 关键参数

| 参数 | 说明 |
|------|------|
| `--model_type` | ms-swift 原生 model_type，如 `qwen2_audio` |
| `--model` | 模型路径或 model_id |
| `--dataset` | 数据集名称或路径 |
| `--train_type` | `lora` 或 `full` |
| `--lora_rank` / `--lora_alpha` | LoRA 参数 |
| `--target_modules` | LoRA 目标模块 |
| `--freeze_llm` | 是否冻结 LLM（full 训练时有用） |
| `--per_device_train_batch_size` | 根据 smoke test 结果设置 |
| `--gradient_accumulation_steps` | 根据目标 global batch size 设置 |
| `--max_length` | 根据音频长度设置 |

## 冻结策略

如果 `registered_model_info.json` 中提供了 `component_trainable`，调用：

```bash
python .swift-adapter-agent/executors/generate_freeze_args.py \
  --model-family {model_family} \
  --model-type {model_type} \
  --component-trainable-json outputs/{run_id}/component_trainable.json \
  --output outputs/{run_id}/freeze_args_report.json
```

示例输出（LLM 和 audio encoder 冻结，adaptor 和 text head 训练）：

```bash
--freeze_llm true \
--freeze_vit true \
--freeze_aligner false \
--trainable_parameters language_model.lm_head \
```

直接填入模板中的 `{{freeze_args}}` 即可。

如果没有提供 `component_trainable`，则按默认策略：

- **LoRA 训练**：通常不需要额外 freeze，LoRA 本身只训练 adapter
- **Full 训练**：
  - `--freeze_llm true`：冻结 LLM，只训练 projector/audio encoder
  - `--freeze_vit true`：冻结 audio encoder（如果 ms-swift 把 audio encoder 识别为 vision_tower）
  - `--freeze_aligner false`：训练 aligner

具体参数取决于 ms-swift 对该模型的 `model_arch` 划分。

## 特殊参数

某些已注册模型需要额外参数。例如：

- `qwen2_5_omni`：可能需要 `MAX_PIXELS=1003520`
- 某些模型需要 `--use_chat_template true`
- 某些模型需要 `--task_type seq_cls`（如分类任务）

这些参数在 `registered/01-model-info.md` 阶段已经确认，直接写进训练脚本。

## 失败处理

- 缺少必要参数：对照 ms-swift examples 补充
- 路径不存在：检查 `model_path` 和 dataset 路径
- GPU 数量超过 7：立即修正为 7
- `--custom_register_path` 被错误加入：删除，已注册模型不需要

## 进入下一阶段

训练脚本验证通过后，进入 `registered/07-full-training.md`。
