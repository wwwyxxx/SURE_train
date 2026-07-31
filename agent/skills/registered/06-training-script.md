# registered/06 - Training Script

## 目标

为 ms-swift 已支持的模型生成训练脚本 `run_{model}_{dataset}.sh`。

如果该 run 使用了组件化初始化（`component_paths` 存在），训练脚本必须加载生成的 custom model-register 脚本（`--external_plugins`），并使用 derived `model_type`（如 `qwen2_5_omni_custom`）。如果数据集也需要 Python 注册，额外通过 `--custom_register_path` 加载 dataset register。

如果未使用 `component_paths`，则沿用原生 `model_type`，不需要 `--external_plugins`。

## Agent Checklist

- [ ] 读取 `hardware.json` 中的 `usable_gpu_count` 和 `max_gpus`
- [ ] 读取 `registered_model_info.json` 确定 `model_type` / `registered_model_type` 和 `base_model_path`
  - 如果 `registered_model_register_path` 存在：
    - `--model_type` 使用 `registered_model_type`（e.g. `qwen2_5_omni_custom`）
    - `--model` 使用 `base_model_path`（官方 checkpoint）
    - 加入 `--external_plugins {registered_model_register_path}`
  - 否则：
    - `--model_type` 使用原生 `model_type`
    - `--model` 使用 `model_path`
- [ ] 如果 `registered_model_info.json` 里有 `component_trainable`：
  - 调用 `generate_freeze_args.py` 生成冻结/训练参数
  - 注意：自定义注册脚本默认冻结所有参数，因此 freeze args 需要包含 `--freeze_parameters_regex '.*'` 和 `--trainable_parameters ...`
  - 把返回的 freeze args 填入 `{{freeze_args}}`
- [ ] 如果数据集通过 Python 注册，填入 `{{custom_register_path}}`
- [ ] 读取 `smoke_test_report.json` 确定最大 batch size
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

参考 `templates/training_script_registered.template.sh`。

带组件化初始化的示例（qwen2_5_omni，3 卡，full）：

```bash
NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=4,5,6 \
MAX_PIXELS=1003520 \
ENABLE_AUDIO_OUTPUT=0 \
swift sft \
  --model_type qwen2_5_omni_custom \
  --model /workspace/model/Qwen2.5-Omni-7B \
  --dataset combined_asr_local \
  --external_plugins outputs/{run_id}/custom/qwen2_5_omni_registered_model_register.py \
  --custom_register_path outputs/{run_id}/custom/qwen2_5_omni_dataset_register.py \
  --train_type full \
  --freeze_parameters_regex '.*' \
  --trainable_parameters thinker.lm_head thinker.audio_tower.proj \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --max_length 1024 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 2 \
  --output_dir outputs/{run_id}/full_training \
  --report_to none
```

## 关键参数

| 参数 | 说明 |
|------|------|
| `--model_type` | 原生 `model_type` 或 derived `{model_type}_custom` |
| `--model` | 官方 checkpoint 路径（`base_model_path`） |
| `--external_plugins` | 自定义 model-register 脚本路径（组件化初始化时必填） |
| `--custom_register_path` | 自定义 dataset-register 脚本路径（数据集需要 Python 注册时必填） |
| `--dataset` | 数据集名称或路径 |
| `--train_type` | `lora` 或 `full` |
| `--freeze_parameters_regex` | 设为 `'.*'` 让注册脚本已冻结的参数保持冻结 |
| `--trainable_parameters` | 需要解冻的参数名，可多次传入 |

## 冻结策略

调用：

```bash
python .swift-adapter-agent/executors/generate_freeze_args.py \
  --model-family {model_family} \
  --model-type {model_type} \
  --component-trainable-json outputs/{run_id}/component_trainable.json \
  --use-model-register \
  --output outputs/{run_id}/freeze_args_report.json
```

示例输出（LLM 和 audio encoder 冻结，aligner 和 text head 训练）：

```bash
--freeze_llm true \
--freeze_vit true \
--freeze_aligner true \
--freeze_parameters_regex '.*' \
--trainable_parameters thinker.audio_tower.proj \
--trainable_parameters thinker.lm_head \
```

直接填入模板中的 `{{freeze_args}}` 即可。

## 特殊参数

某些已注册模型需要额外参数。例如：

- `qwen2_5_omni`：需要 `MAX_PIXELS=1003520`、`ENABLE_AUDIO_OUTPUT=0`
- 某些模型需要 `--use_chat_template true`
- 某些模型需要 `--task_type seq_cls`（如分类任务）

这些参数在 `registered/01-model-info.md` 阶段已经确认，直接写进训练脚本。

## 失败处理

- 缺少必要参数：对照 ms-swift examples 补充
- 路径不存在：检查 `base_model_path` 和 dataset 路径
- GPU 数量超过 7：立即修正为 7
- `--external_plugins` 缺失但存在 `registered_model_register_path`：补回

## 进入下一阶段

训练脚本验证通过后，进入 `registered/07-full-training.md`；训练完成后进入 `registered/08-evaluation.md` 在测试集上计算 CER/WER。
