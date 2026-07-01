# 分模块单独调试指南

本 harness 的每个阶段都是独立的，可以单独调试。

## 调试原则

1. **不需要跑完整 pipeline**：可以只调试失败的阶段。
2. **直接调用 executor/validator**：每个工具脚本都可以独立运行。
3. **手动管理 pipeline 状态**：用 `pipeline.py` 标记某个阶段已完成或失败。

## 初始化一个调试用的 pipeline

```bash
python .swift-adapter-agent/pipeline.py --init --input .swift-adapter-agent/example_input_minimal.json
```

得到 run_id 后，后续调试都基于这个 run_id。

## 各阶段单独调试方法

### 1. Hardware Probe

```bash
python .swift-adapter-agent/executors/hardware_probe.py \
  --max-gpus 7 \
  --output outputs/{run_id}/hardware.json
```

### 2. Environment Setup / Docker

```bash
# 解析应该用哪个 docker image（查询模式，不创建文件）
python .swift-adapter-agent/executors/resolve_docker_image.py \
  --model-family kimi_audio \
  --version v0 \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/docker_resolution.json

# 如果需要新建 Dockerfile，再加 --create-if-missing
python .swift-adapter-agent/executors/resolve_docker_image.py \
  --model-family kimi_audio \
  --version v0 \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/docker_resolution.json \
  --create-if-missing

# 如果需要 build
python .swift-adapter-agent/executors/build_docker.py \
  --dockerfile SURE_train/Dockerfile/kimi-audio_dockerfile/Dockerfile \
  --image-name docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0 \
  --build-context SURE_train \
  --output outputs/{run_id}/docker_build_report.json

# 验证 docker image
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_docker_image.py \
  --image docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0 \
  --output outputs/{run_id}/validation_docker.json
```

### 3. Swift Support Check

```bash
python .swift-adapter-agent/executors/check_swift_support.py \
  --model-family {model_family} \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/swift_support_report.json
```

根据结果选择路径：

- `supported: false` → 继续 Custom Path（skills/04 ~ skills/13）
- `supported: true` → 切换到 Registered Path：
  ```bash
  python .swift-adapter-agent/pipeline.py --switch-path registered --run-id {run_id}
  ```

### 4. Model Analysis

Model analysis 需要 agent 读论文/代码，无法完全自动化。但可以手动写 `model_analysis.json`：

```bash
cat > outputs/{run_id}/model_analysis.json <<'JSON'
{
  "model_name": "Kimi-Audio",
  "model_type": "kimi_audio_text",
  "model_family": "kimi_audio",
  "components": [
    {"name": "shared_llm", "modules": [...], "source": "Qwen2.5-7B", "default_action": "freeze"},
    {"name": "whisper_encoder", "modules": [...], "source": "whisper-large-v3", "default_action": "freeze"},
    {"name": "vq_adaptor", "modules": [...], "source": "random", "default_action": "train"},
    {"name": "text_head", "modules": [...], "source": "copy_from_lm_head", "default_action": "train"}
  ]
}
JSON
```

### 5. User Decision

手动写 `user_decision.json` 和 `download_plan.json`：

```bash
cat > outputs/{run_id}/user_decision.json <<'JSON'
{
  "confirmed": true,
  "components": [
    {"name": "shared_llm", "action": "freeze", "init_source": "/workspace/model/Qwen2.5-7B"},
    {"name": "whisper_encoder", "action": "freeze", "init_source": "/workspace/model/whisper-large-v3"},
    {"name": "vq_adaptor", "action": "train", "init_source": "random"},
    {"name": "text_head", "action": "train", "init_source": "copy_from_lm_head"}
  ],
  "download_plan_confirmed": true
}
JSON

cat > outputs/{run_id}/download_plan.json <<'JSON'
{
  "downloads": [
    {"component": "shared_llm", "model_id": "qwen/Qwen2.5-7B", "local_dir": "/workspace/model/Qwen2.5-7B"},
    {"component": "whisper_encoder", "model_id": "openai/whisper-large-v3", "local_dir": "/workspace/model/whisper-large-v3"}
  ]
}
JSON
```

### 6. Download Weights

```bash
# 查找本地模型
python .swift-adapter-agent/executors/find_local_model.py \
  --model-id qwen/Qwen2.5-7B \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/find_local_qwen.json

# 批量下载（会先检查本地）
python .swift-adapter-agent/executors/batch_download.py \
  --plan outputs/{run_id}/download_plan.json \
  --sure-train-dir SURE_train \
  --output outputs/{run_id}/download_report.json
```

### 7. Dataset Register

假设你已经写好了 `outputs/{run_id}/custom/xxx_swift_register.py` 中的 `register_dataset`：

```bash
# 语法检查
python3 -m py_compile outputs/{run_id}/custom/{model}_swift_register.py

# 验证
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_dataset_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/validation_dataset.json
```

### 8. Model Register

```bash
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_model_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} \
  --model-type {model_type} \
  --output outputs/{run_id}/validation_model.json
```

### 9. Template Register

```bash
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_template_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model-type {model_type} \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/validation_template.json
```

### 10. Integration Test

```bash
python .swift-adapter-agent/executors/run_integration_tests.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} \
  --model-type {model_type} \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/integration_test_report.json
```

### 11. Training Script

```bash
# 语法检查
bash -n outputs/{run_id}/run_{model}_{dataset}.sh

# 验证
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_training_script.py \
  --script outputs/{run_id}/run_{model}_{dataset}.sh \
  --output outputs/{run_id}/validation_training_script.json
```

## 手动管理 pipeline 状态

调试时，你可能需要手动把某个阶段标记为 completed：

```bash
# 标记阶段开始
python .swift-adapter-agent/pipeline.py --start-stage dataset_register --run-id {run_id}

# 标记阶段完成
python .swift-adapter-agent/pipeline.py --complete-stage dataset_register \
  --run-id {run_id} \
  --validation-report outputs/{run_id}/validation_dataset.json

# 标记阶段失败
python .swift-adapter-agent/pipeline.py --fail-stage dataset_register \
  --run-id {run_id} \
  --error-file outputs/{run_id}/error_dataset.json
```

## 查看当前状态

```bash
python .swift-adapter-agent/pipeline.py --status --run-id {run_id}
```

## 典型调试流程

假设 `08-model-register` 验证失败：

```bash
# 1. 查看状态
python .swift-adapter-agent/pipeline.py --status --run-id {run_id}

# 2. 修改代码
vim outputs/{run_id}/custom/{model}_swift_register.py

# 3. 语法检查
python3 -m py_compile outputs/{run_id}/custom/{model}_swift_register.py

# 4. 单独跑 model_register 验证
python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_model_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_swift_register.py \
  --model {model_path} \
  --model-type {model_type} \
  --output outputs/{run_id}/validation_model.json

# 5. 如果通过，手动标记完成
python .swift-adapter-agent/pipeline.py --complete-stage model_register \
  --run-id {run_id} \
  --validation-report outputs/{run_id}/validation_model.json

# 6. 继续下一阶段
```

## 注意事项

1. **不要跳过前置依赖**：虽然可以单独调试某个阶段，但该阶段依赖的前置条件必须满足。例如调试 `09-template-register` 前，`08-model-register` 和 `07-dataset-register` 必须通过。
2. **状态一致性**：手动标记 completed 时，确保验证报告确实是通过的。
3. **代码备份**：修改代码前保留备份，方便回滚。

## Registered Path 调试

如果 `swift_support_check` 返回 `supported: true`，agent 会走 registered path。

### 直接测试 ms-swift 模型加载

```bash
python3 - <<'PY'
from swift.llm import get_model_tokenizer, get_template

model, processor = get_model_tokenizer(
    '/workspace/model/Qwen2-Audio-7B',
    model_type='qwen2_audio'
)
template = get_template('qwen2_audio', processor)
print('Model and template loaded successfully')
PY
```

### 直接测试数据集注册

```bash
python3 -m py_compile outputs/{run_id}/custom/{model}_dataset_register.py

python .swift-adapter-agent/executors/run_validator.py \
  --validator .swift-adapter-agent/validators/core/validate_dataset_registration.py \
  --custom-register-path outputs/{run_id}/custom/{model}_dataset_register.py \
  --dataset-name {dataset_name} \
  --output outputs/{run_id}/validation_dataset.json
```

### 直接测试 swift infer

```bash
swift infer \
  --model_type {model_type} \
  --model {checkpoint_path} \
  --val_dataset {test_jsonl} \
  --max_new_tokens 128 \
  --max_batch_size 1
```
