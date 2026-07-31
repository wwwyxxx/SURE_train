# registered/08 - Evaluation

## 目标

在正式训练（`registered/07-full-training`）完成后，在标准测试集上评估训练好的 checkpoint，计算 **CER（中文）/ WER（英文）** 指标。

与 custom path（`skills/14-evaluation.md`）的唯一区别是推理方式：已注册模型优先使用 `swift infer` / `PtEngine`，也可以用 smoke test 阶段验证过的自定义推理脚本。

## 1. 推理：生成预测

### 方式 1：swift infer（推荐）

```bash
swift infer \
  --model_type {model_type} \
  --model {checkpoint_path} \
  --val_dataset {test_jsonl} \
  --max_new_tokens 128 \
  --max_batch_size 1
```

如果训练时用了组件化初始化，需要同时传 `--external-plugins` / `--custom_register_path`，与 `registered/05-smoke-test.md` 的推理方式一致。

### 方式 2：PtEngine / 自定义推理脚本

复用 smoke test 阶段已验证可行的推理方式即可。

### 预测文件格式约定（必须遵守）

无论哪种推理方式，最终都要把结果整理成 jsonl，每行至少包含：

```json
{"labels": "ground truth 文本", "response": "模型预测文本", ...}
```

放到：

```text
outputs/{run_id}/predictions/{test_dataset_name}.jsonl
```

## 2. 规范化（normalize）

- **中文**：`evaluation/aispeech_norm`（`LANG_CLASSES['zh']`，先 `config()` 再 `pipeline(text)`）
- **英文**：`evaluation/whisper_norm`（`EnglishTextNormalizer()()`）

## 3. 指标计算（CER / WER）

`evaluation/wenet_compute_cer.py` 的 `compute_wer(ref_file, hyp_file, tochar)`：

- `tochar=True` → CER（中文）
- `tochar=False` → WER（英文）

## 4. 一键执行（推荐）

```bash
python3 tools/postprocess_predictions.py \
  --pred-dir outputs/{run_id}/predictions \
  --clean-dir outputs/{run_id}/predictions_clean \
  --eval-dir evaluation \
  --datasets {test_dataset_name}.jsonl:zh
```

产出 `predictions_clean/{name}.metrics.json` 和 `predictions_clean/summary.json`。

## 5. 评估报告

写入 `outputs/{run_id}/evaluation_report.json` 并更新 `pipeline_state.json`，格式同 `skills/14-evaluation.md`。

## Agent Checklist

- [ ] 确认 `registered/07-full-training` 已完成，checkpoint 已保存
- [ ] 推理产出 `predictions/{dataset}.jsonl`（含 `labels` + `response`）
- [ ] 抽样人工检查 prediction 质量
- [ ] 按语言 normalize（zh: aispeech_norm；en: whisper_norm）
- [ ] 用 wenet_compute_cer.py 计算 CER（zh）/ WER（en）
- [ ] 写 `outputs/{run_id}/evaluation_report.json`
- [ ] 更新 `pipeline_state.json`

## 通过标准 / 失败处理

同 `skills/14-evaluation.md`。

## 进入下一阶段

`registered_evaluation` 是 registered path 的最后一个阶段。报告完成后整个适配流程结束。
