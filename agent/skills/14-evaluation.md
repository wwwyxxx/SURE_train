# 14 - Evaluation

## 目标

在正式训练（`13-full-training`）完成后，在标准测试集上对训练好的 checkpoint 做完整推理，计算 **CER（中文）/ WER（英文）** 指标，产出可对比的评估报告。

这是整个 pipeline 的收尾阶段：只有评估指标达标，适配工作才算真正完成。

## 总流程

```text
checkpoint -> [1. 模型推理] -> predictions/{dataset}.jsonl
           -> [2. 规范化 normalize] -> ref/hyp 文本
           -> [3. 指标计算] -> metrics + evaluation_report.json
```

## 1. 推理：用模型自己的推理脚本生成预测

**推理必须使用每个模型自己的推理脚本**，即 smoke test 阶段使用/编写的脚本：

- custom path：`.swift-adapter-agent/executors/model_specific/{model_family}/infer_{model_family}.py`
- 不同模型的输入格式、tokenizer、generate 约定差异很大，禁止写一个新的通用推理脚本绕过

### 预测文件格式约定（必须遵守）

推理输出统一为 jsonl，每行至少包含以下两个字段：

```json
{"labels": "ground truth 文本", "response": "模型预测文本", ...}
```

- `labels`：测试集的 ground truth
- `response`：模型生成的预测文本

其他字段（audio 路径、logprobs 等）可以保留。这样后续的 normalize + score 流程可以直接复用，无需为每个模型单独适配。

输出文件放到：

```text
outputs/{run_id}/predictions/{test_dataset_name}.jsonl
```

### 注意事项

- 推理时的 prompt 应与训练时使用的 prompt 一致，避免 prompt 不一致导致的指标虚高/虚低
- 长测试集建议推理脚本支持 `--resume`（增量写入），防止中断后从头重跑
- 指标出来前先抽样看几条 prediction，确认没有空输出、乱码、prompt 泄漏

## 2. 规范化（normalize）

CER/WER 计算前必须先对文本做规范化，否则大小写、全角半角、标点、数字写法等差异会污染指标：

- **中文**：使用 `evaluation/aispeech_norm`
  - 接口：`aispeech_norm.LANG_CLASSES['zh']`，先调用一次 `config()`，之后用 `pipeline(text)` 规范化
- **英文**：使用 `evaluation/whisper_norm`
  - 接口：`whisper_norm.EnglishTextNormalizer()()`（实例直接 `__call__(text)`）

## 3. 指标计算（CER / WER）

使用 `evaluation/wenet_compute_cer.py`：

- 接口：`compute_wer(ref_file, hyp_file, tochar)`
  - `tochar=True` → 按字符计算 **CER**（中文用）
  - `tochar=False` → 按词计算 **WER**（英文用）
- 输入文件格式：每行 `key text`（key 为 utt id，与 normalize 后的文本用空格分隔）
- 返回值包含 `cor/sub/del/ins/all`，最终 `CER = (sub+del+ins)/all`

## 4. 一键执行（推荐）

`tools/postprocess_predictions.py` 已经把 2 + 3 封装好了，推理产出 predictions jsonl 后直接调用：

```bash
python3 tools/postprocess_predictions.py \
  --pred-dir outputs/{run_id}/predictions \
  --clean-dir outputs/{run_id}/predictions_clean \
  --eval-dir evaluation \
  --datasets {test_dataset_name}.jsonl:zh
```

参数说明：

- `--datasets`：格式为 `文件名:语言`，语言 `zh` 走 aispeech_norm + CER，`en` 走 whisper_norm + WER；可同时传多个
- 产出：
  - `predictions_clean/{name}.metrics.json`：该数据集的 CER/WER 等指标
  - `predictions_clean/summary.json`：所有数据集的汇总

## 5. 评估报告

把评估结果写入 `outputs/{run_id}/evaluation_report.json`：

```json
{
  "passed": true,
  "checkpoint": "outputs/{run_id}/checkpoint-xxx",
  "metrics": {
    "aishell1-test_ASR_infer": {
      "language": "zh",
      "metric": "cer",
      "num_samples": 7176,
      "cer_percent": 4.05
    }
  },
  "notes": []
}
```

并更新 `pipeline_state.json` 中 `evaluation` 阶段的状态和 validation。

## Agent Checklist

- [ ] 确认 `13-full-training` 已完成，checkpoint 已保存
- [ ] 用模型自己的推理脚本在测试集上推理，产出 `predictions/{dataset}.jsonl`（含 `labels` + `response`）
- [ ] 抽样人工检查 prediction 质量（无空输出、乱码、prompt 泄漏）
- [ ] 按语言调用 normalize（zh: `evaluation/aispeech_norm`；en: `evaluation/whisper_norm`）
- [ ] 用 `evaluation/wenet_compute_cer.py` 计算 CER（zh, tochar=True）/ WER（en, tochar=False）
- [ ] 写 `outputs/{run_id}/evaluation_report.json`
- [ ] 更新 `pipeline_state.json`

## 通过标准

- 推理覆盖测试集全部样本（num_samples 与测试集行数一致）
- CER/WER 指标已计算并写入报告
- 指标不低于 smoke test 时期的预期量级；如果明显异常（如 CER > 50%），先排查推理脚本和 checkpoint，而不是直接通过

## 失败处理

- **prediction 为空/乱码**：回退到 `09-template-register` 或检查推理脚本的 generate 约定（是否包含 prompt、special token 处理）
- **CER/WER 异常高**：先对比 smoke test 推理效果；检查 prompt 是否与训练一致、checkpoint 是否加载正确
- **normalize 报错**：检查文本编码；norm 模块需要 `python3` 环境下直接可用，无额外依赖

## 进入下一阶段

`evaluation` 是 pipeline 的最后一个阶段。报告完成后整个适配流程结束。
