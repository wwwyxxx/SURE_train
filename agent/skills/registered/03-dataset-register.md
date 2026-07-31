# registered/03 - Dataset Register

## 目标

为 ms-swift 已支持的模型注册自定义数据集。

对于已支持模型，数据集注册可以更简单：
- 如果数据已经是 ms-swift 标准格式，**可以直接传 jsonl 路径，无需写代码**
- 如果数据是自定义格式（如 `wav`/`txt`/`prompt`），仍需要写 `register_dataset`

## Agent Checklist

- [ ] 读取数据集前 5 行，识别字段
- [ ] 判断数据是否已经是 ms-swift 标准格式：
  - 标准格式包含 `messages` 和 `audios`/`images`/`videos`
  - 例如：`{"messages": [...], "audios": ["xxx.wav"]}`
- [ ] 如果是标准格式：
  - 记录 `--dataset {dataset_path}` 可直接使用
  - 跳过 Python 注册代码
- [ ] 如果是自定义格式：
  - 在 `outputs/{run_id}/custom/{model}_dataset_register.py` 中写 preprocessor
  - 调用 `register_dataset`
- [ ] 运行语法检查
- [ ] 调用验证：
  ```bash
  python .swift-adapter-agent/executors/run_validator.py \
    --validator .swift-adapter-agent/validators/core/validate_dataset_registration.py \
    --custom-register-path outputs/{run_id}/custom/{model}_dataset_register.py \
    --dataset-name {dataset_name} \
    --output outputs/{run_id}/validation_dataset.json
  ```
- [ ] 如果失败，修复 preprocessor 或数据路径，重试最多 3 次
- [ ] 更新 `pipeline_state.json`

## 方案 A：数据已是标准格式（推荐）

如果你的 jsonl 长这样：

```json
{
  "messages": [
    {"role": "user", "content": "Transcribe the speech to text. <audio>"},
    {"role": "assistant", "content": "最 高涨 幅 为 百分 之 六 点 七"}
  ],
  "audios": ["data/BAC009S0002W0263.wav"]
}
```

那么训练时直接：

```bash
swift sft \
  --model_type qwen2_audio \
  --model /workspace/model/Qwen2-Audio-7B \
  --dataset data/combined_asr.jsonl \
  ...
```

不需要写 `register_dataset`。

## 方案 B：自定义格式，使用 dataset_info.json

如果你的 jsonl 是 `wav`/`txt`/`prompt` 格式，可以写 `dataset_info.json`：

```json
[
  {
    "dataset_name": "combined_asr_custom",
    "dataset_path": "data/combined_asr.jsonl",
    "columns": {
      "wav": "audios",
      "txt": "response",
      "prompt": "query"
    }
  }
]
```

训练时：

```bash
swift sft \
  --model_type qwen2_audio \
  --model /workspace/model/Qwen2-Audio-7B \
  --dataset combined_asr_custom \
  --custom_dataset_info dataset_info.json \
  ...
```

## 方案 C：自定义格式，使用 Python register_dataset

如果 `dataset_info.json` 不够灵活，写 Python 代码：

```python
import os
from typing import Any, Dict, Optional

from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset


class MyASRPreprocessor(ResponsePreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        wav = row.get('wav') or row.get('audio') or row.get('audio_path')
        text = row.get('txt') or row.get('text') or row.get('response')
        prompt = row.get('prompt') or 'Transcribe the speech to text.'
        if wav is None or text is None:
            return None
        if not os.path.isabs(wav):
            wav = os.path.abspath(wav)
        return {
            'messages': [
                {'role': 'user', 'content': f'{prompt} <audio>'},
                {'role': 'assistant', 'content': text},
            ],
            'audios': [wav],
        }


register_dataset(
    DatasetMeta(
        dataset_path='data/combined_asr.jsonl',
        dataset_name='combined_asr_custom',
        preprocess_func=MyASRPreprocessor(),
    ),
    exist_ok=True,
)
```

训练时：

```bash
swift sft \
  --model_type qwen2_audio \
  --model /workspace/model/Qwen2-Audio-7B \
  --dataset combined_asr_custom \
  --custom_register_path outputs/{run_id}/custom/{model}_dataset_register.py \
  ...
```

## 选择建议

| 数据格式 | 推荐方案 | 说明 |
|---------|---------|------|
| 标准 `messages` + `audios` | A：直接传路径 | 最简单 |
| 简单字段映射（wav/txt/prompt） | B：dataset_info.json | 不用写 Python |
| 复杂预处理、过滤、动态 prompt | C：Python register_dataset | 最灵活 |

## 输出

- 如果是方案 A：记录 `--dataset {dataset_path}`
- 如果是方案 B：记录 `--dataset {dataset_name} --custom_dataset_info {path}`
- 如果是方案 C：`outputs/{run_id}/custom/{model}_dataset_register.py`

## 失败处理

- 数据集加载失败：检查 jsonl 格式和路径
- 音频文件找不到：检查路径是绝对路径还是相对路径
- preprocessor 输出格式不对：确保包含 `messages` 和对应模态字段
- `--dataset` 直接传路径但 ms-swift 报错：说明格式不标准，转用方案 B/C

## 进入下一阶段

数据集注册验证通过后，进入 `registered/04-integration-test.md`。
