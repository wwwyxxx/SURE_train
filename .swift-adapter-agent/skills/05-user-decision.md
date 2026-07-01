# 05 - User Decision

## 目标

把 model_analysis 发现的组件展示给用户，询问：

1. 每个组件的 train/freeze 策略
2. 每个组件的权重初始化来源
3. 确认下载清单

## Agent Checklist

- [ ] 读取 `outputs/{run_id}/model_analysis.json`
- [ ] 提取 `components` 列表
- [ ] 生成 `outputs/{run_id}/download_plan.json` 初稿
- [ ] 用清晰格式向用户展示：
  - 每个组件名称、作用、推荐策略、初始化来源
  - 需要下载的权重列表
- [ ] 询问用户：
  - 是否接受推荐训练/冻结策略？
  - 是否需要修改某些组件的策略？
  - 下载清单是否正确？有没有本地已有路径需要修改？
  - 是否有找不到的模型，需要用户提供下载链接？
- [ ] 记录用户决策到 `outputs/{run_id}/user_decision.json`
- [ ] 更新下载计划 `outputs/{run_id}/download_plan.json`
- [ ] 更新 `pipeline_state.json`

## 下载计划 JSON 格式

```json
{
  "downloads": [
    {
      "component": "shared_llm",
      "model_id": "qwen/Qwen2.5-7B",
      "local_dir": "/workspace/model/Qwen2.5-7B",
      "source": "modelscope"
    },
    {
      "component": "whisper_encoder",
      "model_id": "openai/whisper-large-v3",
      "local_dir": "/workspace/model/whisper-large-v3",
      "source": "modelscope"
    },
    {
      "component": "vq_adaptor",
      "model_id": null,
      "local_dir": null,
      "source": "random"
    }
  ]
}
```

## 询问格式示例

```text
根据模型分析，Kimi-Audio 包含以下组件：

1. shared_llm（基础 LLM）
   - 模块：model.embed_tokens, model.layers, model.norm
   - 初始化来源：Qwen2.5-7B
   - 推荐策略：freeze
   - 下载：qwen/Qwen2.5-7B -> /workspace/model/Qwen2.5-7B

2. whisper_encoder（音频编码器）
   - 模块：whisper_model
   - 初始化来源：whisper-large-v3
   - 推荐策略：freeze
   - 下载：openai/whisper-large-v3 -> /workspace/model/whisper-large-v3

3. vq_adaptor（音频适配器）
   - 模块：model.vq_adaptor
   - 初始化来源：random
   - 推荐策略：train
   - 无需下载

...

需要下载的权重：
- qwen/Qwen2.5-7B -> /workspace/model/Qwen2.5-7B
- openai/whisper-large-v3 -> /workspace/model/whisper-large-v3

请确认：
1. 是否接受以上 train/freeze 推荐策略？
2. 下载清单是否正确？本地是否已有这些路径？
3. 如果有找不到的模型，请提供下载链接。
```

## 用户决策 JSON 格式

```json
{
  "confirmed": true,
  "components": [
    {
      "name": "shared_llm",
      "action": "freeze",
      "init_source": "/workspace/model/Qwen2.5-7B"
    },
    {
      "name": "whisper_encoder",
      "action": "freeze",
      "init_source": "/workspace/model/whisper-large-v3"
    },
    {
      "name": "vq_adaptor",
      "action": "train",
      "init_source": "random"
    }
  ],
  "download_plan_confirmed": true,
  "user_provided_links": {},
  "notes": "接受推荐策略"
}
```

## 处理用户提供的下载链接

如果某个权重在 modelscope/huggingface 找不到，用户可能会提供链接：

```json
{
  "user_provided_links": {
    "whisper_encoder": "https://huggingface.co/openai/whisper-large-v3"
  }
}
```

更新 `download_plan.json` 后，再进入 download_weights 阶段。

## 特殊说明

对于语音理解 / ASR 任务：

- **不训练 LLM**：保持预训练语言能力
- **冻结 audio encoder**：保持音频特征提取能力
- **训练 adaptor**：让音频特征对齐 LLM
- **训练文本输出头**：让模型输出文本
- **忽略音频输出部件**：如 audio decoder、Patch Decoder 等

如果用户没有特殊要求，默认接受推荐策略即可。

## 失败处理

- 用户对推荐策略有疑问：解释每个决策的原因
- 用户要求修改：调整对应组件，重新生成 user_decision.json 和 download_plan.json
- 用户说某些权重本地已有：更新 download_plan.json 中的 local_dir，跳过下载

## 进入下一阶段

user_decision 和 download_plan 都确认后，进入 `06-download-weights` 阶段。
