# 03 - Model Analysis

## 目标

深入理解模型结构，列出所有组件，为后续询问用户做准备。

## Agent Checklist

- [ ] 读取模型 `config.json`
- [ ] 如果有源码路径，阅读 `modeling_xxx.py` 中的 `forward` 函数
- [ ] 如果有论文，阅读摘要、方法、模型结构图部分
- [ ] 列出模型所有**组件**（component）
- [ ] 对每个组件，判断：
  - 它的作用是什么？
  - 它是否可以从预训练权重初始化？
  - 它通常应该冻结还是训练？
- [ ] 填写 `outputs/{run_id}/model_analysis.json`
- [ ] 自检查 output JSON 是否包含所有必需字段
- [ ] 更新 `pipeline_state.json`

## 组件定义

一个"组件"是模型中一个可独立决定 train/freeze/init 的模块。

**注意：我们只关注语音理解 / ASR 任务，只输出文本。因此输出音频的部件（如 audio decoder、audio head、Patch Decoder）不计入考虑。**

例如 Kimi-Audio 的组件（ASR 场景）：

```json
[
  {
    "name": "shared_llm",
    "modules": ["model.embed_tokens", "model.layers", "model.norm"],
    "source": "Qwen2.5-7B",
    "default_action": "freeze",
    "description": "基础 LLM，通常冻结以保持语言能力"
  },
  {
    "name": "whisper_encoder",
    "modules": ["whisper_model"],
    "source": "whisper-large-v3",
    "default_action": "freeze",
    "description": "音频编码器，通常冻结以保持音频特征提取能力"
  },
  {
    "name": "vq_adaptor",
    "modules": ["model.vq_adaptor"],
    "source": "random",
    "default_action": "train",
    "description": "音频特征到 LLM 的适配器"
  },
  {
    "name": "mimo_layers",
    "modules": ["model.mimo_layers"],
    "source": "random",
    "default_action": "train",
    "description": "音频解码层（在 ASR 场景中用于辅助音频理解）"
  },
  {
    "name": "mimo_norm",
    "modules": ["model.mimo_norm"],
    "source": "random",
    "default_action": "train",
    "description": "音频解码归一化"
  },
  {
    "name": "text_head",
    "modules": ["mimo_output"],
    "source": "Qwen lm_head partial copy",
    "default_action": "train",
    "description": "文本输出头"
  }
]
```

**应忽略的组件示例**：

- Kimi-Audio 的 audio head（生成音频）
- MiMo-Audio 的 Patch Decoder（生成音频）
- 任何明确用于音频合成的 decoder/head

这些组件在 ASR/语音理解任务中不需要训练，也不需要下载权重。


## 输出 JSON 格式

```json
{
  "model_name": "Qwen/Qwen2.5-7B",
  "model_type": "qwen2",
  "model_family": "kimi_audio",
  "architectures": ["Qwen2ForCausalLM"],
  "base_llm": "Qwen2.5-7B",
  "vocab_size": 152064,
  "hidden_size": 3584,
  "num_layers": 28,
  "modalities": ["audio", "text"],
  "requires_wrapper": true,
  "forward_signature": ["input_ids", "text_input_ids", "is_continuous_mask", "whisper_input_feature", "attention_mask", "labels"],
  "components": [
    {
      "name": "shared_llm",
      "modules": ["model.embed_tokens", "model.layers", "model.norm"],
      "source": "Qwen2.5-7B",
      "default_action": "freeze",
      "description": "基础 LLM"
    }
  ],
  "notes": "语音大模型，不训练 LLM，训练 adaptor 和 text head"
}
```

## 必须回答的问题

1. 基础 LLM 是什么？
2. `forward` 函数签名是什么？需要哪些输入？
3. **Audio tokenizer / encoder 在哪里？**
   - 是在 LLM **内部**（如 Kimi-Audio 的 Whisper 在 model forward 里跑）？
   - 还是在 LLM **外部**（如 MiMo-Audio 的独立 `MiMoAudioTokenizer`）？
   - 这直接决定 dataset 阶段是否需要预先缓存 audio tokens。
4. **模型输入形式是什么？**
   - 连续特征（如 Whisper feature）还是离散 token IDs（如 RVQ tokens）？
   - 音频部分是什么形状？例如 `[B, T, dim]`、`[B, audio_channels+1, T]` 等。
5. 有哪些 audio encoder（用于把音频编码成特征）？
6. 有哪些 adaptor/connector（把音频特征接到 LLM）？
7. 有哪些 text output head（用于输出文本）？
8. **有哪些音频输出部件（audio decoder/head）？明确标记为"忽略"**
9. 是否需要 model wrapper？
10. 每个组件的推荐 default_action 是什么？
11. **是否需要扩展 special tokens？** 列出模型需要的所有特殊 token，检查 tokenizer 是否已包含。

注意：
- 第 8 点的音频输出部件虽然要识别出来，但不加入 components 列表供用户决策。
- 第 3、4 点决定后续 dataset/template 的实现方式，必须写进 `model_analysis.json` 的 `notes` 中。

## 失败处理

- 无法读取 config：检查模型路径是否正确
- forward 签名复杂：多看几个相关函数，必要时画出输入输出图
- 不确定架构：搜索论文或 issue，记录假设

## 进入下一阶段

model_analysis 完成后，**不要**直接进入 download_weights。**必须先进入 `05-user-decision` 阶段，让用户确认每个组件的训练策略。**

agent 不允许自行决定最终冻结策略。即使推荐策略看起来很明显（如冻结 LLM、训练 adaptor），也必须向用户展示并等待确认。
