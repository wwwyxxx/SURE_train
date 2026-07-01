# Qwen2-Audio + ms-swift 训练实战指南

## 定位

本 skill 记录将 Qwen2-Audio 适配到 ms-swift 框架下进行 ASR 微调的成功经验。

Qwen2-Audio 是 ms-swift **原生支持**的模型类型（`model_type=qwen2_audio`），但默认 template 在 ASR 任务下有一个关键 bug，必须在 custom register 中修复。

## 模型结构

- **LLM**: Qwen2-7B（ASR 任务通常冻结）
- **Audio Encoder**: Qwen2AudioEncoder（32 层，通常冻结）
- **Multi-modal Projector / Aligner**: `multi_modal_projector`
- **Output Head**: `language_model.lm_head`

## ASR 任务训练范围

典型配置：

- 冻结 `language_model`（除 `lm_head` 外）
- 冻结 `audio_tower`
- 训练 `multi_modal_projector` 和/或 `language_model.lm_head`

实践发现：

- 只训练 `multi_modal_projector` 很难在 mini 数据集上收敛（随机初始化）
- 训练 `language_model.lm_head` 可以在 mini100 上快速达到 100% token_acc

## 关键教训：Audio Token 数量不匹配

### 问题

ms-swift 原生 `qwen2_audio` template 在 `_encode` 时，只在 `input_ids` 里放 **1 个 `<|AUDIO|>` token**。

但 Qwen2-Audio 的 audio encoder 对一条 30s 音频会输出约 **55 个 audio feature frames**，对应 **55 个 `<|AUDIO|>` token**。

这导致：

- 训练时 audio feature 和 token 数量不匹配
- 模型无法正确学习音频-文本对齐
- 即使训练 loss 下降，推理输出也混乱

### 根因

`swift.llm.template.template.qwen.Qwen2AudioTemplate._encode` 加载完 audio feature 后，没有根据 `feature_attention_mask` 的长度展开 audio token placeholder。

### 修复

在 custom register 中继承 `Qwen2AudioTemplate` 并重写 `_encode`：

```python
from functools import partial
from swift.llm.template import Template, register_template
from swift.llm.template.constant import MLLMTemplateType
from swift.llm.template.template.qwen import Qwen2AudioTemplate, QwenTemplateMeta
from swift.llm.template.vision_utils import load_batch, load_audio


class Qwen2AudioTemplateFixed(Qwen2AudioTemplate):
    def _encode(self, inputs):
        encoded = Template._encode(self, inputs)
        if inputs.audios:
            audios = load_batch(inputs.audios, load_func=partial(load_audio, sampling_rate=self.sampling_rate))
            audio_inputs = self.processor.feature_extractor(
                audios, sampling_rate=self.sampling_rate, return_attention_mask=True, return_tensors='pt')
            audio_inputs['feature_attention_mask'] = audio_inputs.pop('attention_mask')
            encoded.update(audio_inputs)

            audio_token_id = self._tokenize('<|AUDIO|>')[0]
            input_ids = encoded['input_ids']
            audio_idx = next((i for i, tid in enumerate(input_ids) if tid == audio_token_id), None)
            if audio_idx is not None:
                # Reproduce Qwen2AudioEncoder._get_feat_extract_output_lengths:
                # conv1 (stride=1, padding=1) keeps length
                # conv2 (stride=2, padding=1) -> (L - 1) // 2 + 1
                # avg_pooler (kernel=2, stride=2) -> (feat_len - 2) // 2 + 1
                num_frames = audio_inputs['feature_attention_mask'].sum(-1)
                feat_lengths = (num_frames - 1) // 2 + 1
                num_audio_tokens = ((feat_lengths - 2) // 2 + 1).tolist()
                if isinstance(num_audio_tokens, int):
                    num_audio_tokens = [num_audio_tokens]
                n_tokens = num_audio_tokens[0]
                new_input_ids = input_ids[:audio_idx] + [audio_token_id] * n_tokens + input_ids[audio_idx + 1:]
                encoded['input_ids'] = new_input_ids
                labels = encoded.get('labels')
                if labels is not None:
                    new_labels = labels[:audio_idx] + [-100] * n_tokens + labels[audio_idx + 1:]
                    encoded['labels'] = new_labels
        return encoded


register_template(
    QwenTemplateMeta(MLLMTemplateType.qwen2_audio, template_cls=Qwen2AudioTemplateFixed),
    exist_ok=True,
)
```

### 验证

修复后应验证：自定义 template 生成的 audio token 数量与原生 `Qwen2AudioProcessor` 处理同一条音频得到的数量完全一致。

可用 1000 条不同长度音频做批量对比，期望 100% 匹配。

## Prompt 格式

Qwen2-Audio base 期望 audio token 在 text prompt 之前：

```python
{'role': 'user', 'content': f'<audio>{prompt}'},
{'role': 'assistant', 'content': text},
```

注意 `<audio>` 会被 template 替换为多个 `<|AUDIO|>` token。

## Flash Attention 注意事项

Qwen2-Audio 直接传 `--attn_impl flash_attn` 会触发 CUDA index-out-of-bounds 错误，因为 audio encoder 不支持 flash attention。

如果要用 flash attention，只能给 **LLM 部分**开：

```python
# 在 custom model loader 中
model_config.text_config._attn_implementation = 'flash_attention_2'
model_config.text_config.attn_implementation = 'flash_attention_2'
```

audio encoder 保持默认 attention 实现。

## 训练脚本示例

```bash
NPROC_PER_NODE=7 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift sft \
  --model_type qwen2_audio \
  --model /workspace/outputs/{run_id}/assembled_model \
  --dataset combined_asr_local \
  --custom_register_path /workspace/outputs/{run_id}/custom/qwen2_audio_dataset_register.py \
  --train_type full \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --trainable_parameters language_model.lm_head \
  --use_chat_template false \
  --split_dataset_ratio 0 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --dataloader_num_workers 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --bf16 true \
  --gradient_checkpointing true \
  --max_length 1024 \
  --save_steps 500 \
  --save_total_limit 2 \
  --save_only_model true \
  --output_dir /workspace/outputs/{run_id}/full_combined_asr_local_assembled \
  --report_to none
```

## 常见错误及解决方案

| 错误 | 原因 | 解决 |
|------|------|------|
| 训练 loss 下降但推理输出混乱 | audio token 数量不匹配 | 用上面的 `Qwen2AudioTemplateFixed` 重写 `_encode` |
| `--attn_impl flash_attn` 报 `index out of bounds` | audio encoder 不支持 flash attention | 只给 `text_config` 设置 flash attention，或直接用 sdpa/eager |
| `multi_modal_projector` 训练不收敛 | projector 随机初始化，参数量小 | 同时训练 `lm_head` 或改训练 `lm_head` |
| 推理时输出与训练无关 | 加载的 checkpoint 不完整 | 确认 `--custom_register_path` 带了修复后的 template |

## 性能参考

- 7x A800 80GB
- `max_length=1024`, `per_device_batch_size=8`, `gradient_accumulation=4`
- 训练 `language_model.lm_head`
- 约 5.8s/it（未开 flash attention）

## 成功标志

- mini 数据集上 `token_acc` 快速上升到接近 1.0
- 推理结果与目标转写文本一致
- 全量训练 loss 持续下降
