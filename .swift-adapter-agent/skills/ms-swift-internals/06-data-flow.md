# ms-swift 完整数据流

## 训练前准备

```text
CLI args
  └─ TrainArguments
      ├─ model_type → MODEL_MAPPING → ModelMeta
      ├─ template_type → TEMPLATE_MAPPING → TemplateMeta
      └─ dataset → DATASET_MAPPING → DatasetMeta
```

## 单条样本处理

```text
raw jsonl row
  └─ preprocess_func(row)
      └─ {messages, audios, images, ...}
          └─ template._encode(inputs)
              └─ {input_ids, attention_mask, labels, ...}
                  └─ data_collator(batch)
                      └─ {input_ids[B, L], labels[B, L], attention_mask[B, L]}
                          └─ model(**batch)
                              └─ outputs.logits / outputs.loss
```

## 训练循环

```text
for batch in dataloader:
    ├─ template.forward_context(model, batch)
    ├─ outputs = model(**batch)
    ├─ loss = compute_loss(outputs, labels)
    ├─ loss.backward()
    └─ optimizer.step()
```

## 关键字段传递

| 阶段 | 字段 | 含义 |
|------|------|------|
| dataset | `messages` | 对话列表 |
| dataset | `audios` | 音频路径列表 |
| dataset | `images` | 图片路径列表 |
| template | `input_ids` | 输入 token ids |
| template | `text_input_ids` | 文本部分 token ids（多模态） |
| template | `is_continuous_mask` | 连续 token mask（多模态） |
| template | `whisper_input_feature` | 音频特征（Kimi-Audio） |
| template | `attention_mask` | 有效位置 mask |
| template | `labels` | 预测目标 |
| template | `text_loss_mask` | 哪些位置算 loss（Kimi-Audio） |
| collator | `loss_scale` | 样本级 loss 权重 |
| trainer | `num_items_in_batch` | batch 有效 token 数 |

## 自定义模型需要控制的点

1. **dataset**：输出正确的 `messages` 和模态字段
2. **template**：把 messages + 模态编码成 model forward 的输入
3. **model wrapper**：接收 template 输出，调用原始模型，返回 loss/logits
4. **trainer**：正常执行 `compute_loss` 和 `training_step`
