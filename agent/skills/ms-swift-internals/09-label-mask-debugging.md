# Label/Mask 调试指南

## 定位

本 skill 用于调试 training loss、token accuracy 或生成质量异常时，证明编码后的训练样本是否正确。

## 适用场景

- loss 或 token_acc 看起来不对
- 怀疑 padding 被算进 loss
- 自定义 template/collator 后结果异常
- 多模态 adapter 需要确认只有 assistant tokens 参与 loss

## 检查目标

```text
loss labels == target tokens + EOS
loss mask 覆盖 exactly 这些 labels
padding/blank/control tokens 不在 loss 中
input, labels, mask 长度一致
```

## 调试流程

1. 检查 template encode 路径和 collator
2. 构建或运行 debug 脚本，不经过训练直接编码原始数据
3. 打印：
   - masked labels
   - decoded text
   - mask positions
   - target token ids
   - 特殊 token 是否错误进入 mask
4. 解释结果后再改超参数或代码

## 必须打印的检查项

```text
input_len
labels_len
loss_mask_sum
target_token_len_plus_eos
mask_positions
masked_label_ids
target_ids_plus_eos
masked_ids_match_target_plus_eos
masked_decoded_skip_special
bad_special_in_masked_labels
```

多模态模型额外检查：

```text
is_continuous_mask_sum（音频特征位置数是否合理）
```

## 通过标准

```text
masked_ids_match_target_plus_eos: True
bad_special_in_masked_labels: []
loss_mask_sum == target_token_len_plus_eos
input_len == labels_len == mask_len
```

## batch size > 1 额外检查

```text
padded label positions are ignored
attention_mask is correct
loss mask is false on padding
sequence lengths did not shift labels across samples
```

## 失败模式分析

### masked_ids_match_target_plus_eos == False

- 检查 next-token shift 方向
- 检查 mask 是否和 labels 一起 shift
- 逐个 token decode 对比

### token_acc 几乎为 0

- **最常见原因：labels 被 shift 了两次**
- 检查点：
  1. Template `_encode` 返回的 labels 是否未 shift（`labels[i] == text_ids[i]`）
  2. Model forward 内部是否又对 labels 做了一次 shift
  3. ms-swift 的 `compute_acc` 会再 shift 一次
- 正确组合：**Template 不 shift → Model forward shift → compute_acc shift**
- 错误组合：Template shift → Model forward shift → compute_acc shift（shift 了两次）

### bad_special_in_masked_labels 非空

- 排除 pad、blank、separator、msg_end tokens
- 确认 assistant EOS 是正确的 EOS

### 长度不一致

- 检查自定义 template 和 collator
- 确保每个流追加的位置数一致
- 避免在 label/mask shift 前加 padding

### 所有检查通过但训练仍差

- padding/mask 不是根因
- 检查 checkpoint save/load、logits head 选择、初始化质量、学习率分组、inference prompt 一致性

## Kimi-Audio 特定注意点

- `text_labels` 由 `text_input_ids` 左移一位并 append pad 得到
- `text_loss_mask` 同样左移
- assistant text tokens 和 `kimia_text_eos` 应该是唯一 masked labels
- `kimia_text_blank`, `msg_end`, `pad` 不应出现在 masked labels 中
- `is_continuous_mask_sum` 应等于 wav 插入的音频特征位置数

## 工具脚本

使用 `.swift-adapter-agent/executors/utils/debug_kimi_audio_encode.py`：

```bash
python .swift-adapter-agent/executors/utils/debug_kimi_audio_encode.py \
  --dataset /workspace/data/reprodata_asr_zh_existing.jsonl \
  --num-samples 5
```

也可以单独使用 validator：

```bash
python .swift-adapter-agent/validators/model_specific/kimi_audio/validate_label_shift.py \
  --custom-register-path custom/kimi_audio_swift_register.py \
  --model-type kimi_audio_text \
  --dataset-name combined_asr_aishell_1
```
