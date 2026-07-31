# ms-swift SFT 训练流程

## 核心文件

| 文件 | 作用 |
|------|------|
| `swift/cli/sft.py` | CLI 入口 |
| `swift/llm/train/sft.py` | `SwiftSft` pipeline |
| `swift/llm/base.py` | `SwiftPipeline` 基类 |
| `swift/llm/train/tuner.py` | `TunerMixin.prepare_model` |
| `swift/trainers/trainers.py` | `Seq2SeqTrainer.compute_loss` |
| `swift/trainers/trainer_factory.py` | `TrainerFactory` |

## CLI → Trainer 数据流

```text
swift/cli/sft.py
  └─ sft_main()
      └─ SwiftSft(args).main()
          └─ SwiftPipeline.main() → self.run()
              └─ SwiftSft.run()

SwiftSft.__init__()
  ├─ _prepare_model_tokenizer()  → model, processor
  ├─ _prepare_template()         → template
  ├─ _prepare_callbacks()
  └─ _prepare_flash_ckpt()

SwiftSft.run()
  ├─ _prepare_dataset()          → train_dataset, val_dataset
  │   ├─ _get_dataset()          → load_dataset()
  │   └─ _encode_dataset()
  ├─ data_collator = partial(template.data_collator, padding_to=...)
  ├─ model = self.prepare_model(args, model, template, train_dataset)
  ├─ trainer_cls = TrainerFactory.get_trainer_cls(args)
  └─ trainer = trainer_cls(...)
      └─ self.train(trainer) → trainer.train(resume_from_checkpoint)
```

## compute_loss

`swift/trainers/trainers.py:281`

```python
class Seq2SeqTrainer:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. 取出特殊字段
        compute_loss_func = inputs.pop('compute_loss_func', None)
        loss_scale = inputs.pop('loss_scale', None)
        
        # 2. forward
        outputs = model(**inputs)
        
        # 3. 计算 loss
        if labels was popped:
            loss = per_token_loss_func(outputs, labels, ...)
            if loss_scale is not None:
                loss = loss * loss_scale
            loss = loss.sum() / num_items_in_batch
        else:
            loss = outputs.loss
        
        # 4. 加 aux_loss
        if router_aux_loss_coef:
            loss += aux_loss
        
        # 5. 计算 token acc
        self._compute_acc(outputs, labels)
        
        return (loss, outputs) if return_outputs else loss
```

## training_step

`swift/trainers/trainers.py:385`

```python
def training_step(self, model, inputs, num_items_in_batch=None):
    with self.template.forward_context(self.model, inputs):
        return super().training_step(model, inputs, num_items_in_batch)
```

`forward_context` 用于多模态模型把 `input_ids` 转成 `inputs_embeds`。

## prepare_model

`swift/llm/train/tuner.py`

根据 `args.train_type` 选择 tuner：

- `lora`
- `full`
- `longlora`
- `adapter`
- `llamapro`
- ...

自定义 tuner 可以通过 `swift/plugin` 扩展。

## 关键注意点

1. `compute_loss` 会处理 `loss_scale` 和 per-token loss。
2. 如果自定义 model wrapper 返回的 outputs 没有 `loss`，trainer 会自己算。
3. `forward_context` 是多模态训练的关键钩子。
4. Trainer 类型由 `task_type` 决定，通常是 `Seq2SeqTrainer`。
