# ms-swift 扩展点

## 模型注册扩展

### 1. register_model

添加新的 `model_type` 和自定义加载函数。

```python
register_model(ModelMeta(...), exist_ok=True)
```

### 2. register_model_arch

定义模块路径，用于 freeze 和 LoRA target。

```python
register_model_arch(MultiModelKeys(...))
```

### 3. custom_register_path

通过 CLI 参数 `--custom_register_path custom/xxx.py` 导入自定义注册。

## 模板扩展

### 1. 继承 Template 基类

```python
class MyTemplate(Template):
    def replace_tag(self, media_type, index, inputs):
        # 处理 <audio>, <image> 等占位符
        pass

    def _encode(self, inputs):
        # 自定义编码
        pass

    def data_collator(self, batch, *, padding_to=None):
        # 自定义 collator
        pass
```

### 2. register_post_encode_hook

用于多模态模型把 input_ids 转成 inputs_embeds。

```python
Template.register_post_encode_hook(model, hook_func)
```

## 数据集扩展

### 1. register_dataset

```python
register_dataset(DatasetMeta(dataset_path='...', preprocess_func=...), exist_ok=True)
```

### 2. register_dataset_info

通过 JSON 批量注册数据集。

```python
register_dataset_info('custom_dataset_info.json')
```

### 3. 自定义 RowPreprocessor

```python
from swift.llm.dataset.preprocessor import RowPreprocessor

class MyPreprocessor(RowPreprocessor):
    def preprocess(self, row):
        return {'messages': [...], 'audios': [...]}
```

## Trainer 扩展

### 1. 自定义 compute_loss_func

```python
trainer = trainer_cls(
    model,
    args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    compute_loss_func=my_loss_func,
)
```

### 2. 自定义 Callbacks

```python
from transformers import TrainerCallback

class MyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        pass
```

### 3. 自定义 Tuner

通过 `swift/plugin` 注册新的 `train_type`。

```python
# swift/plugin/my_tuner.py
from swift.llm.train.tuner import TunerMixin

class MyTuner(TunerMixin):
    pass
```

## 最常用的钩子

| 场景 | 扩展点 |
|------|--------|
| 新模型族 | `register_model` + `register_model_arch` |
| 新输入格式 | 继承 `Template` + `register_template` |
| 新数据格式 | `register_dataset` + 自定义 preprocessor |
| 新 loss | `compute_loss_func` |
| 新训练策略 | plugin tuner / callback |
| 多模态 embedding | `Template.register_post_encode_hook` |
