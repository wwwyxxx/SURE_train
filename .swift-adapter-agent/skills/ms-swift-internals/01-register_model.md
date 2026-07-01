# ms-swift 模型注册机制

## 核心文件

| 文件 | 作用 |
|------|------|
| `swift/llm/model/register.py` | `ModelMeta`, `register_model`, `get_model_tokenizer` |
| `swift/llm/model/model_arch.py` | `ModelKeys`, `MultiModelKeys`, `register_model_arch` |
| `swift/llm/model/constant.py` | `LLMModelType`, `MLLMModelType` 常量 |
| `swift/llm/model/model/*.py` | 各模型族的 register_model 调用 |
| `swift/llm/model/utils.py` | `HfConfigFactory`, `ModelInfo`, `get_llm_model` |

## ModelMeta

`swift/llm/model/register.py:58`

```python
@dataclass
class ModelMeta:
    model_type: str
    model_groups: List[str]
    template: str              # 默认 template_type
    get_function: Callable     # 加载函数
    model_arch: ModelKeys      # 模型架构
    architectures: List[str]   # HF architectures
    torch_dtype: torch.dtype
    is_multimodal: bool
    is_reward: bool
    task_type: TaskType
```

## register_model

`swift/llm/model/register.py:126`

```python
def register_model(model_meta: ModelMeta, exist_ok: bool = False):
    MODEL_MAPPING[model_meta.model_type] = model_meta
```

调用时机：

- 在自定义注册文件（如 `custom/xxx_swift_register.py`）中调用
- 导入该文件时执行，写入 `MODEL_MAPPING`

## register_model_arch

`swift/llm/model/model_arch.py:142`

```python
@dataclass
class MultiModelKeys(ModelKeys):
    language_model: List[str]
    vision_tower: List[str]
    aligner: List[str]
    generator: List[str]

register_model_arch(MultiModelKeys(...))
```

作用：

- 定义模型各模块的参数路径
- 用于 `--freeze_llm`, `--freeze_vit`, `--freeze_aligner` 等参数
- 也用于 LoRA target 选择

## get_model_tokenizer 数据流

```text
swift/cli/sft.py
  └─ sft_main()
      └─ SwiftSft(args)
          └─ _prepare_model_tokenizer()
              └─ args.get_model_processor()
                  └─ get_model_tokenizer(**kwargs)          [register.py:710]
                      ├─ get_model_info_meta()              [register.py:638]
                      │   ├─ get_matched_model_meta(model_id_or_path) [register.py:554]
                      │   ├─ safe_snapshot_download()       [utils.py:276]
                      │   ├─ AutoConfig.from_pretrained()
                      │   ├─ get_matched_model_types(architectures) [register.py:579]
                      │   └─ MODEL_MAPPING[model_type]
                      └─ model_meta.get_function(model_dir, model_info, model_kwargs, ...)
```

## 自定义模型加载

实际 HF 加载发生在 `get_model_tokenizer_from_local`：`register.py:276`

```python
automodel_class.from_pretrained(
    model_dir,
    config=model_config,
    trust_remote_code=True,
    **model_kwargs
)
```

## 自定义模型注册示例

```python
from swift.llm import register_model, ModelMeta, MultiModelKeys, register_model_arch

register_model_arch(
    MultiModelKeys(
        'my_audio_arch',
        language_model=['model.embed_tokens', 'model.layers', 'model.norm'],
        vision_tower=['whisper_model'],
        aligner=['model.vq_adaptor', 'mimo_output'],
        generator=['lm_head'],
    )
)

register_model(
    ModelMeta(
        model_type='my_audio_text',
        model_groups=['my_audio'],
        template='my_audio_text',
        get_function=get_model_tokenizer_my_audio,
        model_arch='my_audio_arch',
        architectures=['MyAudioModel'],
        torch_dtype=torch.bfloat16,
        is_multimodal=True,
    ),
    exist_ok=True,
)
```

## get_function 签名

```python
def get_model_tokenizer_my_audio(
    model_dir: str,
    model_info: ModelInfo,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs
) -> Tuple[nn.Module, PreTrainedTokenizerBase]:
    # 1. 加载 config
    # 2. 构建/加载模型
    # 3. 返回 (model, tokenizer)
    pass
```

## 关键注意点

1. `model_type` 必须唯一，否则会覆盖已有注册。
2. `get_function` 返回的 tokenizer 会被 template 使用。
3. `model_arch` 影响 `--freeze_llm/freeze_vit/freeze_aligner`。
4. 自定义注册文件通过 `--custom_register_path` 参数传入，ms-swift 会 import 它。
