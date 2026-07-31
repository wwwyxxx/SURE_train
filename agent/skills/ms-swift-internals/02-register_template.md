# ms-swift 模板注册机制

## 核心文件

| 文件 | 作用 |
|------|------|
| `swift/llm/template/register.py` | `register_template`, `get_template` |
| `swift/llm/template/template_meta.py` | `TemplateMeta` dataclass |
| `swift/llm/template/base.py` | `Template` 基类，`_encode`, `data_collator` |
| `swift/llm/template/template/*.py` | 各模型族的 register_template 调用 |

## TemplateMeta

`swift/llm/template/template_meta.py:14`

```python
@dataclass
class TemplateMeta:
    template_type: str
    prefix: List                   # 对话前缀
    prompt: List                   # 单轮 prompt
    chat_sep: List                 # 多轮分隔符
    suffix: List                   # 对话后缀
    system_prefix: List
    template_cls: Type[Template]   # 默认 Template
    default_system: Optional[str]
    auto_add_bos: bool
    stop_words: List[str]
```

## register_template

`swift/llm/template/register.py:12`

```python
def register_template(template_meta: TemplateMeta, exist_ok: bool = False):
    TEMPLATE_MAPPING[template_meta.template_type] = template_meta
```

调用时机：

- 在自定义注册文件中调用
- `model_meta.template` 引用该 `template_type`

## Template 基类

`swift/llm/template/base.py:42`

关键方法：

| 方法 | 作用 | 行号 |
|------|------|------|
| `_encode(inputs)` | 主编码入口 | `base.py:1329` |
| `_swift_encode(inputs)` | 基于 prefix/prompt/chat_sep/suffix 构建 | `base.py:1128` |
| `_encode_context_list(...)` | 把上下文字符串转成 token ids 和 labels | `base.py:969` |
| `_pre_tokenize(...)` | 替换 `<image>`, `<audio>` 等特殊标签 | `base.py:889` |
| `data_collator(batch, padding_to=None)` | batch 填充入口 | `base.py:1504` |
| `_data_collator(...)` | 核心 padding 逻辑 | `base.py:1696` |
| `replace_tag(media_type, index, inputs)` | 多模态标签替换钩子 | `base.py:791` |

## 数据流

```text
TrainArguments.__post_init__()
  └─ TemplateArguments.__post_init__()
      └─ self.template = self.model_meta.template

SwiftSft._prepare_template()
  └─ args.get_template(self.processor)
      └─ get_template(template_type, processor, ...)
          └─ template_meta.template_cls(processor, template_meta, ...)
              └─ Template.__init__()
```

## 自定义 Template 示例

```python
from swift.llm import register_template, TemplateMeta, Template

class MyAudioTemplate(Template):
    def _encode(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # 加载音频、编码文本、构建 input_ids/labels
        return {
            'input_ids': ...,
            'attention_mask': ...,
            'labels': ...,
        }

    def data_collator(self, batch: List[Dict], *, padding_to=None) -> Dict:
        # padding 逻辑
        return {...}

register_template(
    TemplateMeta(
        template_type='my_audio_text',
        prefix=[],
        prompt=['{{query}}'],
        chat_sep=[],
        suffix=[],
        system_prefix=[],
    ),
    use_slots=True,
    lazy_tokenize=True,
)
```

## 关键注意点

1. `_encode` 返回的 dict 就是 model forward 的输入。
2. `labels` 的 padding 位置必须是 `-100`，否则 loss 会算错。
3. `data_collator` 必须能处理 batch_size > 1 和变长序列。
4. 如果模型 forward 内部会 shift labels，template 里就不要 shift。
5. `replace_tag` 是处理 `<audio>`, `<image>` 等占位符的钩子。
