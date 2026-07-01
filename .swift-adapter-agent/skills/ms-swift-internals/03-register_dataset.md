# ms-swift 数据集注册机制

## 核心文件

| 文件 | 作用 |
|------|------|
| `swift/llm/dataset/register.py` | `DatasetMeta`, `register_dataset` |
| `swift/llm/dataset/loader.py` | `DatasetSyntax`, `DatasetLoader`, `load_dataset` |
| `swift/llm/dataset/preprocessor/core.py` | `RowPreprocessor`, `AutoPreprocessor`, `MessagesPreprocessor` |
| `swift/llm/dataset/data/dataset_info.json` | 内置数据集注册表 |

## DatasetMeta

`swift/llm/dataset/register.py:44`

```python
@dataclass
class DatasetMeta:
    ms_dataset_id: Optional[str]
    hf_dataset_id: Optional[str]
    dataset_path: Optional[str]
    dataset_name: Optional[str]
    subsets: List[SubsetDataset]
    split: Optional[str]
    preprocess_func: Optional[Callable]
    load_function: Optional[Callable]
```

## register_dataset

`swift/llm/dataset/register.py:88`

```python
def register_dataset(dataset_meta: DatasetMeta, exist_ok: bool = False):
    key = dataset_meta.dataset_name or ...
    DATASET_MAPPING[key] = dataset_meta
```

## load_dataset 数据流

```text
SwiftSft._prepare_dataset()
  └─ _get_dataset()
      └─ load_dataset(args.dataset, split_dataset_ratio=..., shuffle=..., **kwargs)
          ├─ DatasetSyntax.parse(dataset)
          ├─ dataset_meta = DATASET_MAPPING[dataset_syntax.dataset]
          ├─ load_function(dataset_syntax, dataset_meta, **load_kwargs)
          │   └─ DatasetLoader.load()
          │       ├─ _load_dataset_path (jsonl/local dir)
          │       └─ _load_repo_dataset (Hub)
          │           └─ subset.preprocess_func(dataset)
          └─ DatasetLoader.post_process() → train/val split + sampling
  └─ _encode_dataset(train_dataset, val_dataset)
      └─ EncodePreprocessor / AddLengthPreprocessor
```

## jsonl 加载

对于本地 jsonl 文件：`DatasetLoader._load_dataset_path` (`loader.py:197`)

```python
datasets.load_dataset('json', data_files=dataset_path, split='train')
```

## 自定义 Dataset 示例

```python
from swift.llm import register_dataset, DatasetMeta

class MyASRPreprocessor:
    def __call__(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'messages': [
                {'role': 'user', 'content': row.get('prompt', '')},
                {'role': 'assistant', 'content': row['txt']},
            ],
            'audios': [row['wav']],
        }

register_dataset(
    DatasetMeta(
        dataset_path='data/my.jsonl',
        dataset_name='my_dataset',
        preprocess_func=MyASRPreprocessor(),
    ),
    exist_ok=True,
)
```

## 关键注意点

1. `dataset_path` 是 jsonl 文件或目录路径。
2. `preprocess_func` 返回标准 keys：`messages`, `images`, `videos`, `audios`, `tools`, `objects`, `label`。
3. `dataset_name` 用于 `--dataset` 参数。
4. 如果路径不存在，`load_dataset` 会报错。
5. `columns` 参数可以把原始列名映射到标准列名。
