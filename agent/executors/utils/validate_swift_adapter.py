#!/usr/bin/env python
import argparse
import os
from typing import Any, Dict, Iterable, List


def _import_custom(path: str) -> None:
    from swift.utils import import_external_file
    import_external_file(path)


def _shape(x: Any):
    return getattr(x, 'shape', None)


def _print_tree(name: str, value: Any) -> None:
    if isinstance(value, dict):
        print(f'{name}: dict')
        for k, v in value.items():
            print(f'  {k}: {type(v).__name__} {_shape(v)}')
    else:
        print(f'{name}: {type(value).__name__} {_shape(value)}')


def _move_to_cuda(obj: Any):
    import torch
    if torch.is_tensor(obj):
        return obj.cuda()
    if isinstance(obj, dict):
        return {k: _move_to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cuda(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cuda(v) for v in obj)
    return obj


def _dataset_registered(dataset_name: str) -> bool:
    from swift.llm.dataset.register import DATASET_MAPPING
    for key in DATASET_MAPPING:
        if key == dataset_name:
            return True
        if isinstance(key, tuple) and dataset_name in key:
            return True
    return False


def validate_register(args) -> None:
    from swift.llm import MODEL_MAPPING, TEMPLATE_MAPPING
    _import_custom(args.custom_register_path)
    print(f'model registered: {args.model_type in MODEL_MAPPING}')
    print(f'template registered: {args.template in TEMPLATE_MAPPING}')
    if args.dataset:
        print(f'dataset registered: {_dataset_registered(args.dataset)}')
    assert args.model_type in MODEL_MAPPING, args.model_type
    assert args.template in TEMPLATE_MAPPING, args.template


def load_first_row(args) -> Dict[str, Any]:
    from swift.llm import load_dataset
    _import_custom(args.custom_register_path)
    train_dataset = load_dataset([args.dataset])[0]
    print(f'dataset: {train_dataset}')
    assert len(train_dataset) > 0, 'dataset is empty'
    row = train_dataset[0]
    print(f'row: {row}')
    return row


def validate_dataset(args) -> Dict[str, Any]:
    row = load_first_row(args)
    assert isinstance(row, dict)
    assert 'messages' in row, row.keys()
    assert isinstance(row['messages'], list) and row['messages'], row['messages']
    if 'audios' in row:
        audios = row['audios']
        if isinstance(audios, str):
            audios = [audios]
        for audio in audios:
            assert os.path.exists(audio), audio
    return row


def get_processor(args, *, load_model: bool):
    from swift.llm import get_model_tokenizer
    _import_custom(args.custom_register_path)
    return get_model_tokenizer(args.model, model_type=args.model_type, load_model=load_model)


def get_template(args, processor):
    from swift.llm import get_template
    template = get_template(args.template, processor)
    template.set_mode('train')
    return template


def validate_encode(args) -> Dict[str, Any]:
    model, processor = get_processor(args, load_model=False)
    template = get_template(args, processor)
    row = validate_dataset(args)
    encoded = template.encode(row)
    print('encoded:')
    for k, v in encoded.items():
        _print_tree(k, v)
    for key in args.required_keys:
        assert key in encoded, f'missing encoded key: {key}'
    if args.assert_text_audio_shapes:
        assert encoded['input_ids'].shape == encoded['text_input_ids'].shape
        assert encoded['input_ids'].shape == encoded['is_continuous_mask'].shape
        if isinstance(encoded['labels'], dict):
            text_labels = encoded['labels']['text_labels']
            text_loss_mask = encoded['labels']['text_loss_mask']
        else:
            text_labels = encoded['labels']
            text_loss_mask = encoded['text_loss_mask']
        assert text_labels.shape == encoded['input_ids'].shape
        assert text_loss_mask.shape == encoded['input_ids'].shape
        assert text_loss_mask.sum().item() > 0
        assert encoded['is_continuous_mask'].sum().item() > 0
    return encoded


def validate_collate(args) -> Dict[str, Any]:
    model, processor = get_processor(args, load_model=False)
    template = get_template(args, processor)
    row = validate_dataset(args)
    encoded = template.encode(row)
    batch = template.data_collator([encoded])
    print('batch:')
    for k, v in batch.items():
        _print_tree(k, v)
    for key in args.required_batch_keys:
        assert key in batch, f'missing batch key: {key}'
    return batch


def validate_init(args):
    model, processor = get_processor(args, load_model=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f'num trainable tensors: {len(trainable)}')
    print(f'trainable examples: {trainable[:args.print_limit]}')
    for prefix in args.must_train_prefix:
        ok = any(n.startswith(prefix) for n in trainable)
        print(f'must train {prefix}: {ok}')
        assert ok, prefix
    for prefix in args.must_freeze_prefix:
        ok = not any(n.startswith(prefix) for n in trainable)
        print(f'must freeze {prefix}: {ok}')
        assert ok, prefix
    return model, processor


def validate_forward(args) -> None:
    import torch
    assert torch.cuda.is_available(), 'CUDA is required for forward validation'
    model, processor = validate_init(args)
    model.cuda()
    model.train()
    template = get_template(args, processor)
    row = validate_dataset(args)
    batch = template.data_collator([template.encode(row)])
    batch = _move_to_cuda(batch)
    outputs = model(**batch)
    loss = outputs.loss
    print(f'loss: {loss.item()}')
    assert torch.isfinite(loss), loss
    if not args.no_backward:
        loss.backward()
        grad_names = [n for n, p in model.named_parameters() if p.grad is not None]
        print(f'num grad tensors: {len(grad_names)}')
        print(f'grad examples: {grad_names[:args.print_limit]}')
        assert grad_names, 'no gradients found'


STAGE_ORDER = ['register', 'dataset', 'encode', 'collate', 'init', 'forward']


def expand_stages(stage: str) -> List[str]:
    if stage == 'all':
        return STAGE_ORDER
    if stage == 'cpu':
        return ['register', 'dataset', 'encode', 'collate']
    return [stage]


def parse_args():
    parser = argparse.ArgumentParser(description='Validate a custom ms-swift model adapter.')
    parser.add_argument('--custom-register-path', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--model-type', required=True)
    parser.add_argument('--template', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument(
        '--stage',
        choices=['register', 'dataset', 'encode', 'collate', 'init', 'forward', 'cpu', 'all'],
        default='cpu')
    parser.add_argument('--required-keys', nargs='*', default=['input_ids', 'labels'])
    parser.add_argument(
        '--required-batch-keys',
        nargs='*',
        default=['input_ids', 'labels'])
    parser.add_argument('--assert-text-audio-shapes', action='store_true')
    parser.add_argument('--must-train-prefix', nargs='*', default=[])
    parser.add_argument('--must-freeze-prefix', nargs='*', default=[])
    parser.add_argument('--no-backward', action='store_true')
    parser.add_argument('--print-limit', type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for stage in expand_stages(args.stage):
        print(f'\n===== validate {stage} =====')
        if stage == 'register':
            validate_register(args)
        elif stage == 'dataset':
            validate_dataset(args)
        elif stage == 'encode':
            validate_encode(args)
        elif stage == 'collate':
            validate_collate(args)
        elif stage == 'init':
            validate_init(args)
        elif stage == 'forward':
            validate_forward(args)
        else:
            raise ValueError(stage)
    print('\nvalidation passed')


if __name__ == '__main__':
    main()
