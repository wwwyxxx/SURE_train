#!/usr/bin/env python3
"""Post-process Qwen2-Audio ASR predictions.

Pipeline per dataset:
  1. clean:      predictions/{name}.jsonl -> predictions_clean/{name}.jsonl
                 (keep only ground_truth + prediction)
  2. normalize:  zh -> evaluation/aispeech_norm, en -> evaluation/whisper_norm
  3. score:      evaluation/wenet_compute_cer.py (CER for zh, WER for en)

All outputs are written to the clean dir.

Usage:
  python3 tools/postprocess_predictions.py \
      --pred-dir outputs/20260629-183012/predictions \
      --clean-dir outputs/20260629-183012/predictions_clean \
      --eval-dir evaluation
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# (prediction file name, language, tochar for wenet)
DATASETS = [
    ('aishell1-test_ASR_infer.jsonl', 'zh', True),
    ('librispeech_test-clean_ASR.jsonl', 'en', False),
    ('librispeech_test-other_ASR.jsonl', 'en', False),
]


def load_package(pkg_name, pkg_dir):
    """Load a package from a directory under a custom name (avoids name clashes)."""
    pkg_dir = Path(pkg_dir)
    spec = importlib.util.spec_from_file_location(
        pkg_name, str(pkg_dir / '__init__.py'),
        submodule_search_locations=[str(pkg_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_module(mod_name, file_path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', default='outputs/20260629-183012/predictions')
    ap.add_argument('--clean-dir', default='outputs/20260629-183012/predictions_clean')
    ap.add_argument('--eval-dir', default='evaluation')
    ap.add_argument('--datasets', nargs='*', default=None,
                    help='Override DATASETS: each item is "fname:lang" (lang=zh|en), '
                         'e.g. aishell1-test_ASR_infer.jsonl:zh. tochar=True for zh, False for en.')
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    clean_dir = Path(args.clean_dir)
    eval_dir = Path(args.eval_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = []
        for item in args.datasets:
            fname, lang = item.rsplit(':', 1)
            datasets.append((fname, lang, lang == 'zh'))
    else:
        datasets = DATASETS

    aispeech_norm = load_package('aispeech_norm_impl', eval_dir / 'aispeech_norm' / 'normalization_impl')
    whisper_norm = load_package('whisper_norm_impl', eval_dir / 'whisper_norm' / 'normalization_impl')
    wenet = load_module('wenet_compute_cer', eval_dir / 'wenet_compute_cer.py')

    zh_normalizer = aispeech_norm.LANG_CLASSES['zh']
    # config() compiles the module-level regexes and loads the tn rule maps;
    # must be called once before pipeline().
    zh_normalizer.config()
    en_normalizer = whisper_norm.EnglishTextNormalizer()

    def normalize_text(text, lang):
        if not text:
            return ''
        if lang == 'zh':
            return zh_normalizer.pipeline(text)
        return en_normalizer(text)

    summary = {}
    for fname, lang, tochar in datasets:
        pred_path = pred_dir / fname
        if not pred_path.exists():
            print(f'SKIP {fname}: not found in {pred_dir}', file=sys.stderr)
            continue
        name = pred_path.stem

        # 1. Clean: keep only ground_truth + prediction.
        pairs = []
        with open(pred_path, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                gt = row.get('ground_truth', row.get('labels'))
                pr = row.get('prediction', row.get('response'))
                if gt is None and pr is None:
                    # Skip summary lines or rows without prediction content.
                    continue
                pairs.append((gt or '', pr or ''))
        clean_path = clean_dir / fname
        with open(clean_path, 'w', encoding='utf-8') as f:
            for gt, pr in pairs:
                f.write(json.dumps({'ground_truth': gt, 'prediction': pr}, ensure_ascii=False) + '\n')

        # 2. Normalize ref/hyp into key-text files for wenet scoring.
        ref_norm = clean_dir / f'{name}.ref.norm.txt'
        hyp_norm = clean_dir / f'{name}.hyp.norm.txt'
        with open(ref_norm, 'w', encoding='utf-8') as fr, open(hyp_norm, 'w', encoding='utf-8') as fh:
            for i, (gt, pr) in enumerate(pairs):
                key = f'utt{i:06d}'
                fr.write(f'{key} {normalize_text(gt, lang)}\n')
                fh.write(f'{key} {normalize_text(pr, lang)}\n')

        # 3. Compute CER (zh) / WER (en).
        result = wenet.compute_wer(str(ref_norm), str(hyp_norm), tochar=tochar)
        n = result['all']
        rate = (result['sub'] + result['del'] + result['ins']) / n if n else 0.0
        metric = 'cer' if tochar else 'wer'
        metrics = {
            'dataset': name,
            'language': lang,
            'metric': metric,
            'num_samples': len(pairs),
            'num_tokens': n,
            'correct': result['cor'],
            'sub': result['sub'],
            'del': result['del'],
            'ins': result['ins'],
            f'{metric}_percent': round(rate * 100, 4),
        }
        metrics_path = clean_dir / f'{name}.metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        summary[name] = metrics
        print(f'[{name}] {metric.upper()} = {rate * 100:.2f}% '
              f'(N={n} C={result["cor"]} S={result["sub"]} D={result["del"]} I={result["ins"]})')

    summary_path = clean_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()
