import json
import sys


def edit_distance(a: list, b: list) -> int:
    """Standard Levenshtein distance."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def main(path: str):
    total_err = 0
    total_len = 0
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            pred = row.get('response', '') or ''
            ref = row.get('labels', '') or ''
            err = edit_distance(list(ref), list(pred))
            total_err += err
            total_len += len(ref)
            count += 1
    cer = total_err / total_len if total_len > 0 else 0.0
    print(f'Samples: {count}')
    print(f'Total reference chars: {total_len}')
    print(f'Total edit distance: {total_err}')
    print(f'CER: {cer * 100:.4f}%')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'outputs/qwen2_audio_asr_infer_aishell1_test.jsonl')
