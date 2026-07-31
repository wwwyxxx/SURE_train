import json
import sys

def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = curr[-1] + 1
            dels = prev[j + 1] + 1
            subs = prev[j] + (c1 != c2)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]

path = sys.argv[1]
total_edits = 0
total_chars = 0
num = 0
with open(path) as f:
    for line in f:
        row = json.loads(line)
        pred = row.get('response', '')
        label = row.get('labels', '')
        ed = edit_distance(pred, label)
        total_edits += ed
        total_chars += len(label)
        num += 1

cer = total_edits / total_chars if total_chars > 0 else 0
print(f'Samples: {num}')
print(f'Total chars: {total_chars}')
print(f'Total edits: {total_edits}')
print(f'CER: {cer * 100:.2f}%')
