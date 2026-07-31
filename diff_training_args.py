import json, re

v1 = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom/v1-20260701-191554/args.json'
control = '/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train/outputs/20260630-172621/smoke_test/overfit1_custom_control/v0-20260702-060512/args.json'

a1 = json.load(open(v1))
a2 = json.load(open(control))

# extract training_args string and find key params
ta1 = str(a1.get('training_args', ''))
ta2 = str(a2.get('training_args', ''))

# parse simple key=value pairs
import ast

def parse_training_args_str(s):
    # Find the substring inside Seq2SeqTrainingArguments(...)
    start = s.find('Seq2SeqTrainingArguments(')
    if start == -1:
        return {}
    # balanced parens
    depth = 0
    end = None
    for i in range(start, len(s)):
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
            if depth == 0:
                end = i
                break
    content = s[start+len('Seq2SeqTrainingArguments('):end]
    # rough parse: split on commas not inside brackets/quotes
    result = {}
    # use regex for key=value where value is simple
    for m in re.finditer(r'(\w+)=([^,]+)', content):
        k, v = m.group(1), m.group(2).strip()
        result[k] = v
    return result

p1 = parse_training_args_str(ta1)
p2 = parse_training_args_str(ta2)

for k in sorted(set(p1.keys()) | set(p2.keys())):
    v1_val = p1.get(k, 'MISSING')
    v2_val = p2.get(k, 'MISSING')
    if v1_val != v2_val:
        print(f'DIFF {k}: v1={v1_val} control={v2_val}')
