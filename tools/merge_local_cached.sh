#!/usr/bin/env bash
# Merge the validated non-aishell-1 cached jsonl with the (re-generated) aishell-1 cached jsonl.
set -euo pipefail
PROJECT=/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train
IMAGE="docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-mimoaudio:v0"

cd ${PROJECT}

# Validate both inputs first
python3 tools/verify_cached_jsonl.py \
  --jsonl data/combined_asr_local_non_aishell1_cached.jsonl \
  --expected-lines 1430455

python3 tools/verify_cached_jsonl.py \
  --jsonl data/combined_asr_aishell-1_cached.jsonl \
  --expected-lines 134423

# Merge: non-aishell-1 first, then aishell-1 (matches original combined_asr_local.jsonl order)
python3 tools/merge_cached_jsonl.py \
  --inputs data/combined_asr_local_non_aishell1_cached.jsonl data/combined_asr_aishell-1_cached.jsonl \
  --output data/combined_asr_local_cached.jsonl

# Final count check
echo "Merged lines:"
wc -l data/combined_asr_local_cached.jsonl
