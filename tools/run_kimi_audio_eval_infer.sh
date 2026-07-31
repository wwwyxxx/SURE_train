#!/usr/bin/env bash
# 8-GPU data-parallel ASR inference for the Kimi-Audio SFT checkpoint.
# Intended to run INSIDE the kimiaudio docker image with cwd=/workspace.
#
#   docker run --rm --gpus all --shm-size=32g \
#     -v .../SURE_train:/workspace -v .../home:/aistor/... \
#     -w /workspace <image> bash tools/run_kimi_audio_eval_infer.sh
#
# Each GPU loads the base model once, processes its strided shard of every
# dataset, and appends to {OUT_DIR}/{name}.shard{ID}.jsonl (resumable).

set -euo pipefail

CHECKPOINT=${CHECKPOINT:-output/run1/v0-20260624-032554/checkpoint-19158}
OUT_DIR=${OUT_DIR:-output/run1/predictions}
NUM_SHARDS=${NUM_SHARDS:-8}
LOG_DIR=${LOG_DIR:-output/run1/infer_logs/eval_shards}
DATASETS=(
  data/test/aishell1-test_ASR_infer.jsonl
  data/test/librispeech_test-clean_ASR.jsonl
  data/test/librispeech_test-other_ASR.jsonl
)

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=${i} python batch_infer_asr.py \
    --checkpoint "${CHECKPOINT}" \
    --datasets "${DATASETS[@]}" \
    --out-dir "${OUT_DIR}" \
    --shard-id "${i}" \
    --num-shards "${NUM_SHARDS}" \
    > "${LOG_DIR}/shard${i}.log" 2>&1 &
done

wait
echo "ALL_SHARDS_DONE"
