#!/usr/bin/env bash
# Recreate the full aishell-1 cached jsonl on GPU 7 only.
set -euo pipefail
PROJECT=/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train
INPUT_JSONL=/workspace/data/combined_asr_aishell-1.jsonl
CACHE_DIR=/workspace/data/audio_tokens_cache/combined_asr_aishell-1
OUTPUT_JSONL=/workspace/data/combined_asr_aishell-1_cached.jsonl
IMAGE="docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-mimoaudio:v0"

cd ${PROJECT}
mkdir -p data/audio_tokens_cache/combined_asr_aishell-1

# Re-tokenize the full aishell-1 dataset on GPU 7.
# --skip-existing avoids overwriting already-cached .pt files whose keys happen to match.
docker run --rm --gpus device=7 --ipc=host --shm-size=64g \
  -v ${PROJECT}:/workspace \
  -v /aistor:/aistor \
  -w /workspace ${IMAGE} bash -c \
  "cd /workspace && /opt/conda/bin/python tools/precompute_mimo_audio_tokens.py \
    --dataset ${INPUT_JSONL} \
    --output-jsonl ${OUTPUT_JSONL} \
    --cache-dir ${CACHE_DIR} \
    --gpu 0 \
    --skip-existing"

# Final validation
docker run --rm --gpus device=7 --ipc=host --shm-size=64g \
  -v ${PROJECT}:/workspace \
  -v /aistor:/aistor \
  -w /workspace ${IMAGE} bash -c \
  "cd /workspace && /opt/conda/bin/python tools/verify_cached_jsonl.py \
    --jsonl ${OUTPUT_JSONL} \
    --expected-lines 134423"
