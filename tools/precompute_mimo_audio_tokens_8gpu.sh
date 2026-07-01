#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-data/combined_asr_aishell-1.jsonl}"
OUTPUT_JSONL="${2:-data/combined_asr_aishell-1_cached.jsonl}"
CACHE_DIR="${3:-data/audio_tokens_cache/combined_asr_aishell-1}"
BATCH_SIZE="${4:-32}"
N_GPUS=8

ROOT="/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train"
IMAGE="docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-mimoaudio:v0"

mkdir -p "${CACHE_DIR}"
SPLIT_DIR="${CACHE_DIR}/splits"
mkdir -p "${SPLIT_DIR}"

echo "Splitting ${DATASET} into ${N_GPUS} parts..."
total_lines=$(wc -l < "${DATASET}")
lines_per_split=$(( (total_lines + N_GPUS - 1) / N_GPUS ))
split -l "${lines_per_split}" -d "${DATASET}" "${SPLIT_DIR}/split_"
# rename split_00 -> split_0.jsonl etc.
for i in $(seq 0 $((N_GPUS - 1))); do
    old="${SPLIT_DIR}/split_$(printf '%02d' ${i})"
    new="${SPLIT_DIR}/split_${i}.jsonl"
    if [ -f "${old}" ]; then
        mv "${old}" "${new}"
    fi
done

PIDS=()
for i in $(seq 0 $((N_GPUS - 1))); do
    SPLIT_PATH="data/audio_tokens_cache/combined_asr_aishell-1/splits/split_${i}.jsonl"
    SPLIT_OUT="data/audio_tokens_cache/combined_asr_aishell-1/cached_split_${i}.jsonl"
    echo "[GPU ${i}] starting preprocessing..."
    docker run --rm --gpus "\"device=${i}\"" --ipc=host --shm-size=64g \
        -v "${ROOT}":/workspace \
        -w /workspace \
        "${IMAGE}" \
        bash -c "cd /workspace && /opt/conda/bin/python tools/precompute_mimo_audio_tokens.py \
            --dataset ${SPLIT_PATH} \
            --output-jsonl ${SPLIT_OUT} \
            --cache-dir ${CACHE_DIR} \
            --gpu 0 \
            --skip-existing" > "${CACHE_DIR}/gpu_${i}.log" 2>&1 &
    PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} GPU processes to finish..."
for pid in "${PIDS[@]}"; do
    wait "${pid}"
done

echo "Merging outputs into ${OUTPUT_JSONL}..."
> "${OUTPUT_JSONL}"
for i in $(seq 0 $((N_GPUS - 1))); do
    SPLIT_OUT="${CACHE_DIR}/cached_split_${i}.jsonl"
    if [ -f "${SPLIT_OUT}" ]; then
        cat "${SPLIT_OUT}" >> "${OUTPUT_JSONL}"
    fi
done

echo "Done. Total cached samples: $(wc -l < ${OUTPUT_JSONL})"
