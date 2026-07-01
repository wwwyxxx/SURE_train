#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-data/combined_asr_aishell-1.jsonl}"
OUTPUT_JSONL="${2:-data/combined_asr_aishell-1_cached.jsonl}"
CACHE_DIR="${3:-data/audio_tokens_cache/combined_asr_aishell-1}"
CHUNK_SIZE="${4:-1000}"
N_GPUS="${5:-${N_GPUS:-8}}"

ROOT="/aistor/hpc_stor01/home/yixuan.wang_sx/SURE_train"
IMAGE="docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-mimoaudio:v0"

mkdir -p "${CACHE_DIR}"
CHUNK_DIR="${CACHE_DIR}/chunks"
mkdir -p "${CHUNK_DIR}"
LOG_DIR="${CACHE_DIR}/chunk_logs"
mkdir -p "${LOG_DIR}"

echo "Splitting ${DATASET} into chunks of ${CHUNK_SIZE}..."
rm -f "${CHUNK_DIR}"/chunk_*
split -l "${CHUNK_SIZE}" -d "${DATASET}" "${CHUNK_DIR}/chunk_"

chunk_files=("${CHUNK_DIR}"/chunk_*)
total=${#chunk_files[@]}
echo "Total chunks: ${total}"

process_chunk() {
    local gpu_id=$1
    local chunk_file=$2
    local chunk_idx=$3
    local output_file="${CACHE_DIR}/chunk_out_${chunk_idx}.jsonl"

    docker run --rm --gpus "\"device=${gpu_id}\"" --ipc=host --shm-size=64g \
        -v "${ROOT}":/workspace \
        -w /workspace \
        "${IMAGE}" \
        bash -c "cd /workspace && /opt/conda/bin/python tools/precompute_mimo_audio_tokens.py \
            --dataset ${chunk_file} \
            --output-jsonl ${output_file} \
            --cache-dir ${CACHE_DIR} \
            --gpu 0 \
            --skip-existing" > "${LOG_DIR}/chunk_${chunk_idx}.log" 2>&1
}

for ((i=0; i<total; i+=N_GPUS)); do
    pids=()
    for ((j=0; j<N_GPUS && i+j<total; j++)); do
        chunk_idx=$((i+j))
        chunk_file=${chunk_files[$chunk_idx]}
        echo "[batch $((i/N_GPUS + 1))] Starting chunk ${chunk_idx} on GPU ${j}"
        process_chunk ${j} "${chunk_file}" ${chunk_idx} &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
done

echo "Merging outputs into ${OUTPUT_JSONL}..."
> "${OUTPUT_JSONL}"
missing=0
for ((i=0; i<total; i++)); do
    output_file="${CACHE_DIR}/chunk_out_${i}.jsonl"
    if [ -f "${output_file}" ]; then
        cat "${output_file}" >> "${OUTPUT_JSONL}"
    else
        echo "[WARN] Missing output for chunk ${i}"
        missing=$((missing + 1))
    fi
done

echo "Done. Missing chunks: ${missing}. Total cached samples: $(wc -l < ${OUTPUT_JSONL})"
