#!/usr/bin/env bash
set -euo pipefail

MIN_FREE_MB="${MIN_FREE_MB:-70000}"
INTERVAL="${INTERVAL:-10}"

echo "Waiting for any GPU with at least ${MIN_FREE_MB} MiB free..."

while true; do
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)

  for GPU_ID in $(seq 0 $((GPU_COUNT - 1))); do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | tr -d ' ')
    echo "$(date '+%F %T') GPU ${GPU_ID} free: ${FREE_MB} MiB"

    if [ "${FREE_MB}" -ge "${MIN_FREE_MB}" ]; then
      echo "Using GPU ${GPU_ID}"
      docker run --rm -it --gpus "\"device=${GPU_ID}\"" --ipc=host --shm-size=64g \
        -v "$PWD":/workspace \
        -v "$PWD":/mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train \
        -w /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train \
        docker.v2.aispeech.com/sjtu/sjtu_yukai-sure_train:v0 \
        bash run1.sh
      exit $?
    fi
  done

  sleep "${INTERVAL}"
done
