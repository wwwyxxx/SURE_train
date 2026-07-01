#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_yukai-sure_train:v0}"
WORKDIR_HOST="${WORKDIR_HOST:-$PWD}"
MIN_FREE_MIB="${MIN_FREE_MIB:-45000}"
MAX_UTIL="${MAX_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_ID="${GPU_ID:-}"

if [ ! -f "$WORKDIR_HOST/run_reprodata_asr.sh" ]; then
  echo "run_reprodata_asr.sh not found under WORKDIR_HOST=$WORKDIR_HOST" >&2
  exit 1
fi

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v min_free="$MIN_FREE_MIB" -v max_util="$MAX_UTIL" '
      {
        gsub(/^[ \t]+|[ \t]+$/, "", $1)
        gsub(/^[ \t]+|[ \t]+$/, "", $2)
        gsub(/^[ \t]+|[ \t]+$/, "", $3)
        if ($2 >= min_free && $3 <= max_util) {
          print $1
          exit
        }
      }'
}

while true; do
  if [ -n "$GPU_ID" ]; then
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')"
    if [ "$free_mib" -ge "$MIN_FREE_MIB" ] && [ "$util" -le "$MAX_UTIL" ]; then
      selected="$GPU_ID"
      break
    fi
    echo "$(date '+%F %T') GPU $GPU_ID not ready: free=${free_mib}MiB util=${util}%"
  else
    selected="$(pick_gpu || true)"
    if [ -n "$selected" ]; then
      break
    fi
    echo "$(date '+%F %T') no GPU ready: need free>=${MIN_FREE_MIB}MiB and util<=${MAX_UTIL}%"
  fi
  sleep "$POLL_SECONDS"
done

echo "$(date '+%F %T') selected GPU $selected, starting training"

exec docker run --rm -it --gpus "\"device=${selected}\"" --ipc=host --shm-size=64g \
  -v "$WORKDIR_HOST":/workspace \
  -w /workspace \
  "$IMAGE" \
  ./run_reprodata_asr.sh
