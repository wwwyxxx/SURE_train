#!/usr/bin/env bash
set -euo pipefail

# Run ASR inference on the 3 datasets under data/test/ with the trained Qwen2-Audio checkpoint.
# - aishell1: reuse the previously computed predictions (same dataset content)
# - librispeech test-clean / test-other: run swift infer in Docker

CKPT_DIR="/workspace/outputs/20260629-183012/full_combined_asr_local_assembled/v3-20260704-013603/checkpoint-33528"
PRED_DIR_HOST="$PWD/outputs/20260629-183012/predictions"
PRED_DIR_CT="/workspace/outputs/20260629-183012/predictions"

mkdir -p "$PRED_DIR_HOST"

# Reuse the existing aishell1-test predictions (7176 utts, same dataset content).
if [ ! -f "$PRED_DIR_HOST/aishell1-test_ASR_infer.jsonl" ]; then
  cp outputs/qwen2_audio_asr_infer_aishell1_test.jsonl "$PRED_DIR_HOST/aishell1-test_ASR_infer.jsonl"
  echo "Copied existing aishell1-test predictions."
fi

run_infer() {
  local dataset_name="$1"
  local result_file="$2"
  if [ -f "$PRED_DIR_HOST/$result_file" ]; then
    echo "Skip $dataset_name: $result_file already exists."
    return 0
  fi
  docker run --rm --gpus all --ipc=host --shm-size=64g \
    -v "$PWD":/workspace \
    -v /aistor/hpc_stor01/home/yixuan.wang_sx/test_datasets:/aistor/hpc_stor01/home/yixuan.wang_sx/test_datasets \
    -w /workspace \
    docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-qwen2audio:v0 \
    swift infer \
      --custom_register_path /workspace/custom/qwen2_audio_asr_infer_register.py \
      --model "$CKPT_DIR" \
      --model_type qwen2_audio_flash_llm \
      --template qwen2_audio \
      --val_dataset "$dataset_name" \
      --infer_backend pt \
      --max_batch_size 8 \
      --max_new_tokens 64 \
      --temperature 0.0 \
      --result_path "$PRED_DIR_CT/$result_file"
}

run_infer librispeech_test_clean_asr librispeech_test-clean_ASR.jsonl
run_infer librispeech_test_other_asr librispeech_test-other_ASR.jsonl

echo "All inference done. Predictions in: $PRED_DIR_HOST"
