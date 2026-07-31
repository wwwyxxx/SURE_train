#!/usr/bin/env bash
set -euo pipefail

# Run ASR inference on aishell1-test with the trained Qwen2-Audio checkpoint.
# Mounts the project root to /workspace inside the container.

CKPT_DIR="/workspace/outputs/20260629-183012/full_combined_asr_local_assembled/v3-20260704-013603/checkpoint-33528"
RESULT_PATH="/workspace/outputs/qwen2_audio_asr_infer_aishell1_test.jsonl"

mkdir -p "$PWD/outputs"

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
    --val_dataset aishell1_test_asr_infer \
    --infer_backend pt \
    --max_batch_size 8 \
    --max_new_tokens 64 \
    --temperature 0.0 \
    --result_path "$RESULT_PATH"

echo "Inference complete. Results saved to: $RESULT_PATH"
