#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/model/Qwen2.5-7B}"
CHECKPOINT="${CHECKPOINT:-}"
AUDIO="${AUDIO:-example/BAC009S0002W0263.wav}"
if [ "$#" -gt 0 ]; then
  STAGES=("$@")
else
  STAGES=(cpu)
fi

python skills/swift-model-adapter/scripts/run_adapter_harness.py \
  --custom-register-path custom/kimi_audio_swift_register.py \
  --model "$MODEL_DIR" \
  --model-type kimi_audio_text \
  --template kimi_audio_text \
  --dataset kimi_audio_asr_overfit100 \
  --stages "${STAGES[@]}" \
  --required-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels text_loss_mask \
  --required-batch-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels text_loss_mask \
  --assert-text-audio-shapes \
  --must-train-prefix whisper_model. model.vq_adaptor. mimo_output. \
  --must-freeze-prefix model.layers. \
  --train-type full \
  --freeze-llm true \
  --freeze-vit false \
  --freeze-aligner false \
  --bf16 \
  --gradient-checkpointing \
  --num-train-epochs 3 \
  --save-steps 300 \
  --save-total-limit 1 \
  --save-only-model \
  --output-dir output/kimi_audio_asr_overfit100 \
  --infer-script infer_overfit.py \
  ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
  --audio "$AUDIO"
