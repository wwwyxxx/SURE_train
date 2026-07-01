在训练容器里用。先进入项目目录：

  cd /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train

  先跑 CPU 侧检查，不加载大模型权重：

  python skills/swift-model-adapter/scripts/validate_swift_adapter.py \
    --custom-register-path custom/kimi_audio_swift_register.py \
    --model /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B \
    --model-type kimi_audio_text \
    --template kimi_audio_text \
    --dataset kimi_audio_asr_overfit100 \
    --stage cpu \
    --required-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels \
    --required-batch-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels
  \
    --assert-text-audio-shapes

  这个会依次检查：

  register -> dataset -> encode -> collate

  通过后，再跑 GPU forward/backward 检查：

  python skills/swift-model-adapter/scripts/validate_swift_adapter.py \
    --custom-register-path custom/kimi_audio_swift_register.py \
    --model /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B \
    --model-type kimi_audio_text \
    --template kimi_audio_text \
    --dataset kimi_audio_asr_overfit100 \
    --stage forward \
    --required-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels \
    --required-batch-keys input_ids text_input_ids is_continuous_mask whisper_input_feature labels
  \
    --assert-text-audio-shapes \
    --must-train-prefix whisper_model. model.vq_adaptor. mimo_output. \
    --must-freeze-prefix model.layers.

  如果只想跑某一层，比如只检查注册：

  python skills/swift-model-adapter/scripts/validate_swift_adapter.py \
    --custom-register-path custom/kimi_audio_swift_register.py \
    --model /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train/model/Qwen2.5-7B \
    --model-type kimi_audio_text \
    --template kimi_audio_text \
    --dataset kimi_audio_asr_overfit100 \
    --stage register

  推荐顺序：

  先 --stage cpu
  再 --stage forward
  最后再 swift sft 跑 overfit

  forward 会加载 Qwen2.5-7B 和 Whisper，并做一次 backward，所以需要 GPU 和足够显存。