FROM docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV HF_HOME=/workspace/.cache/huggingface
ENV MODELSCOPE_CACHE=/workspace/.cache/modelscope

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    ffmpeg \
    sox \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssh-server \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Qwen2.5-Omni model code is included in transformers >= 4.50.
# The official Qwen repo recommends 4.52.3; ms-swift accepts >=4.50,<4.58.
ARG TRANSFORMERS_VERSION=4.52.3
RUN pip install "transformers==${TRANSFORMERS_VERSION}" accelerate

# Multimodal dependencies required by ms-swift for qwen2_5_omni:
#   transformers>=4.50, soundfile, qwen_omni_utils, decord
# qwen-omni-utils provides the qwen_omni_utils import used by the template/model code.
# decord is optional but recommended for faster video loading; if the wheel is
# unavailable for the container's Python version it can be skipped and torchvision
# will be used as a fallback.
RUN pip install \
    "qwen-omni-utils>=0.0.8" \
    soundfile \
    librosa \
    torchaudio \
    torchvision \
    av \
    ffmpeg-python

ARG INSTALL_DECORD=false
RUN if [ "${INSTALL_DECORD}" = "true" ]; then \
        pip install decord || echo "decord build failed, will fall back to torchvision"; \
    fi

# Flash Attention 2 (recommended for BF16 training/inference with Qwen2.5-Omni).
# Force a local source build and avoid fetching prebuilt wheels from GitHub.
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV MAX_JOBS=8
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

# Install the local ms-swift checkout.
COPY ms-swift /workspace/ms-swift
WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace

CMD ["/bin/bash"]
