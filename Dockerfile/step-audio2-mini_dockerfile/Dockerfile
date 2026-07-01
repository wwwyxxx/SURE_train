FROM docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV HF_HOME=/workspace/.cache/huggingface
ENV MODELSCOPE_CACHE=/workspace/.cache/modelscope
ENV XDG_CACHE_HOME=/workspace/.cache

WORKDIR /workspace

# Ensure cache directories are writable regardless of the user id used at runtime.
RUN mkdir -p /workspace/.cache /workspace/output && chmod -R 777 /workspace/.cache /workspace/output

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

# Step-Audio-2-mini dependencies from ms-swift registration and the official repo:
#   transformers==4.53.3, torchaudio, librosa
#   transformers==4.49.0 torchaudio librosa onnxruntime s3tokenizer diffusers hyperpyyaml
# ms-swift explicitly requires transformers==4.53.3 for step_audio2_mini, so use that.
ARG TRANSFORMERS_VERSION=4.53.3
RUN pip install \
    "transformers==${TRANSFORMERS_VERSION}" \
    accelerate \
    torchaudio \
    librosa \
    onnxruntime \
    s3tokenizer \
    diffusers \
    hyperpyyaml

# Flash Attention 2 (used by get_model_tokenizer_with_flash_attn in ms-swift).
# Force a local source build because GitHub prebuilt wheels may not be reachable.
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV MAX_JOBS=8
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

# Install the local ms-swift checkout.
COPY ms-swift /workspace/ms-swift
WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace

CMD ["/bin/bash"]
