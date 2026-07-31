FROM docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV HF_HOME=/workspace/.cache/huggingface
ENV MODELSCOPE_CACHE=/workspace/.cache/modelscope

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential ninja-build cmake pkg-config \
    ffmpeg sox libsndfile1 libgl1-mesa-glx libglib2.0-0 openssh-server ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY SLAM-LLM /workspace/SLAM-LLM
COPY ms-swift /workspace/ms-swift

WORKDIR /workspace/SLAM-LLM

# Core SLAM-LLM dependencies. Avoid torch override since the base image pins it.
RUN pip install \
    accelerate \
    appdirs \
    loralib \
    bitsandbytes \
    datasets \
    fire \
    peft \
    "transformers>=4.31.0,<4.58.0" \
    sentencepiece \
    py7zr \
    scipy \
    optimum \
    hydra-core>=1.3.2 \
    soundfile \
    librosa \
    einops

# Install fairseq from source for WavLM feature extraction (optional but used by upstream recipes).
RUN pip install git+https://github.com/pytorch/fairseq.git@main || true

WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace
ENV PYTHONPATH=/workspace/SLAM-LLM/src:$PYTHONPATH
CMD ["/bin/bash"]
