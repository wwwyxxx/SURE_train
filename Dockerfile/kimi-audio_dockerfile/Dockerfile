FROM docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

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
    openssh-server \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY Kimi-Audio /workspace/Kimi-Audio
COPY ms-swift /workspace/ms-swift

WORKDIR /workspace/Kimi-Audio

# Kimi-Audio dependencies
RUN grep -v '^flash_attn==' requirements.txt > /tmp/kimi-audio-requirements.txt && \
    pip install -r /tmp/kimi-audio-requirements.txt

# flash-attn is safer with no build isolation
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

# ms-swift 3.12.x compatible core versions
RUN pip install \
    "transformers==4.57.6" \
    "datasets>=3.0,<4.0" \
    "peft>=0.11,<0.19" \
    "trl>=0.15,<0.25" \
    "modelscope>=1.23"

WORKDIR /workspace/ms-swift

# Install local ms-swift
RUN pip install -e .

WORKDIR /workspace

CMD ["/bin/bash"]
