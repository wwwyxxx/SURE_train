FROM docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential ninja-build cmake pkg-config \
    ffmpeg sox libsndfile1 libgl1-mesa-glx openssh-server ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY qwen2-5-omni /workspace/qwen2-5-omni
COPY ms-swift /workspace/ms-swift

WORKDIR /workspace/qwen2-5-omni
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace
CMD ["/bin/bash"]
