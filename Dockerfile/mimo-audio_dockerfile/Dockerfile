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

RUN conda install -y python=3.12 && conda clean -afy

RUN python -m pip install --upgrade pip setuptools wheel

COPY MiMo-Audio /workspace/MiMo-Audio
COPY ms-swift /workspace/ms-swift

WORKDIR /workspace/MiMo-Audio

# MiMo-Audio pins torch/torchaudio/transformers/triton. Keep those pins as the
# shared baseline, because ms-swift accepts the same core version ranges.
RUN pip install -r requirements.txt

# MiMo-Audio imports flash_attn directly. Force a local build so setup.py does
# not fail when it cannot fetch the prebuilt wheel from GitHub.
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 \
    pip install flash-attn==2.7.4.post1 --no-build-isolation

WORKDIR /workspace/ms-swift

# Install the local ms-swift checkout. Its requirements/framework.txt remains
# compatible with MiMo-Audio's torch 2.6.0 and transformers 4.49.0 pins.
RUN pip install -e .

# Optional packages commonly used by ms-swift training/deployment examples.
# Enable at build time with:
#   docker build --build-arg INSTALL_SWIFT_OPTIONAL=true ...
ARG INSTALL_SWIFT_OPTIONAL=false
RUN if [ "${INSTALL_SWIFT_OPTIONAL}" = "true" ]; then \
        pip install "deepspeed>=0.14,<0.18" bitsandbytes qwen_vl_utils qwen_omni_utils \
            decord soundfile liger_kernel wandb swanlab nvitop ; \
    fi

WORKDIR /workspace

CMD ["/bin/bash"]
