---
name: kimi-audio-swift-docker
description: Build or explain a Docker environment for Kimi-Audio fine-tuning together with local ms-swift 3.12.x, including base image choice, dependency pins, Dockerfile template, and validation commands.
---

# Kimi-Audio + ms-swift Docker

Use this skill when setting up, reviewing, or troubleshooting a Docker image for Kimi-Audio fine-tuning with a local `ms-swift` 3.12.x checkout.

## Core Decision

Use the Kimi-Audio training stack as the compatibility anchor, then install ms-swift into that environment.

Kimi-Audio has hard pins:

```text
torch==2.6.0
torchaudio==2.6.0
flash_attn==2.7.4.post1
deepspeed==0.16.9
```

ms-swift 3.12.x is more flexible:

```text
python >=3.8, but prefer 3.10/3.11
torch >=2.0
transformers >=4.33,<4.58
datasets >=3.0,<4.0
peft >=0.11,<0.19
trl >=0.15,<0.25
```

Prefer Python 3.10. Avoid Python 3.12 unless the user explicitly wants it, because CUDA extension packages such as `flash_attn` and `deepspeed` are more fragile there.

## Preferred Base Image

If available, use:

```dockerfile
FROM mirrors.aispeech.com/docker/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

Alternative public image:

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

Fallback:

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
```

For the fallback, explicitly install Python 3.10, pip, `torch==2.6.0`, and `torchaudio==2.6.0`.

Do not prefer `pytorch:2.2.1-cuda12.1-cudnn8-devel`; it is likely to cause torch/CUDA/flash-attn ABI churn because Kimi-Audio expects torch 2.6.0.

## Dockerfile Template

Use this when the build context contains both `Kimi-Audio/` and `ms-swift/`:

```dockerfile
FROM mirrors.aispeech.com/docker/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

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

RUN grep -v '^flash_attn==' requirements.txt > /tmp/kimi-audio-requirements.txt && \
    pip install -r /tmp/kimi-audio-requirements.txt
RUN pip install flash_attn==2.7.4.post1 --no-build-isolation

RUN pip install \
    "transformers==4.57.6" \
    "datasets>=3.0,<4.0" \
    "peft>=0.11,<0.19" \
    "trl>=0.15,<0.25" \
    "modelscope>=1.23"

WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace

CMD ["/bin/bash"]
```

## Build And Run

Build from the directory containing `Kimi-Audio/`, `ms-swift/`, and the Dockerfile:

```bash
docker build -t kimi-audio-swift:torch260-cu124 .
```

Run with GPU access:

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --network=host \
  -v /mnt/lustre/hpc_stor01/home/yixuan.wang_sx/SURE_train:/workspace/SURE_train \
  kimi-audio-swift:torch260-cu124
```

Adjust the host volume path to the user's workspace.

## Validation

Inside the container, verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import flash_attn; print('flash_attn ok')"
python -c "import deepspeed; print(deepspeed.__version__)"
python -c "import swift; print(swift.__version__)"
```

Expected key values:

```text
torch: 2.6.0
cuda: 12.4
cuda available: True
deepspeed: 0.16.9
swift: 3.12.x, for this workspace usually 3.12.6
```

If `import swift` does not show the expected version, inspect `ms-swift/swift/version.py`. Editable install from local source is preferred:

```bash
pip install -e /workspace/ms-swift
```

## Troubleshooting Notes

- If `docker images` crashes with a Go panic, the local Docker CLI or cluster wrapper is broken; this is not evidence that the image is missing.
- If the mirror image was pulled as `mirrors.aispeech.com/docker/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, use that exact name in `FROM`.
- If pip reports a common package such as `pandas` has no matching distribution, the Docker build is probably using a broken or incomplete pip index. Set `PIP_INDEX_URL`, for example `https://pypi.tuna.tsinghua.edu.cn/simple`, or use the company's internal PyPI mirror.
- Do not install Kimi-Audio's `requirements.txt` as-is when it contains `flash_attn==2.7.4.post1`. Pip's default build isolation may create a temporary build env without `torch`, causing `ModuleNotFoundError: No module named 'torch'`. Filter `flash_attn` out of requirements, then install it separately with `--no-build-isolation`.
- If `flash_attn` still fails, confirm the image is a `devel` image with `nvcc`, then reinstall using `pip install flash_attn==2.7.4.post1 --no-build-isolation`.
- Keep `transformers<4.58`; ms-swift 3.12.x requires it and Kimi-Audio does not need a newer version.
- Keep Kimi-Audio's torch, flash-attn, and deepspeed pins unless the user is deliberately changing the training stack.
