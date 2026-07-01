---
name: model-swift-docker
description: Build or explain a Docker environment for any model project together with a user-specified ms-swift version, using the model's own requirements as the compatibility anchor and producing dependency decisions, Dockerfile templates, and validation commands.
---

# Model + ms-swift Docker

Use this skill when the user wants a Docker image for an arbitrary model repository plus a specified `ms-swift` version. The model and the ms-swift version are inputs, for example: "Qwen2.5-Omni + ms-swift 3.12.6" or "this repo + ms-swift 3.10.2".

## Required Inputs

Collect or infer:

```text
model repo/path:
ms-swift version or local ms-swift path:
CUDA/GPU target:
Python preference:
Docker base image preference or local mirror:
training mode: inference / SFT / full finetune / DeepSpeed / vLLM / flash-attn
```

If the user provides a local model repo or local `ms-swift`, inspect local files first. Prefer `rg --files` and read only relevant files:

```bash
rg --files <repo> | rg 'requirements|pyproject|setup|Dockerfile|environment|conda|constraints'
```

For ms-swift local source, verify version from:

```text
ms-swift/swift/version.py
```

For PyPI ms-swift, pin explicitly:

```bash
pip install "ms-swift==<version>"
```

## Dependency Priority

Choose the environment by this priority:

1. The model repo's hard pins for `torch`, `torchvision`, `torchaudio`, `transformers`, `flash_attn`, `deepspeed`, CUDA extension packages, and model-specific libraries.
2. The requested `ms-swift` version's dependency range.
3. Optional acceleration/runtime packages such as `vllm`, `sglang`, `lmdeploy`, `bitsandbytes`, `liger_kernel`, `auto_gptq`.

If the model pins a torch/CUDA stack, make that the anchor. ms-swift is usually flexible about torch but may restrict `transformers`, `peft`, `trl`, and `datasets`.

Common ms-swift 3.x ranges to check in `requirements/framework.txt`:

```text
transformers
datasets
peft
trl
modelscope
accelerate
```

Never blindly install `ms-swift[all]` when combining with a fragile model stack; it may upgrade torch-adjacent runtimes or install incompatible inference engines.

## Base Image Selection

Prefer a PyTorch devel image matching the selected torch/CUDA stack:

```dockerfile
FROM pytorch/pytorch:<torch-version>-cuda<cuda-version>-cudnn<major>-devel
```

If a local mirror was already pulled, use its exact image name:

```dockerfile
FROM <mirror>/<path>/pytorch/pytorch:<tag>
```

Use an NVIDIA CUDA devel image when no matching PyTorch image exists:

```dockerfile
FROM nvidia/cuda:<cuda-version>-cudnn-devel-ubuntu22.04
```

Use `devel`, not `runtime`, when any of these are needed:

```text
flash_attn
deepspeed ops
apex
xformers built from source
custom CUDA extensions
```

## Python Selection

Prefer Python 3.10 or 3.11 for GPU training images unless the model or user requires otherwise.

Be cautious with Python 3.12 because CUDA extension packages and training runtimes may lag.

## Conflict Handling

Check these conflicts explicitly:

```text
torch / torchvision / torchaudio version alignment
torch CUDA wheel vs base image CUDA compatibility
flash_attn version vs torch/CUDA/Python
deepspeed version vs torch/Python
transformers version required by the model vs ms-swift upper/lower bounds
peft / trl / datasets ranges required by ms-swift
vllm / sglang / lmdeploy constraints if installed
```

When conflicts exist, present a table:

```text
package | model requires | ms-swift requires | decision | risk
```

Prefer not to upgrade the model repo's hard pins unless the user is intentionally porting the model to a newer stack.

## Dockerfile Pattern: Local Model Repo + Local ms-swift

Use this when build context contains `<MODEL_REPO>/` and `ms-swift/`:

```dockerfile
FROM <chosen-base-image>

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# Use a reachable PyPI mirror when the build network cannot access public PyPI reliably.
# ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY <MODEL_REPO> /workspace/<MODEL_REPO>
COPY ms-swift /workspace/ms-swift

WORKDIR /workspace/<MODEL_REPO>
RUN grep -v '^flash_attn==' requirements.txt > /tmp/model-requirements.txt && \
    pip install -r /tmp/model-requirements.txt

# Add model-specific reinstall commands here, for example:
# RUN pip install flash_attn==<version> --no-build-isolation
# RUN pip install deepspeed==<version>

# Pin ms-swift-compatible core packages only if needed after conflict analysis.
# RUN pip install "transformers<..." "datasets..." "peft..." "trl..."

WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace
CMD ["/bin/bash"]
```

## Dockerfile Pattern: PyPI ms-swift

Use this if the user wants a specific published ms-swift version:

```dockerfile
FROM <chosen-base-image>

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
# Use a reachable PyPI mirror when the build network cannot access public PyPI reliably.
# ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential ninja-build cmake pkg-config \
    ffmpeg sox libsndfile1 libgl1-mesa-glx ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY <MODEL_REPO> /workspace/<MODEL_REPO>

WORKDIR /workspace/<MODEL_REPO>
RUN grep -v '^flash_attn==' requirements.txt > /tmp/model-requirements.txt && \
    pip install -r /tmp/model-requirements.txt

# If the model requires flash-attn, install it separately after torch is available.
# RUN pip install flash_attn==<version> --no-build-isolation

RUN pip install "ms-swift==<MS_SWIFT_VERSION>"

WORKDIR /workspace
CMD ["/bin/bash"]
```

## Build And Run Template

```bash
docker build -t <model>-swift:<tag> .
```

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --network=host \
  -v <host-workspace>:/workspace/work \
  <model>-swift:<tag>
```

Add shared memory and ulimit options for heavy dataloading or distributed training if needed:

```bash
--shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864
```

## Validation Commands

Always propose validation commands tailored to the resolved stack:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
python -c "import datasets; print(datasets.__version__)"
python -c "import swift; print(swift.__version__)"
```

If installed:

```bash
python -c "import flash_attn; print('flash_attn ok')"
python -c "import deepspeed; print(deepspeed.__version__)"
python -c "import vllm; print(vllm.__version__)"
```

For a local ms-swift install, expected `swift.__version__` should match `ms-swift/swift/version.py`.

## Troubleshooting Notes

- If pip reports a common package such as `pandas` has no matching distribution, the build is likely using a broken or incomplete pip index. Set `PIP_INDEX_URL` to a reachable mirror, such as `https://pypi.tuna.tsinghua.edu.cn/simple`, or use the company's internal PyPI mirror.
- If a CUDA extension such as `flash_attn` is listed directly in `requirements.txt`, do not assume `pip install -r requirements.txt` is safe. Pip's default build isolation may create a temporary build env without `torch`, causing errors such as `ModuleNotFoundError: No module named 'torch'`.
- For `flash_attn`, prefer filtering it out of the model requirements and installing it separately after torch is available:

```dockerfile
RUN grep -v '^flash_attn==' requirements.txt > /tmp/model-requirements.txt && \
    pip install -r /tmp/model-requirements.txt
RUN pip install flash_attn==<version> --no-build-isolation
```

- Use the same pattern for other fragile compiled packages when they need access to installed torch, CUDA headers, or nvcc during build.

## Response Shape

When answering the user, provide:

1. Compatibility decision in one paragraph.
2. Dependency table for the important packages.
3. Recommended Dockerfile.
4. Build/run commands.
5. Validation commands.
6. Known risks or alternatives.

Keep model-specific conclusions grounded in inspected local files when available.
