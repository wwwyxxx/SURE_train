# Docker 环境构建指南

## 定位

本 skill 记录如何为 ms-swift 模型适配构建 Docker 镜像。包括通用模型和 Kimi-Audio 特定配置。

## 通用原则

### 依赖优先级

1. **模型仓库的硬约束**（最高优先级）
   - `torch`, `torchvision`, `torchaudio`
   - `flash_attn`
   - `deepspeed`
   - 模型特定库

2. **ms-swift 版本依赖范围**
   - `transformers`
   - `datasets`
   - `peft`
   - `trl`
   - `modelscope`
   - `accelerate`

3. **可选加速/推理包**
   - `vllm`, `sglang`, `lmdeploy`
   - `bitsandbytes`, `liger_kernel`

### 基础镜像选择

优先使用匹配 torch/CUDA 的 PyTorch devel 镜像：

```dockerfile
FROM pytorch/pytorch:<torch-version>-cuda<cuda-version>-cudnn<major>-devel
```

本地镜像：

```dockerfile
FROM mirrors.aispeech.com/docker/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

必须使用 `devel` 镜像（不是 runtime）当需要：

- `flash_attn`
- `deepspeed ops`
- `apex`
- `xformers`
- 自定义 CUDA 扩展

### Python 版本

- 优先 **Python 3.10 或 3.11**
- 谨慎使用 Python 3.12（CUDA 扩展包可能不支持）

### 冲突检查清单

```text
torch / torchvision / torchaudio 版本对齐
torch CUDA wheel vs 基础镜像 CUDA 兼容性
flash_attn 版本 vs torch/CUDA/Python
deepspeed 版本 vs torch/Python
transformers 版本：模型要求 vs ms-swift 上下界
peft / trl / datasets 范围：ms-swift 要求
vllm / sglang / lmdeploy 约束（如果安装）
```

## 通用 Dockerfile 模板

### 本地模型仓库 + 本地 ms-swift

```dockerfile
FROM <chosen-base-image>

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

COPY <MODEL_REPO> /workspace/<MODEL_REPO>
COPY ms-swift /workspace/ms-swift

WORKDIR /workspace/<MODEL_REPO>
RUN grep -v '^flash_attn==' requirements.txt > /tmp/model-requirements.txt && \
    pip install -r /tmp/model-requirements.txt

# 单独安装 flash-attn
RUN pip install flash_attn==<version> --no-build-isolation

WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace
CMD ["/bin/bash"]
```

### PyPI ms-swift

```dockerfile
FROM <chosen-base-image>

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

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

RUN pip install flash_attn==<version> --no-build-isolation
RUN pip install "ms-swift==<MS_SWIFT_VERSION>"

WORKDIR /workspace
CMD ["/bin/bash"]
```

## Kimi-Audio 特定配置

Kimi-Audio 的硬约束：

```text
torch==2.6.0
torchaudio==2.6.0
flash_attn==2.7.4.post1
deepspeed==0.16.9
```

推荐基础镜像：

```dockerfile
FROM docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

Kimi-Audio Dockerfile（与 `SURE_train/Dockerfile/kimi-audio_dockerfile/Dockerfile` 一致）：

```dockerfile
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

## 验证命令

容器启动后必须验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
python -c "import datasets; print(datasets.__version__)"
python -c "import swift; print(swift.__version__)"
python -c "import flash_attn; print('flash_attn ok')"
python -c "import deepspeed; print(deepspeed.__version__)"
```

Kimi-Audio 期望值：

```text
torch: 2.6.0
cuda: 12.4
cuda available: True
deepspeed: 0.16.9
swift: 3.12.x
```

## Troubleshooting

1. **flash_attn 编译失败**
   - 确认是 `devel` 镜像，有 `nvcc`
   - 用 `--no-build-isolation`
   - 先从 requirements.txt 中过滤掉 flash_attn

2. **pandas 等常见包找不到**
   - 设置 `PIP_INDEX_URL` 为可达的镜像

3. **swift 版本不对**
   - 检查 `ms-swift/swift/version.py`
   - 确认是 `pip install -e /workspace/ms-swift`

4. **transformers 版本冲突**
   - ms-swift 3.12.x 要求 `transformers < 4.58`
   - 优先满足模型硬约束，再调整 ms-swift 兼容包
