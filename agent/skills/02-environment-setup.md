# 02 - Environment Setup

## 目标

为模型找到或构建合适的 Docker 镜像。

## 命名规范

Docker 镜像名必须是：

```text
docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-<模型名>:<版本>
```

例如：

```text
docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0
docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-mimoaudio:v0
```

模型名转换规则：

- `kimi_audio` → `kimiaudio`
- `mimo_audio` → `mimoaudio`
- 去掉所有 `_` 和 `-`，全部小写

Dockerfile 目录命名规则：

- `SURE_train/Dockerfile/<model-family-with-hyphens>_dockerfile/`
- 例如：`SURE_train/Dockerfile/kimi-audio_dockerfile/`

## Agent Checklist

- [ ] 调用 `resolve_docker_image.py`：
  ```bash
  python .swift-adapter-agent/executors/resolve_docker_image.py \
    --model-family {model_family} \
    --version v0 \
    --sure-train-dir SURE_train \
    --output outputs/{run_id}/docker_resolution.json \
    --create-if-missing
  ```
- [ ] 读取 `docker_resolution.json`
- [ ] 根据 `action` 字段执行：
  - `use_existing`：镜像已存在，直接使用
  - `build_existing`：Dockerfile 已存在，需要 build
  - `build_new`：需要新建 Dockerfile 并 build（仅在传入 `--create-if-missing` 时才会实际创建文件）
- [ ] 如果需要 build，调用 `build_docker.py`：
  ```bash
  python .swift-adapter-agent/executors/build_docker.py \
    --dockerfile SURE_train/Dockerfile/kimi-audio_dockerfile/Dockerfile \
    --image-name docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0 \
    --build-context SURE_train \
    --output outputs/{run_id}/docker_build_report.json
  ```
- [ ] 调用验证：
  ```bash
  python .swift-adapter-agent/executors/run_validator.py \
    --validator .swift-adapter-agent/validators/core/validate_docker_image.py \
    --image docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0 \
    --output outputs/{run_id}/validation_docker.json
  ```
- [ ] 更新 `pipeline_state.json`

## 三层查找逻辑

```text
1. docker images
   检查是否存在 docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-<模型名>:v0
   ↓ 存在 -> use_existing
   ↓ 不存在
2. 检查 SURE_train/Dockerfile/<模型名_dockerfile>/Dockerfile
   ↓ 存在 -> build_existing
   ↓ 不存在
3. 新建 SURE_train/Dockerfile/<模型名_dockerfile>/Dockerfile
   -> build_new（需要 `--create-if-missing` 才会实际创建）
```

## 输出 JSON 格式

`docker_resolution.json`：

```json
{
  "action": "use_existing",
  "image_name": "docker.v2.aispeech.com/sjtu/sjtu_yukai-yixuanwang-kimiaudio:v0",
  "dockerfile_path": "SURE_train/Dockerfile/kimi-audio_dockerfile/Dockerfile",
  "exists_locally": true,
  "dockerfile_exists": true,
  "created": false
}
```

## 新建 Dockerfile 规则

只有在步骤 1 和步骤 2 都失败时，才允许新建 Dockerfile。且必须显式传入 `--create-if-missing`，否则 `resolve_docker_image.py` 只返回查询结果，不会写入任何文件。

新建 Dockerfile 必须放在：

```text
SURE_train/Dockerfile/<模型名_dockerfile>/Dockerfile
```

可以参考已有 Dockerfile：

- `SURE_train/Dockerfile/kimi-audio_dockerfile/Dockerfile`
- `SURE_train/Dockerfile/mimo-audio_dockerfile/Dockerfile`

新建 Dockerfile 的基本要求：

```dockerfile
FROM docker.1ms.run/pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \\
    git curl build-essential ninja-build cmake pkg-config \\
    ffmpeg sox libsndfile1 libgl1-mesa-glx openssh-server ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY <model-package> /workspace/<model-package>
COPY ms-swift /workspace/ms-swift

WORKDIR /workspace/<model-package>
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

WORKDIR /workspace/ms-swift
RUN pip install -e .

WORKDIR /workspace
CMD ["/bin/bash"]
```

## 失败处理

- `docker images` 失败：检查 docker daemon
- build 失败：检查依赖冲突、网络、Dockerfile 语法
- import 测试失败：补充缺失包
- 镜像命名错误：立即修正为规范格式

## 进入下一阶段

Docker image 可用后，进入 `model_analysis`。
