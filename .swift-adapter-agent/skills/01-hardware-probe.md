# 01 - Hardware Probe

## 目标

探测 GPU 资源，但**最多只使用 7 张 GPU** 进行训练。

## Agent Checklist

- [ ] 从 `input.json` 读取 `max_gpus`（默认 7）
- [ ] 调用 executor：`python .swift-adapter-agent/executors/hardware_probe.py --max-gpus {max_gpus} --output outputs/{run_id}/hardware.json`
- [ ] 读取 `hardware.json`
- [ ] 确认可用 GPU 数量 `usable_gpu_count` <= `max_gpus`
- [ ] 检查是否有其他进程占用显存（可选：`nvidia-smi`）
- [ ] 更新 `pipeline_state.json`

## 输出 JSON

```json
{
  "passed": true,
  "gpu_count": 8,
  "max_gpus": 7,
  "usable_gpu_count": 7,
  "usable_gpus": [
    {"id": 0, "name": "NVIDIA A800-SXM4-80GB", "memory_total_gb": 80.0, "compute_capability": "8.0"},
    {"id": 1, "name": "NVIDIA A800-SXM4-80GB", "memory_total_gb": 80.0, "compute_capability": "8.0"}
  ],
  "cuda_version_pytorch": "12.1"
}
```

## 约束

- 即使机器有 8 张 GPU，训练脚本也最多使用 7 张。
- `CUDA_VISIBLE_DEVICES` 最多包含 7 个 id。
- `NPROC_PER_NODE` 最多为 7。

## 决策信息

- `usable_gpu_count` 决定训练脚本中 `NPROC_PER_NODE`
- 单卡显存影响 batch size 选择
- CUDA 版本影响 Dockerfile 基础镜像

## 失败处理

- 没有 GPU：停止 pipeline
- 可用 GPU 数量 < 1：停止 pipeline
