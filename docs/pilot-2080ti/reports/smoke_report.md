# XAI01-04 试跑 P0：合成图首通（初次尝试）

生成时间（UTC）：`2026-09-11T13:37:41.018256+00:00`

合成图校准仅用于计时，不是 ImageNet、VOC 或 CHNCXR 的质量结果。

本次初次尝试为 5/6 成功；RISE 因布尔张量双线性上采样报错。修正后的完整复核见 `rise_smoke_report.md`（6/6 成功）。

## 环境

- 设备：`cuda`；GPU：`NVIDIA GeForce RTX 2080 Ti`
- PyTorch / torchvision / CUDA runtime：`2.8.0` / `0.23.0` / `12.8`
- 官方权重下载目录：`/home/hycx233/Courses/machine-learning/xai01-04-pilot-20260911/weights_cache`
- 计时副本记录总数：`6`

## 方法 × 模型计时汇总

| 模型 | 方法 | 成功/总计 | 墙钟均值(s) | CUDA event 均值(s) | 前向样本数均值 | CUDA 保留峰值(MiB) | GPU 显存峰值(MiB) | CPU RSS 峰值(MiB) | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| resnet50 | gradcam | 1/1 | 0.0123 | 0.0122 | 1.0000 | 328.0000 | 3001.7500 | 1168.9023 | {'ok': 1} |
| resnet50 | kernelshap | 1/1 | 0.0344 | 0.0344 | 16.0000 | 226.0000 | 2899.8125 | 1431.2695 | {'ok': 1} |
| resnet50 | ig | 1/1 | 0.0201 | 0.0200 | 4.0000 | 496.0000 | 3173.8125 | 1452.5938 | {'ok': 1} |
| resnet50 | rise | 0/1 | null | null | null | null | null | null | {'error': 1} |
| resnet50 | ablation | 1/1 | 0.0765 | 0.0765 | 50.0000 | 252.0000 | 2929.8125 | 1488.0547 | {'ok': 1} |
| resnet50 | lime | 1/1 | 0.0299 | 0.0299 | 16.0000 | 226.0000 | 2903.8125 | 1565.0469 | {'ok': 1} |

## 复现命令

```bash
cd /home/hycx233/Courses/machine-learning/xai01-04-pilot-20260911
.venv/bin/python pilot_runner.py --smoke --resume --records smoke_records.jsonl --summary smoke_summary.json --report smoke_report.md
```

原始记录追加写入 `smoke_records.jsonl`；聚合 JSON 为 `smoke_summary.json`。
