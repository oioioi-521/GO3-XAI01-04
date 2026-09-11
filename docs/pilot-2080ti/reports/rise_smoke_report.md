# XAI01-04 试跑：合成图计时校准

生成时间（UTC）：`2026-09-11T13:37:41.018256+00:00`

合成图校准仅用于计时，不是 ImageNet、VOC 或 CHNCXR 的质量结果。

## 环境

- 设备：`cuda`；GPU：`NVIDIA GeForce RTX 2080 Ti`
- PyTorch / torchvision / CUDA runtime：`2.8.0` / `0.23.0` / `12.8`
- 官方权重下载目录：`/home/hycx233/Courses/machine-learning/xai01-04-pilot-20260911/weights_cache`
- 计时副本记录总数：`6`

## 方法 × 模型计时汇总

| 模型 | 方法 | 成功/总计 | 墙钟均值(s) | CUDA event 均值(s) | 前向样本数均值 | CUDA 保留峰值(MiB) | GPU 显存峰值(MiB) | CPU RSS 峰值(MiB) | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| resnet50 | gradcam | 1/1 | 0.0121 | 0.0120 | 1.0000 | 328.0000 | 3001.8125 | 1169.1367 | {'ok': 1} |
| resnet50 | kernelshap | 1/1 | 0.0342 | 0.0342 | 16.0000 | 226.0000 | 2899.8125 | 1431.6094 | {'ok': 1} |
| resnet50 | ig | 1/1 | 0.0197 | 0.0196 | 4.0000 | 496.0000 | 3173.8125 | 1452.9492 | {'ok': 1} |
| resnet50 | rise | 1/1 | 0.0280 | 0.0278 | 16.0000 | 246.0000 | 2923.8125 | 1453.2539 | {'ok': 1} |
| resnet50 | ablation | 1/1 | 0.0744 | 0.0743 | 50.0000 | 252.0000 | 2929.8125 | 1485.6211 | {'ok': 1} |
| resnet50 | lime | 1/1 | 0.0305 | 0.0305 | 16.0000 | 226.0000 | 2903.8125 | 1562.7500 | {'ok': 1} |

## 复现命令

```bash
cd /home/hycx233/Courses/machine-learning/xai01-04-pilot-20260911
.venv/bin/python pilot_runner.py --calibration --resume --global-deadline-sec 900 --task-timeout-sec 180
```

原始记录追加写入 `calibration_records.jsonl`；聚合 JSON 为 `calibration_summary.json`。
