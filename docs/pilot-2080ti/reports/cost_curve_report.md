# XAI01-04 试跑 P3：成本曲线

生成时间（UTC）：`2026-09-11T14:08:55.401820+00:00`

数据集 `voc2007` 样本 `000246`，模型 `resnet50`，单张 224x224 输入，同步墙钟计时。

| 方法 | 参数 | 取值 | 墙钟(s) | CUDA event(s) | 前向样本数 | 显存峰值(MiB) | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| ig | n_steps | 16 | 0.1063 | 0.1063 | 16.0000 | 1682.0000 | ok |
| ig | n_steps | 32 | 0.0957 | 0.0956 | 32.0000 | 1682.0000 | ok |
| ig | n_steps | 64 | 0.1816 | 0.1815 | 64.0000 | 1682.0000 | ok |
| rise | n_masks | 128 | 0.1490 | 0.1490 | 128.0000 | 1682.0000 | ok |
| rise | n_masks | 512 | 0.5952 | 0.5951 | 512.0000 | 1682.0000 | ok |
| rise | n_masks | 2048 | 2.3799 | 2.3796 | 2048.0000 | 1682.0000 | ok |
| kernelshap | n_samples | 128 | 0.8541 | 0.8540 | 128.0000 | 1684.0000 | ok |
| kernelshap | n_samples | 512 | 0.7012 | 0.7011 | 512.0000 | 1684.0000 | ok |
| kernelshap | n_samples | 2048 | 2.7828 | 2.7825 | 2048.0000 | 1690.0000 | ok |
| lime | n_samples | 128 | 0.2236 | 0.2236 | 128.0000 | 1690.0000 | ok |
| lime | n_samples | 512 | 0.7399 | 0.7398 | 512.0000 | 1690.0000 | ok |
| lime | n_samples | 2048 | 2.9648 | 2.9645 | 2048.0000 | 1692.0000 | ok |
| ablation | grid | 7 | 0.0723 | 0.0722 | 50.0000 | 1692.0000 | ok |
| ablation | grid | 8 | 0.0818 | 0.0817 | 65.0000 | 1692.0000 | ok |
| rise | internal_batch_size | 8 | 0.6772 | 0.6771 | 512.0000 | 1692.0000 | ok |
| rise | internal_batch_size | 16 | 0.5974 | 0.5973 | 512.0000 | 1692.0000 | ok |
| rise | internal_batch_size | 32 | 0.5698 | 0.5697 | 512.0000 | 1986.0000 | ok |
| kernelshap | internal_batch_size | 8 | 0.7639 | 0.7638 | 512.0000 | 1986.0000 | ok |
| kernelshap | internal_batch_size | 16 | 0.7003 | 0.7002 | 512.0000 | 1986.0000 | ok |
| kernelshap | internal_batch_size | 32 | 0.6698 | 0.6697 | 512.0000 | 1988.0000 | ok |

> 解读边界：每个配置只测了同一张图的一次，适合判断成本量级，不适合拟合精确缩放曲线。512→2048 时 RISE / KernelSHAP / LIME 都约增至 4 倍；KernelSHAP 的 128 样本点反而慢于 512，属于需要重复测量才能解释的波动。internal batch 16→32 的加速约 4–5%。

## 复现命令

```bash
cd /home/hycx233/Courses/machine-learning/xai01-04-pilot-20260911
.venv/bin/python pilot_eval.py --cost-curve --resume --global-deadline-sec 1200 --task-timeout-sec 300
```
