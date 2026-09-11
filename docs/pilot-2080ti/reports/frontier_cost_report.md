# XAI01-04 试跑：前沿方法成本曲线

生成时间（UTC）：`2026-09-11T14:31:23.812555+00:00`

| 方法 | 参数 | 取值 | 墙钟(s) | 前向样本 | 显存峰值(MiB) | 状态 |
|---|---|---:|---:|---:|---:|---|
| ma_gig | num_steps | 4 | 0.5250 | 7.0000 | 1822.0000 | ok |
| ma_gig | num_steps | 8 | 0.8635 | 15.0000 | 1826.0000 | ok |
| ma_gig | num_steps | 16 | 1.8391 | 31.0000 | 1830.0000 | ok |
| fourier_shap | n_samples | 128 | 0.2315 | 128.0000 | 2418.0000 | ok |
| fourier_shap | n_samples | 512 | 0.6102 | 512.0000 | 2418.0000 | ok |
| fourier_shap | n_samples | 2048 | 2.2399 | 2048.0000 | 2418.0000 | ok |
| ma_gig | num_steps | 32 | 3.9394 | 63.0000 | 1840.0000 | ok |
| ma_gig | num_steps | 64 | 7.7547 | 127.0000 | 1858.0000 | ok |
| ma_gig | num_steps | 128 | 15.7200 | 255.0000 | 2196.0000 | ok |
| ma_gig | num_steps | 200 | 24.7379 | 399.0000 | 2708.0000 | ok |

```bash
cd /home/hycx233/Courses/machine-learning/xai01-04-pilot-20260911
.venv/bin/python pilot_frontier.py --cost --resume
```
