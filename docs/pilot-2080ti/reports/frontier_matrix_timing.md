# XAI01-04 试跑：并入前沿方法后的完整用时大表

生成时间（UTC）：`2026-09-11T14:47:44.190156+00:00`

范围：PDF 候选 6 标准方法（Grad-CAM / KernelSHAP / IG / RISE / Ablation / LIME）+ 2 前沿方法（FourierShap / MA-GIG）
× 3 模型（VGG-16 / ResNet-50 / DenseNet-121）× 3 数据集（ImageNet / VOC2007 / CHNCXR）= **72 条件**。

- 单图总耗时含：原图归因 + 缓存热图上的 Insertion/Deletion（20 步）+ K=2 稳定性重新归因 + 预测与 IO；单位秒/图。
- 用时 = 单图总耗时 × 图像数 × 1.30（30% 工程余量）。
- **FourierShap** 取 512 采样；**MA-GIG** 取论文默认 200 步（单图约 24.7 s 归因）。
- 标注 * 的 ImageNet 行为外推：224×224 下计算量只取决于模型与方法，取同 (模型, 方法) 在两个已测数据集上的均值。

| 数据集 | 模型 | 方法 | 单图归因(s) | 单图总(s) | 100 图 | 500 图 |
|---|---|---|---:|---:|---:|---:|
| ImageNet * | VGG-16 | Grad-CAM | 0.008 | 0.170 | 22.2 秒 | 1.8 分 |
| ImageNet * | VGG-16 | KernelSHAP | 1.123 | 3.518 | 7.6 分 | 38.1 分 |
| ImageNet * | VGG-16 | IG | 0.129 | 0.520 | 67.6 秒 | 5.6 分 |
| ImageNet * | VGG-16 | RISE | 1.010 | 3.163 | 6.9 分 | 34.3 分 |
| ImageNet * | VGG-16 | Ablation | 0.106 | 0.457 | 59.4 秒 | 5.0 分 |
| ImageNet * | VGG-16 | LIME | 1.156 | 3.604 | 7.8 分 | 39.0 分 |
| ImageNet * | VGG-16 | FourierShap | 1.007 | 3.140 | 6.8 分 | 34.0 分 |
| ImageNet * | VGG-16 | MA-GIG（200 步） | 24.783 | 74.491 | 2.69 小时 | 13.45 小时 |
| ImageNet * | ResNet-50 | Grad-CAM | 0.012 | 0.111 | 14.4 秒 | 72.0 秒 |
| ImageNet * | ResNet-50 | KernelSHAP | 0.744 | 2.299 | 5.0 分 | 24.9 分 |
| ImageNet * | ResNet-50 | IG | 0.089 | 0.339 | 44.1 秒 | 3.7 分 |
| ImageNet * | ResNet-50 | RISE | 0.617 | 1.927 | 4.2 分 | 20.9 分 |
| ImageNet * | ResNet-50 | Ablation | 0.069 | 0.285 | 37.0 秒 | 3.1 分 |
| ImageNet * | ResNet-50 | LIME | 0.764 | 2.383 | 5.2 分 | 25.8 分 |
| ImageNet * | ResNet-50 | FourierShap | 0.617 | 1.913 | 4.1 分 | 20.7 分 |
| ImageNet * | ResNet-50 | MA-GIG（200 步） | 24.724 | 74.221 | 2.68 小时 | 13.40 小时 |
| ImageNet * | DenseNet-121 | Grad-CAM | 0.018 | 0.134 | 17.4 秒 | 86.9 秒 |
| ImageNet * | DenseNet-121 | KernelSHAP | 0.806 | 2.528 | 5.5 分 | 27.4 分 |
| ImageNet * | DenseNet-121 | IG | 0.100 | 0.379 | 49.3 秒 | 4.1 分 |
| ImageNet * | DenseNet-121 | RISE | 0.696 | 2.169 | 4.7 分 | 23.5 分 |
| ImageNet * | DenseNet-121 | Ablation | 0.083 | 0.336 | 43.6 秒 | 3.6 分 |
| ImageNet * | DenseNet-121 | LIME | 0.841 | 2.603 | 5.6 分 | 28.2 分 |
| ImageNet * | DenseNet-121 | FourierShap | 0.660 | 2.058 | 4.5 分 | 22.3 分 |
| ImageNet * | DenseNet-121 | MA-GIG（200 步） | 25.233 | 75.759 | 2.74 小时 | 13.68 小时 |
| VOC2007 | VGG-16 | Grad-CAM | 0.008 | 0.159 | 20.7 秒 | 1.7 分 |
| VOC2007 | VGG-16 | KernelSHAP | 1.140 | 3.578 | 7.8 分 | 38.8 分 |
| VOC2007 | VGG-16 | IG | 0.133 | 0.530 | 68.9 秒 | 5.7 分 |
| VOC2007 | VGG-16 | RISE | 1.032 | 3.229 | 7.0 分 | 35.0 分 |
| VOC2007 | VGG-16 | Ablation | 0.108 | 0.469 | 61.0 秒 | 5.1 分 |
| VOC2007 | VGG-16 | LIME | 1.179 | 3.677 | 8.0 分 | 39.8 分 |
| VOC2007 | VGG-16 | FourierShap | 1.020 | 3.167 | 6.9 分 | 34.3 分 |
| VOC2007 | VGG-16 | MA-GIG（200 步） | 24.794 | 74.523 | 2.69 小时 | 13.46 小时 |
| VOC2007 | ResNet-50 | Grad-CAM | 0.014 | 0.120 | 15.6 秒 | 78.0 秒 |
| VOC2007 | ResNet-50 | KernelSHAP | 0.778 | 2.356 | 5.1 分 | 25.5 分 |
| VOC2007 | ResNet-50 | IG | 0.090 | 0.344 | 44.7 秒 | 3.7 分 |
| VOC2007 | ResNet-50 | RISE | 0.628 | 1.958 | 4.2 分 | 21.2 分 |
| VOC2007 | ResNet-50 | Ablation | 0.070 | 0.290 | 37.7 秒 | 3.1 分 |
| VOC2007 | ResNet-50 | LIME | 0.774 | 2.417 | 5.2 分 | 26.2 分 |
| VOC2007 | ResNet-50 | FourierShap | 0.618 | 1.919 | 4.2 分 | 20.8 分 |
| VOC2007 | ResNet-50 | MA-GIG（200 步） | 24.701 | 74.140 | 2.68 小时 | 13.39 小时 |
| VOC2007 | DenseNet-121 | Grad-CAM | 0.019 | 0.138 | 17.9 秒 | 89.7 秒 |
| VOC2007 | DenseNet-121 | KernelSHAP | 0.826 | 2.581 | 5.6 分 | 28.0 分 |
| VOC2007 | DenseNet-121 | IG | 0.103 | 0.388 | 50.5 秒 | 4.2 分 |
| VOC2007 | DenseNet-121 | RISE | 0.712 | 2.212 | 4.8 分 | 24.0 分 |
| VOC2007 | DenseNet-121 | Ablation | 0.085 | 0.342 | 44.4 秒 | 3.7 分 |
| VOC2007 | DenseNet-121 | LIME | 0.848 | 2.622 | 5.7 分 | 28.4 分 |
| VOC2007 | DenseNet-121 | FourierShap | 0.660 | 2.064 | 4.5 分 | 22.4 分 |
| VOC2007 | DenseNet-121 | MA-GIG（200 步） | 25.267 | 75.850 | 2.74 小时 | 13.70 小时 |
| CHNCXR | VGG-16 | Grad-CAM | 0.009 | 0.181 | 23.6 秒 | 2.0 分 |
| CHNCXR | VGG-16 | KernelSHAP | 1.106 | 3.458 | 7.5 分 | 37.5 分 |
| CHNCXR | VGG-16 | IG | 0.126 | 0.509 | 66.2 秒 | 5.5 分 |
| CHNCXR | VGG-16 | RISE | 0.988 | 3.098 | 6.7 分 | 33.6 分 |
| CHNCXR | VGG-16 | Ablation | 0.104 | 0.446 | 57.9 秒 | 4.8 分 |
| CHNCXR | VGG-16 | LIME | 1.132 | 3.531 | 7.7 分 | 38.3 分 |
| CHNCXR | VGG-16 | FourierShap | 0.995 | 3.113 | 6.7 分 | 33.7 分 |
| CHNCXR | VGG-16 | MA-GIG（200 步） | 24.771 | 74.460 | 2.69 小时 | 13.44 小时 |
| CHNCXR | ResNet-50 | Grad-CAM | 0.010 | 0.102 | 13.2 秒 | 66.1 秒 |
| CHNCXR | ResNet-50 | KernelSHAP | 0.710 | 2.242 | 4.9 分 | 24.3 分 |
| CHNCXR | ResNet-50 | IG | 0.088 | 0.335 | 43.5 秒 | 3.6 分 |
| CHNCXR | ResNet-50 | RISE | 0.606 | 1.897 | 4.1 分 | 20.6 分 |
| CHNCXR | ResNet-50 | Ablation | 0.067 | 0.280 | 36.4 秒 | 3.0 分 |
| CHNCXR | ResNet-50 | LIME | 0.755 | 2.348 | 5.1 分 | 25.4 分 |
| CHNCXR | ResNet-50 | FourierShap | 0.616 | 1.908 | 4.1 分 | 20.7 分 |
| CHNCXR | ResNet-50 | MA-GIG（200 步） | 24.746 | 74.301 | 2.68 小时 | 13.42 小时 |
| CHNCXR | DenseNet-121 | Grad-CAM | 0.017 | 0.129 | 16.8 秒 | 84.0 秒 |
| CHNCXR | DenseNet-121 | KernelSHAP | 0.787 | 2.476 | 5.4 分 | 26.8 分 |
| CHNCXR | DenseNet-121 | IG | 0.098 | 0.369 | 48.0 秒 | 4.0 分 |
| CHNCXR | DenseNet-121 | RISE | 0.681 | 2.126 | 4.6 分 | 23.0 分 |
| CHNCXR | DenseNet-121 | Ablation | 0.081 | 0.329 | 42.8 秒 | 3.6 分 |
| CHNCXR | DenseNet-121 | LIME | 0.833 | 2.584 | 5.6 分 | 28.0 分 |
| CHNCXR | DenseNet-121 | FourierShap | 0.659 | 2.053 | 4.4 分 | 22.2 分 |
| CHNCXR | DenseNet-121 | MA-GIG（200 步） | 25.199 | 75.669 | 2.73 小时 | 13.66 小时 |

## 矩阵级合计（含 30% 工程余量）

| 范围 | 条件数 | 每图合计(s) | 100 图 | 500 图 | 1000 图 |
|---|---:|---:|---:|---:|---:|
| 标准 6 方法（PDF 主体，每方法 9 条件） | 54 | 80.78 | 2.92 小时 | 14.58 小时 | 29.17 小时 |
| 前沿 FourierShap 512 采样 | 9 | 21.34 | 46.2 分 | 3.85 小时 | 7.70 小时 |
| 前沿 MA-GIG 32 步 | 9 | 111.86 | 4.04 小时 | 20.20 小时 | 40.39 小时 |
| 前沿 MA-GIG 200 步（论文默认） | 9 | 673.41 | 24.32 小时 | 121.59 小时 | 243.18 小时 |
| 前沿 2 方法合计（MA-GIG 32 步） | 18 | 133.19 | 4.81 小时 | 24.05 小时 | 48.10 小时 |
| **合计 72 条件（MA-GIG 32 步）** | 72 | 213.97 | 7.73 小时 | 38.63 小时 | 77.27 小时 |
| **合计 72 条件（MA-GIG 200 步）** | 72 | 775.53 | 28.01 小时 | 140.03 小时 | 280.05 小时 |

## 前沿方法成本曲线（ResNet-50，单图，秒）

| 方法 | 参数 | 取值 | 归因墙钟(s) | 每单位成本 |
|---|---|---:|---:|---:|
| MA-GIG | num_steps | 4 | 0.525 | 131 ms/步 |
| MA-GIG | num_steps | 8 | 0.864 | 108 ms/步 |
| MA-GIG | num_steps | 16 | 1.839 | 115 ms/步 |
| MA-GIG | num_steps | 32 | 3.939 | 123 ms/步 |
| MA-GIG | num_steps | 64 | 7.755 | 121 ms/步 |
| MA-GIG | num_steps | 128 | 15.720 | 123 ms/步 |
| MA-GIG | num_steps | 200 | 24.738 | 124 ms/步 |
| FourierShap | n_samples | 128 | 0.232 | 1808.94 µs/采样 |
| FourierShap | n_samples | 512 | 0.610 | 1191.81 µs/采样 |
| FourierShap | n_samples | 2048 | 2.240 | 1093.68 µs/采样 |

## 前沿方法质量（12 条件均值，OOD 管线诊断，非任务质量）

| 数据集 | 模型 | 方法 | Insertion AUC | Deletion AUC | 稳定性 cosine |
|---|---|---|---:|---:|---:|
| VOC2007 | VGG-16 | FourierShap | 0.262 | 0.043 | 0.946 |
| VOC2007 | VGG-16 | MA-GIG | 0.025 | 0.017 | 0.006 |
| VOC2007 | ResNet-50 | FourierShap | 0.333 | 0.061 | 0.936 |
| VOC2007 | ResNet-50 | MA-GIG | 0.098 | 0.012 | 0.003 |
| VOC2007 | DenseNet-121 | FourierShap | 0.233 | 0.029 | 0.953 |
| VOC2007 | DenseNet-121 | MA-GIG | 0.080 | 0.012 | 0.007 |
| CHNCXR | VGG-16 | FourierShap | 0.051 | 0.025 | 0.659 |
| CHNCXR | VGG-16 | MA-GIG | 0.006 | 0.008 | 0.022 |
| CHNCXR | ResNet-50 | FourierShap | 0.085 | 0.010 | 0.829 |
| CHNCXR | ResNet-50 | MA-GIG | 0.004 | 0.004 | 0.006 |
| CHNCXR | DenseNet-121 | FourierShap | 0.087 | 0.019 | 0.903 |
| CHNCXR | DenseNet-121 | MA-GIG | 0.066 | 0.131 | 0.004 |

## 结论

- **FourierShap（512 采样）**：单图总耗时约 1.9–3.2 s（与 RISE/KernelSHAP/LIME 同量级），Insertion AUC 与 LIME/KernelSHAP 相当、稳定性 0.66–0.95，**性价比最高的候选前沿方法**。
- **MA-GIG**：成本随步数近线性（约 0.12 s/步）。32 步单图总耗时约 12–13 s；论文默认 200 步单图归因约 24.7 s、单图总耗时约 75 s，在 72 条件矩阵下 500 图约需 140.03 小时，**不具备可操作性**。
- 若只保留 6 标准方法 + FourierShap（MA-GIG 不进矩阵），则完整矩阵为 54+9=63 条件，500 图约 18.44 小时；仍显著低于含 MA-GIG 的 72 条件版本。
- 在本 OOD 试跑设置下，MA-GIG 的隐空间路径积分不收敛（稳定性≈0），质量结论不足以支持其入主矩阵。
- 注：FourierShap 与 MA-GIG 均为试跑级操作化，不是作者原实现，详见 `frontier_conclusions.md`。
