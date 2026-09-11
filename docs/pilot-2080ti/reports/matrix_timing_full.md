# XAI01-04 完整候选矩阵用时表

生成时间（UTC）：`2026-09-11T14:14:07.587310+00:00`

矩阵来源：`project-pool.pdf` 第 63–65 页 XAI01-04 的建议实验空间，取全部候选：
方法 6（Grad-CAM / IG / KernelSHAP / RISE / LIME / Ablation）× 模型 3（VGG-16 / ResNet-50 / DenseNet-121）× 数据集 3（ImageNet / VOC / CHNCXR）= **54 个条件**。

单图成本为实测值，包含原始归因 + 对缓存热图的一次 Insertion/Deletion（20 步）+ K=2 稳定性重新归因（sigma=0.01）+ 预测与 IO；单位：秒/图。
用时 = 单图成本 × 图像数 × 1.30（30% 工程余量）。

> 说明：VOC2007 与 CHNCXR 为本次实测；**ImageNet 用 * 标注，为同 (模型, 方法) 在两个已测数据集上的均值外推**——
> 因为 224×224 输入下前向/反向计算量只取决于模型与方法，与图像来自哪个数据集无关。

## 逐条件用时（N=100 / N=500）

| 数据集 | 模型 | 方法 | 单图总(s) | 仅归因(s) | 100 图用时 | 500 图用时 |
|---|---|---|---:|---:|---:|---:|
| ImageNet * | VGG-16 | Grad-CAM | 0.170 | 0.008 | 22.2 秒 | 1.8 分 |
| ImageNet * | VGG-16 | IG | 0.520 | 0.129 | 67.6 秒 | 5.6 分 |
| ImageNet * | VGG-16 | KernelSHAP | 3.518 | 1.123 | 7.6 分 | 38.1 分 |
| ImageNet * | VGG-16 | RISE | 3.163 | 1.010 | 6.9 分 | 34.3 分 |
| ImageNet * | VGG-16 | LIME | 3.604 | 1.156 | 7.8 分 | 39.0 分 |
| ImageNet * | VGG-16 | Ablation | 0.457 | 0.106 | 59.4 秒 | 5.0 分 |
| ImageNet * | ResNet-50 | Grad-CAM | 0.111 | 0.012 | 14.4 秒 | 72.0 秒 |
| ImageNet * | ResNet-50 | IG | 0.339 | 0.089 | 44.1 秒 | 3.7 分 |
| ImageNet * | ResNet-50 | KernelSHAP | 2.299 | 0.744 | 5.0 分 | 24.9 分 |
| ImageNet * | ResNet-50 | RISE | 1.927 | 0.617 | 4.2 分 | 20.9 分 |
| ImageNet * | ResNet-50 | LIME | 2.383 | 0.764 | 5.2 分 | 25.8 分 |
| ImageNet * | ResNet-50 | Ablation | 0.285 | 0.069 | 37.0 秒 | 3.1 分 |
| ImageNet * | DenseNet-121 | Grad-CAM | 0.134 | 0.018 | 17.4 秒 | 86.9 秒 |
| ImageNet * | DenseNet-121 | IG | 0.379 | 0.100 | 49.3 秒 | 4.1 分 |
| ImageNet * | DenseNet-121 | KernelSHAP | 2.528 | 0.806 | 5.5 分 | 27.4 分 |
| ImageNet * | DenseNet-121 | RISE | 2.169 | 0.696 | 4.7 分 | 23.5 分 |
| ImageNet * | DenseNet-121 | LIME | 2.603 | 0.841 | 5.6 分 | 28.2 分 |
| ImageNet * | DenseNet-121 | Ablation | 0.336 | 0.083 | 43.6 秒 | 3.6 分 |
| VOC2007 | VGG-16 | Grad-CAM | 0.159 | 0.008 | 20.7 秒 | 1.7 分 |
| VOC2007 | VGG-16 | IG | 0.530 | 0.133 | 68.9 秒 | 5.7 分 |
| VOC2007 | VGG-16 | KernelSHAP | 3.578 | 1.140 | 7.8 分 | 38.8 分 |
| VOC2007 | VGG-16 | RISE | 3.229 | 1.032 | 7.0 分 | 35.0 分 |
| VOC2007 | VGG-16 | LIME | 3.677 | 1.179 | 8.0 分 | 39.8 分 |
| VOC2007 | VGG-16 | Ablation | 0.469 | 0.108 | 61.0 秒 | 5.1 分 |
| VOC2007 | ResNet-50 | Grad-CAM | 0.120 | 0.014 | 15.6 秒 | 78.0 秒 |
| VOC2007 | ResNet-50 | IG | 0.344 | 0.090 | 44.7 秒 | 3.7 分 |
| VOC2007 | ResNet-50 | KernelSHAP | 2.356 | 0.778 | 5.1 分 | 25.5 分 |
| VOC2007 | ResNet-50 | RISE | 1.958 | 0.628 | 4.2 分 | 21.2 分 |
| VOC2007 | ResNet-50 | LIME | 2.417 | 0.774 | 5.2 分 | 26.2 分 |
| VOC2007 | ResNet-50 | Ablation | 0.290 | 0.070 | 37.7 秒 | 3.1 分 |
| VOC2007 | DenseNet-121 | Grad-CAM | 0.138 | 0.019 | 17.9 秒 | 89.7 秒 |
| VOC2007 | DenseNet-121 | IG | 0.388 | 0.103 | 50.5 秒 | 4.2 分 |
| VOC2007 | DenseNet-121 | KernelSHAP | 2.581 | 0.826 | 5.6 分 | 28.0 分 |
| VOC2007 | DenseNet-121 | RISE | 2.212 | 0.712 | 4.8 分 | 24.0 分 |
| VOC2007 | DenseNet-121 | LIME | 2.622 | 0.848 | 5.7 分 | 28.4 分 |
| VOC2007 | DenseNet-121 | Ablation | 0.342 | 0.085 | 44.4 秒 | 3.7 分 |
| CHNCXR-Shenzhen | VGG-16 | Grad-CAM | 0.181 | 0.009 | 23.6 秒 | 2.0 分 |
| CHNCXR-Shenzhen | VGG-16 | IG | 0.509 | 0.126 | 66.2 秒 | 5.5 分 |
| CHNCXR-Shenzhen | VGG-16 | KernelSHAP | 3.458 | 1.106 | 7.5 分 | 37.5 分 |
| CHNCXR-Shenzhen | VGG-16 | RISE | 3.098 | 0.988 | 6.7 分 | 33.6 分 |
| CHNCXR-Shenzhen | VGG-16 | LIME | 3.531 | 1.132 | 7.7 分 | 38.3 分 |
| CHNCXR-Shenzhen | VGG-16 | Ablation | 0.446 | 0.104 | 57.9 秒 | 4.8 分 |
| CHNCXR-Shenzhen | ResNet-50 | Grad-CAM | 0.102 | 0.010 | 13.2 秒 | 66.1 秒 |
| CHNCXR-Shenzhen | ResNet-50 | IG | 0.335 | 0.088 | 43.5 秒 | 3.6 分 |
| CHNCXR-Shenzhen | ResNet-50 | KernelSHAP | 2.242 | 0.710 | 4.9 分 | 24.3 分 |
| CHNCXR-Shenzhen | ResNet-50 | RISE | 1.897 | 0.606 | 4.1 分 | 20.6 分 |
| CHNCXR-Shenzhen | ResNet-50 | LIME | 2.348 | 0.755 | 5.1 分 | 25.4 分 |
| CHNCXR-Shenzhen | ResNet-50 | Ablation | 0.280 | 0.067 | 36.4 秒 | 3.0 分 |
| CHNCXR-Shenzhen | DenseNet-121 | Grad-CAM | 0.129 | 0.017 | 16.8 秒 | 84.0 秒 |
| CHNCXR-Shenzhen | DenseNet-121 | IG | 0.369 | 0.098 | 48.0 秒 | 4.0 分 |
| CHNCXR-Shenzhen | DenseNet-121 | KernelSHAP | 2.476 | 0.787 | 5.4 分 | 26.8 分 |
| CHNCXR-Shenzhen | DenseNet-121 | RISE | 2.126 | 0.681 | 4.6 分 | 23.0 分 |
| CHNCXR-Shenzhen | DenseNet-121 | LIME | 2.584 | 0.833 | 5.6 分 | 28.0 分 |
| CHNCXR-Shenzhen | DenseNet-121 | Ablation | 0.329 | 0.081 | 42.8 秒 | 3.6 分 |

## 矩阵级合计

| 范围 | 条件数 | 每图合计(s) | 100 图 | 500 图 | 1000 图 |
|---|---:|---:|---:|---:|---:|
| 本次实测矩阵（VOC+CHNCXR） | 36 | 53.85 | 1.94 小时 | 9.72 小时 | 19.45 小时 |
| PDF 完整矩阵（ImageNet+VOC+CHNCXR） | 54 | 80.78 | 2.92 小时 | 14.58 小时 | 29.17 小时 |

## 推荐主矩阵（PLAN 第 6 节：4 方法 × 2 模型 × 3 数据集 = 24 条件）

方法 Grad-CAM / IG / KernelSHAP / RISE，模型 VGG-16 / ResNet-50，数据集 ImageNet / VOC / CHNCXR。

| 图像数（每条件） | 用时（含 30% 余量） |
|---:|---:|
| 100 | 78.3 分 |
| 500 | 6.53 小时 |
| 1000 | 13.05 小时 |

## 结论

- PDF 完整 54 条件：每图合计 80.8 秒；100 图约 2.92 小时，500 图约 14.58 小时。
- 实测 36 条件：100 图约 1.94 小时，500 图约 9.72 小时。
- 推荐主矩阵 24 条件：100 图约 78.3 分，500 图约 6.53 小时。
- 昂贵方法（RISE/KernelSHAP/LIME）占矩阵绝大部分时间；K 是比图像数更陡的杠杆（K 2→10 约为 3.5 倍）。
- 全 54 条件在 500 图上约半天 GPU 墙钟，属于一次夜间批量可完成的范围。
