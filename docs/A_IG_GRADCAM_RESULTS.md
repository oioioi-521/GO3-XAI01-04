# A：IG / Grad-CAM 正式实验结果

更新时间：2026-09-20。

## 结论摘要

成员 A 的主体矩阵已完成忠实性与效率正式运行：

```text
2 种方法 × 3 个模型 × 2 个数据集 = 12 个正式单元
每单元 460 张，共 5,520 次归因
```

这里的“完成”不包含稳定性。按全组“忠实性、稳定性、效率”三项均齐全的严格口径，目前 36 个主体单元中仍有 0 个完全闭环；A 已为其中 12 个单元提供前两项结果。

在当前统一的 raw MoRF 口径下，Integrated Gradients（IG）在 6 个“模型 × 数据集”组合中均取得更低的删除 AUC，Grad-CAM 则在全部组合中明显更快。IG 的平均 raw AUC 相比 Grad-CAM 低 1.22～11.20 倍，但归因耗时高 9.71～29.75 倍。这说明本任务包内存在稳定的忠实性—效率权衡，但在稳定性指标冻结前不能声称已完成三维 Pareto 结论。

报告/PPT 可直接使用的汇总图：[`../analysis/a_ig_gradcam_summary.png`](../analysis/a_ig_gradcam_summary.png)。作图数据和复现脚本分别为 [`../analysis/a_ig_gradcam_summary.csv`](../analysis/a_ig_gradcam_summary.csv) 与 [`../analysis/plot_a_results.py`](../analysis/plot_a_results.py)。

## 固定实验口径

- 数据：冻结的 ImageNet 与 VOC `eval` 划分，各 460 张。
- 归因 target：metadata 中的真值类别；VOC 冻结评估图为单标签，模型仍按 20 类多标签 sigmoid 语义加载。
- IG：50 步 Gauss-Legendre，`internal_batch_size: 10`，归一化 RGB 黑色 baseline。
- Grad-CAM：VGG16 `features.28`、ResNet50 `layer4.2`、DenseNet121 `features.norm5`。
- 忠实性：`faithfulness_morf_auc_raw`，使用 21 个删除比例点 `0, 0.05, ..., 1`；ImageNet 对 true-target softmax、VOC 对 true-target sigmoid 积分，值域 `[0,1]`，越低越好。
- 效率：每个单元先执行 1 次不落盘的归因预热；记录归因 wrapper 的 wall time。时间包含 wrapper 内的 target 合法性前向，排除单独 prediction、MoRF 评分与文件 I/O。
- 产物：每次归因同时保存可视化 PNG 和未经 8-bit 量化的 float32 NPY。

正式运行提交为 `ae5c54a034e0b3ff5a255c17fe36c8042dcc8c09`。环境为 Python 3.13.9、PyTorch 2.8.0、torchvision 0.23.0、Captum 0.9.0、CUDA 12.8、NVIDIA driver 610.57.04、RTX 2080 Ti。

## 正式汇总

均值和标准差均基于每单元 460 张图；AUC 越低越好。

| 数据集 | 方法 | 模型 | raw MoRF AUC（mean ± SD） | 归因时间 ms（mean ± SD） | 配置哈希 |
| --- | --- | --- | ---: | ---: | --- |
| ImageNet | IG | VGG16 | 0.021651 ± 0.016341 | 298.087 ± 0.605 | `808492b7aa38` |
| ImageNet | IG | ResNet50 | 0.038372 ± 0.052087 | 168.759 ± 0.340 | `0d5aefac053b` |
| ImageNet | IG | DenseNet121 | 0.044469 ± 0.079427 | 191.834 ± 0.695 | `6466168496b6` |
| ImageNet | Grad-CAM | VGG16 | 0.242406 ± 0.234131 | 10.094 ± 0.082 | `6b629004f09a` |
| ImageNet | Grad-CAM | ResNet50 | 0.087643 ± 0.068742 | 8.110 ± 0.230 | `b103251d4a4d` |
| ImageNet | Grad-CAM | DenseNet121 | 0.056758 ± 0.060631 | 19.752 ± 0.338 | `45749d6c44aa` |
| VOC | IG | VGG16 | 0.113781 ± 0.104179 | 297.530 ± 0.602 | `b0f3058aa805` |
| VOC | IG | ResNet50 | 0.441888 ± 0.097023 | 168.390 ± 0.356 | `2a7264315a4d` |
| VOC | IG | DenseNet121 | 0.326695 ± 0.177955 | 191.636 ± 0.751 | `a1cc02d5bbde` |
| VOC | Grad-CAM | VGG16 | 0.336448 ± 0.267803 | 10.000 ± 0.063 | `caf355cf4391` |
| VOC | Grad-CAM | ResNet50 | 0.538512 ± 0.132886 | 7.808 ± 0.255 | `6efaa935e970` |
| VOC | Grad-CAM | DenseNet121 | 0.436635 ± 0.208627 | 19.228 ± 0.262 | `7d7e2ed44019` |

模型预测只计算一次并由两种方法复用。ImageNet Top-1 为 VGG16 363/460（78.91%）、ResNet50 409/460（88.91%）、DenseNet121 371/460（80.65%）。VOC 表中记录的是冻结单标签目标与 20 维 sigmoid 分数 argmax 的命中率，不是多标签 mAP：VGG16 402/460（87.39%）、ResNet50 385/460（83.70%）、DenseNet121 387/460（84.13%）。

## 产物验收与溯源

两数据集分别独立验收，结果为：

- 每个数据集 5,520 条逐图指标、12 条单元汇总、1,380 条预测；合计 11,040 / 24 / 2,760。
- 每个数据集 2,760 张 PNG 与 2,760 份 float32 NPY；合计各 5,520 份。
- 所有逐图键唯一；所有指标、汇总和图均为有限值；raw AUC 与归因图均在 `[0,1]`；NPY 均为 `float32`、`224×224`。
- 12 份最新正式日志全部为 `processed=460, skipped=0, warmup_runs=1`，且与单元配置哈希一致。
- 保留了 4 张合法的全零 Grad-CAM，没有人工替换：ImageNet/VGG16 的 `ILSVRC2012_val_00010220`、`ILSVRC2012_val_00034515`，VOC/VGG16 的 `298`，VOC/DenseNet121 的 `2848`。

完整 `results/` 目录由 Git 忽略，已从实验台式机回传到本地集成工作树。对目录中每个文件按相对路径排序、先逐文件 SHA256、再对清单内容计算 SHA256，实验机与本机结果一致：

```text
4f3dba21c7f30d702e5f8ca5c4fd46bfc6c1ce9dca8528978a7a4528c4e8b90d
```

权重 SHA256：

| 数据集 | 模型 | SHA256 |
| --- | --- | --- |
| ImageNet | VGG16 | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` |
| ImageNet | ResNet50 | `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| ImageNet | DenseNet121 | `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395` |
| VOC | VGG16 | `61640717641bfc1e9ae5d6172de667ce45755e6f046f458c848f70cb4a6c9d58` |
| VOC | ResNet50 | `1e54867cf18ed02383ba329c944a9143871fc37fe268bec6e769a8d364ae134f` |
| VOC | DenseNet121 | `ab61772eaf3459842f4b1e34be9ec15062e02a940d521390d80df0371168feca` |

## 限制与后续输入

仓库当前只给出“稳定性”这一高层维度，没有冻结具体扰动、强度、重复次数、相似度公式或 metric 字段，`requirements.txt` 也没有锁定 Quantus。因此本任务没有擅自发明第三种指标。float32 原始归因图保证 raw MoRF 等排序型指标可以无损复算；稳定性通常还需要对扰动输入重新归因，仍应在全组冻结统一协议后执行。

在稳定性协议落地前，本文件只支持 IG 与 Grad-CAM 的阶段性忠实性—效率结论，不能用于最终三维 Pareto、RQ2 或 RQ3 的完整结论。
