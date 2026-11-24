# A：IG / Grad-CAM 正式稳定性结果

2026-10-01：**12/12 正式单元完成并通过验收**。整批于 22:30（Asia/Shanghai）结束，
退出码 `0`；A 的忠实性、稳定性和效率三项指标现已齐全。

## 协议与环境

- 执行代码：`a96749ac153c80f01e7cd32c1384f17a331eca5b`，GPG 签名有效；
- 协议：`rgb-gaussian-spearman-v1`，反归一化 RGB 域高斯噪声 `sigma=0.005`，
  裁剪至 `[0,1]`，每图 5 次，metadata 真值 target，按数据集/图片/重复配对 seed；
- 原模型、checkpoint 和归因参数不变；仅补跑稳定性，不重跑 MoRF；
- 环境：Python 3.13.9、PyTorch 2.8.0、torchvision 0.23.0、Captum 0.9.0、
  SciPy 1.17.1、CUDA、RTX 2080 Ti；
- 协议 YAML SHA-256：`324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf`。

## 正式汇总

下表是 460 张图的 `stability_spearman` 均值 ± 样本标准差，包含退化重复的 0 分。
有效率是 `stability_valid_rate` 的逐图均值；每个单元共 2,300 次扰动。

| 数据集 | 模型 | IG Spearman | Grad-CAM Spearman | IG 有效率 | Grad-CAM 有效率 | Grad-CAM 退化重复 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ImageNet | ResNet50 | 0.673541 ± 0.174047 | 0.989534 ± 0.014436 | 100.0000% | 100.0000% | 0 |
| ImageNet | DenseNet121 | 0.761753 ± 0.154024 | 0.998853 ± 0.002903 | 100.0000% | 100.0000% | 0 |
| ImageNet | VGG16 | 0.796449 ± 0.127498 | 0.931388 ± 0.099560 | 100.0000% | 99.4348% | 13 |
| VOC | ResNet50 | 0.573837 ± 0.167400 | 0.977126 ± 0.036252 | 100.0000% | 100.0000% | 0 |
| VOC | DenseNet121 | 0.713087 ± 0.144333 | 0.992261 ± 0.067179 | 100.0000% | 99.5652% | 10 |
| VOC | VGG16 | 0.777346 ± 0.125627 | 0.959514 ± 0.059016 | 100.0000% | 99.7826% | 5 |

逐单元完整精度、配置哈希及退化数见
[`../analysis/a_stability_summary.csv`](../analysis/a_stability_summary.csv)。

在本协议和这 6 个模型/数据集组合中，Grad-CAM 的平均稳定性均高于 IG。
结合原忠实性/效率结果，IG 的 raw MoRF AUC 更低，Grad-CAM 更快且对当前小幅 RGB
扰动更稳定。该比较适用于本协议及 A 的方法范围；全组最终排名和 RQ1–RQ3 分析
仍需其余方法结果齐全。

Grad-CAM 共 28 次退化重复：ImageNet/VGG16 为 `both_constant=6`、
`reference_constant=4`、`candidate_constant=3`；VOC/DenseNet121 为
`both_constant=5`、`candidate_constant=5`；VOC/VGG16 为 `both_constant=3`、
`reference_constant=2`。这些重复的 raw Spearman 留空、正式 score 为 0，保留有效率
及原常量图。IG 无退化重复。未把 2026-09-22 pilot 数据混入正式结果。

## 验收与产物

- 12 个单元首次均为 `processed=460, skipped=0`；立即续跑均为
  `processed=0, skipped=460`，独立单图门禁均通过；
- 新增 11,040 条稳定性逐图指标、24 条稳定性汇总、27,600 条逐重复 trace；
  合并基础结果后 `per_image.csv` 共 22,080 行，`units.csv` 共 48 行；
- ID、target、重复号、seed、协议/配置身份、退化计分、逐图与单元聚合均复核通过；
- 原忠实性/效率行及基础配置哈希、预测表、5,520 份参考 NPY 保持不变；
- 旧图经重算核对后增加 5,520 份 `verified_recomputation` provenance sidecar；
- 本地交接包经过 ZIP CRC、整包 SHA-256 和 5,717 个包内文件逐项大小/哈希校验，
  与实验机一致。执行期间采用原结果独立副本，原实验目录保持不变。

轻量交接包：`A_STABILITY_EVAL_LIGHT_20261001.zip`，字节数 `7417668`，
SHA-256：

```text
69a4b1962919d97f59c9d390017c636da3118e16bcfb3f9899ed5fdb849852d1
```

本地位置为 `results/a_stability_eval_20261001/`。包中包含统一 CSV、正式 trace、
基础和稳定性日志、配置快照、续跑状态、单图门禁、provenance、基础表备份及完整验收
回执；参考 NPY 沿用既有正式结果，通过 sidecar 中的 `map_sha256` 核对。
包内 `ARTIFACT_MANIFEST.json` 列出全部文件的大小和 SHA-256。

原跑批回执：`results/a_stability_batches/20261001T132233566810Z/validation_report.json`；
完成后独立复核回执：`results/a_stability_final_validation_20261001.json`。
文件名中的 UTC 时间与正文中的 Asia/Shanghai 完成时间使用各自明确的时区。

复现和恢复入口见 [A_STABILITY_RUNBOOK.md](A_STABILITY_RUNBOOK.md)。
