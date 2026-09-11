# XAI01-04 试跑结果索引

生成时间（UTC）：`2026-09-11T14:45:11.113044+00:00`

主机：RTX 2080 Ti 11 GiB（桌面显示卡），i7-8700K，46 GiB 内存。PyTorch 2.8.0 / torchvision 0.23.0 / CUDA runtime 12.8。

## 阶段状态

| 阶段 | 状态 | 记录数 | 主要交付物 |
|---|---|---:|---|
| P0 环境与首通 | 完成 | - | `environment_snapshot.json`、`smoke_records.jsonl`、`smoke_report.md`、`rise_smoke_*` |
| P1 6×3 合成图校准 | 完成 | 54 成功 | `calibration_records.jsonl`、`calibration_summary.json`、`calibration_report.md` |
| P2 小规模端到端评价 | 完成 | 576 成功 / 576 总计 | `eval_records.jsonl`、`eval_summary.json`、`eval_report.md`、`eval_per_image.csv`、`eval_units.csv` |
| P3 采样成本曲线 | 完成 | 20 | `cost_curve_records.jsonl`、`cost_curve_summary.json`、`cost_curve_report.md` |
| P4 随机方法种子复核 | 完成 | 24 | `recheck_records.jsonl`、`recheck_summary.json`、`recheck_report.md` |
| P5 前沿方法探针（MA-GIG / FourierShap） | 完成 | 96 成功 / 96 | `frontier_records.jsonl`、`frontier_summary.json`、`frontier_report.md`、`frontier_units.csv`、`frontier_cost_report.md` |
| 成本外推 | 完成 | - | `extrapolation.md`、`extrapolation.json` |

## 协议摘要

- P2 规模：2 数据集 × 3 模型 × 6 方法 × 16 图 = 576 个单元，全部 `status=ok`。
- 归因目标取模型在原图上的 ImageNet top-1 预测，逐单元记录。
- 忠实性：20 步 / 21 个节点梯形积分 AUC，Insertion 与 Deletion，基于缓存热图，排序按带符号求和热图的绝对值降序。
- 稳定性：K=2 高斯噪声，原始 [0,1] 空间 sigma=0.01，同一图像共享扰动，cosine 在带符号热图上计算。
- P4 将 Monte-Carlo 种子方差（独立归因种子）与输入扰动稳定性分开测量。
- 所有 VOC2007 / CHNCXR 质量行均标记 `ood_pipeline_diagnostic`：ImageNet 分类头属域外，因此此处没有任何任务质量或医学结论。

## 试跑关键观察（描述性，不具统计显著性）

- 效率：Grad-CAM 与 Ablation 最便宜（归因约 0.01–0.11 秒/图）；KernelSHAP、LIME、RISE 约 0.6–1.2 秒/图；IG 约 0.09–0.13 秒/图。
- 忠实性（本协议下 Insertion AUC 越高越好）：VOC2007 上 LIME 较高、IG 最低；CHNCXR 因分类头域外，数值明显偏低。
- 稳定性（输入扰动 cosine）：RISE 接近 1.0（随机掩码对小幅输入噪声不敏感），IG 最低（约 0.35–0.63，梯度噪声大），其余方法居中。
- Monte-Carlo 种子方差（P4）：KernelSHAP 最不稳定（跨种子 cosine 约 0.18–0.73），LIME 中等（约 0.87–0.98），RISE 很稳定（约 0.999）。
- 成本曲线（P3）：成本随采样数/掩码数近似线性增长；内部 batch 从 16 提到 32 有适度加速；2048 采样约为 512 设置的 3–4 倍。
- 前沿方法（P5，仅试跑级操作化）：MA-GIG 约 0.12 s/步，论文默认 200 步约 24.7 s/图，本 OOD 设置下路径积分不收敛（32 步 Insertion AUC≈0.05、稳定性≈0.01），不建议直接进主矩阵；FourierShap 512 采样约 0.6–1.0 s/图，Insertion AUC 与 LIME/KernelSHAP 同量级、稳定性≈0.66–0.95，可作为候选。

## 局限

- 每数据集 16 张仅为计时子集，不是评估集。
- 不做 ANOVA / Pareto / 显著性结论；P2 只验证测量管线，并保存后续分析所需的逐图数据。
- CHNCXR 仅将 ImageNet 头作为域外管线探针，未做结核分类。
