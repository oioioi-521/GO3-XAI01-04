# 试跑报告目录（2080 Ti 计时/成本校准）

本目录是 XAI01-04 的 **M1/M2 限时试跑**文档与结果；代码在
[`../../experiments/pilot-2080ti/`](../../experiments/pilot-2080ti/)，
试跑总说明与复现见该处的 [README](../../experiments/pilot-2080ti/README.md)。

## 文件组织

| 路径 | 内容 |
|---|---|
| `PLAN.md` | 试跑方案（依据 PDF、机器快照、分阶段 P0–P4、外推公式） |
| `HANDOVER.md` | 交接总结（范围、环境、数据、结果、组合爆炸策略、局限、待办） |
| `RESULT_INDEX.md` | 阶段状态与交付物总索引 |
| `reports/` | 各阶段报告与汇总表（校准、评价、成本曲线、种子复核、前沿方法、外推、完整矩阵用时） |
| `tables/` | 逐图/单元 CSV 与完整矩阵用时 CSV |
| `summaries/` | 各阶段聚合 JSON（`*_summary.json`、`extrapolation.json`） |
| `environment/` | 只读环境/硬件快照与沙箱诊断记录 |

## 重要声明

- 这是**试跑级**结果：每数据集 16 张（前沿方法 8 张）计时子集，不做统计显著性结论。
- VOC2007 / CHNCXR 上的 ImageNet 预训练分类头属域外，所有质量行标记
  `quality_scope=ood_pipeline_diagnostic`，不是任务忠实性或医学结论。
- 前沿方法 MA-GIG / FourierShap 为**试跑级操作化**（`frontier_pilot_approximation`），不是作者原实现；
  FourierShap 在 PDF 中未给引用，本试跑依据 arXiv:2410.06300。

## 原始记录

逐条原始记录（`*_records.jsonl`）、归因缓存与权重/VAE 缓存不入库（指导文件 §7.2），
需要时按 `../../experiments/pilot-2080ti/README.md` §4 复现生成。
