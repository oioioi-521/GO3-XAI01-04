# 成员 D：统计分析骨架

本目录只提供可复现的分析程序，不包含也不伪造正式实验结论。正式结论必须来自全组统一
schema 的真实结果，并注明数据、配置和模型版本。

## 已提供

- `schema.py`：检查字段、空值、非有限值、负耗时和重复实验记录。
- `aggregate.py`：把逐图长表聚合为单元长表；无法从逐图表恢复的 `config_hash` 保持空白，
  或从 runner 生成的 `units.csv` 连接，绝不猜测。
- `anova.py`：三因素 ANOVA、partial eta-squared、残差 Shapiro 和 Levene 诊断，另提供
  按 `image_id × model × dataset` 配对的 Friedman 对照结果。
- `pareto.py`：显式指定每个指标的优化方向，避免把 MoRF AUC、Max-Sensitivity 或耗时
  的“越低越好”解释反。
- `correlation.py`：在同一图像/模型/数据集/方法的配对观测上计算 Pearson、Spearman、
  样本数和 p 值。

## 使用顺序

```bash
# 1. 原始结果门禁
python -m analysis.validate_results --per-image results/per_image.csv --units results/units.csv

# 2. 聚合（建议同时提供 runner 生成的 units.csv，以保留 config_hash）
python -m analysis.aggregate --input results/per_image.csv --units results/units.csv --output results/analysis/data_long.csv

# 3. 单个指标的 ANOVA 与 Friedman 对照
python -m analysis.anova --input results/per_image.csv --metric faithfulness_morf_auc_raw --output-dir results/analysis/anova_faithfulness

# 4. Pareto；三个指标的真实名称应以最终 schema 为准
python -m analysis.pareto --input results/analysis/data_long.csv --metric faithfulness_morf_auc_raw:min --metric max_sensitivity:min --metric efficiency_time_ms:min --output results/analysis/pareto.csv

# 5. 指标冲突分析
python -m analysis.correlation --input results/per_image.csv --output-dir results/analysis/correlation
```

## 解释边界

- 当前脚本“可以运行”不等于 ANOVA 假设成立，也不等于已得到 RQ1–RQ3 结论。
- Friedman 只使用包含全部方法的完整配对块，输出会报告实际保留的块数。
- Friedman 的平均秩默认按数值从小到大；解释前必须先确认该指标究竟是越大还是越小越好。
- `p < 0.05` 仍需结合效应量和多重比较；当前骨架尚未把 Nemenyi 事后检验冒充为完成项。
- Pareto 结果取决于传入的指标及方向，命令行必须明确写 `:min` 或 `:max`。
- `max_sensitivity` 只是示例字段；稳定性协议仍需全组冻结，正式结果中没有该字段时不要运行三维 Pareto。
