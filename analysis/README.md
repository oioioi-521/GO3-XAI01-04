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

KernelSHAP 的六个 460 张正式单元完成后，先独立校验冻结协议、460 张基础/稳定性逐图指标、
2,300 条重复 trace、配对 seed、metadata target、逐图/单元重算、float32 参考图来源和
`processed=0, skipped=460` 立即续跑记录。运行中也可用 `--unit` 只验已完成的单元：

```bash
python -m analysis.validate_kernelshap_formal
python -m analysis.validate_kernelshap_formal --unit resnet50/imagenet
```

该命令只读结果和已冻结的配置/权重，不运行 GPU 归因；全组六方法统计仍需其他方法的正式输入。

六单元全部验收后，可生成一张只比较 KernelSHAP 自身的报告/PPT 图；缺少任一单元或
误用 40 张门禁结果时，脚本会拒绝出图。图中点/误差线为 460 张图的均值/样本标准差，
不是跨方法排名或显著性检验：

```bash
python -m analysis.plot_kernelshap_formal
```

```bash
# 0. 分目录正式结果交接：先校验 manifest、配置/权重身份、每单元样本数和汇总，
#    再生成统一长表及 top-1 正确/错误分组（VOC 不是多标签 exact match）
python -m analysis.merge_eval_handoff \
  --root results/incoming/B_OCCLUSION_EVAL_LIGHT \
  --output-dir results/analysis/occlusion_eval_acceptance

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
- `merge_eval_handoff` 的 `top1_correct` 对 VOC 表示冻结单目标类别是否为 sigmoid 输出的
  最高分标签，不是 20 类多标签 exact-match accuracy；分组名称不得改写为“多标签预测正确”。
- `merge_eval_handoff` 还会生成 `common_correct_cohort.csv`。跨模型配对比较使用三模型都
  命中冻结单目标类别的共同样本；单模型主分析使用该模型命中组，全样本只作敏感性分析。
- `validate_received_handoffs` 只读校验接收的轻量/float32 eval 包及单图、40 张稳定性
  门禁包，包括逐文件哈希、NPY 类型/形状/范围、trace seed、逐图/单元重算和 provenance：

```bash
python -m analysis.validate_received_handoffs \
  --eval-manifest results/incoming/B_OCCLUSION_EVAL_LIGHT/results/occlusion_handoff/B_OCCLUSION_EVAL_FILES.json \
  --eval-root results/incoming/B_OCCLUSION_EVAL_LIGHT \
  --eval-root results/incoming/B_OCCLUSION_EVAL_FLOAT32 \
  --stability-root results/incoming/occlusion_stability_gates_light
```
