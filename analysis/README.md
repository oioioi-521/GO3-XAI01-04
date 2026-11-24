# 成员 D：统计分析与报告

本目录同时保留早期分析骨架和已验收输入上的真实报告，不能将骨架示例输出当实验结论。
正式结论必须注明数据、配置、模型版本及计时限制。

## 当前正式统计入口（2026-10-15）

已接收完整六方法 DCU 表格包；不再沿用下方旧流程的“其他方法输入待收齐”状态。
见 [统计报告](../docs/DCU_STATISTICAL_ANALYSIS_20261015.md)和
[图表/派生 JSON](dcu_20261015/statistics.json)。

```powershell
# 使用独立 Python 3.12 CPU 分析环境；不升级原 runner 环境，不需 torch。
python -m pip install -r requirements-analysis.txt
python -m analysis.dcu_statistics '本地路径/dcu_team_handoff_20261014.zip' --output analysis/dcu_20261015/statistics.json
python -m analysis.plot_dcu_statistics
python -m analysis.report_dcu_statistics
python -m pytest -q tests/test_dcu_statistics.py
```

`validate_dcu_team_handoff.py` 固定 ZIP SHA，只读核对全部表格、样本、配置分片、预测和 trace。
`dcu_statistics.py` 按原图配对，使用命中组/共同命中组/全样本三口径；含多重比较、效应量、
条件 bootstrap CI、共同命中组跨模型比较、相关及两种 Pareto，不加载 torch 或启动归因。
`plot_dcu_statistics.py` 和 `report_dcu_statistics.py` 从派生 JSON 重建四图和报告。
不会执行交接包代码，不读取外部 NPY/权重；表格验收不等于原始生成过程的独立复现。
共享负载/两轮计时只作探索性分析，不是硬件速度因果比较。

下方 `anova.py`/`correlation.py`/`pareto.py` 为早期工程工具示例，不是本次 DCU 统计主流程。
尤其普通 pooled ANOVA 不适合直接处理本次重复图像设计，`max_sensitivity` 也不是本次指标。

2026-10-14 更新：六方法 DCU 正式输入现已齐全，统一交接入口见
[36 组数据交接](../docs/DCU_TEAM_HANDOFF_20261014.md)。交接包已包含全样本、单模型正确组、
三模型共同正确组及各自重算汇总。当时统计结论尚待分析，现已由上方 10 月 15 日入口交付；
下方中期/单方法示例保留为历史流程。
本轮 Pareto 的稳定性指标为 `stability_spearman:max`，不要沿用旧示例的 `max_sensitivity:min`。

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
- `plot_midterm_d.py`：中期限定口径的四方法 ImageNet 全样本忠实性图，以及只使用 A
  同一台 GPU 数据的 IG/Grad-CAM 双目标成对 Pareto 图；具体来源、结论及限制见
  `docs/MIDTERM_D_COMPARISON.md`。

## 原始 KernelSHAP GPU 单元及早期工程工具使用顺序

KernelSHAP 的六个 460 张正式单元完成后，先独立校验冻结协议、460 张基础/稳定性逐图指标、
2,300 条重复 trace、配对 seed、metadata target、逐图/单元重算、float32 参考图来源和
`processed=0, skipped=460` 立即续跑记录。运行中也可用 `--unit` 只验已完成的单元：

```bash
python -m analysis.validate_kernelshap_formal
python -m analysis.validate_kernelshap_formal --unit resnet50/imagenet
```

该命令只读旧 GPU 结果和已冻结的配置/权重，不运行 GPU 归因；新 DCU 全组输入使用上方新入口。

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
- `max_sensitivity` 只是早期示例字段；本次实际指标是 `stability_spearman:max`，不要拿示例字段生成新 DCU 前沿。
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
