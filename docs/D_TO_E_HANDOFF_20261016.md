# 给 E / A：D 统计图表与报告整合清单

日期：2026-10-16。使用 `feat/kernelshap-analysis` 分支；尚未合并 main。
这是可直接整合的仓库交付说明，不代表已发送微信群或获得组长签收。

## 从这里开始

1. [完整统计报告](DCU_STATISTICAL_ANALYSIS_20261015.md)：三条汇报结论、样本口径、
   配对检验/效应量/区间、多重比较、方法选择和限制，均附可追溯输入 SHA。
2. [统一交付入口](D_DELIVERY_INDEX.md)：包含中期图、旧 GPU 单方法结果和新 DCU 统计，
   请按日期/运行分开使用。
3. [全部派生数值](../analysis/dcu_20261015/statistics.json)与
   [主分析 180 对、共同正确组跨模型 72 对质量比较表](../analysis/dcu_20261015/paired_quality_tables.md)。

## PPT / 报告建议放什么

| 用途 | 可直接用的图 | 必须保留的口径 |
| --- | --- | --- |
| 六方法总体展示 | [均值与 95% CI](../analysis/dcu_20261015/six_method_means.png) | 各模型命中组；CI 是条件分层 bootstrap 边际区间，不是 SD |
| 方法选择 | [质量 / 记录时间前沿](../analysis/dcu_20261015/quality_pareto.png) | 逐模型/数据集比较；双目标质量与三目标记录计时分开，前沿不等于显著性 |
| 配对比较证据 | [质量效应量](../analysis/dcu_20261015/paired_quality_effects.png) | 正值有利于行方法；星号对应 180 项质量族的 Holm 校正，不是均值检验 |
| 指标关系 | [方法内相关](../analysis/dcu_20261015/faithfulness_stability_correlation.png) | 探索性、方法内相关；AUC 越低越好，不解释成因果 |

可以使用的三句结论：

- 固定预算、模型命中组中，IG 的平均 raw MoRF AUC 六组中五组最低，但存在
  ImageNet/DenseNet121 例外；该例外均值差 CI 跨零，而且配对秩优势偏向 IG，
  不应宣称 KernelSHAP 的均值优势显著。
- RISE 的平均 RGB 噪声稳定性六组最高；高稳定性不自动代表高忠实性，不能据此宣布全指标冠军。
- 质量前沿随数据集、模型改变，方法选择存在取舍；记录时间来自共享 DCU 两轮运行，
  只能说明本批次计时，不是受控跨 GPU 加速比。

## 讲解时不能省略

- ImageNet / VOC 各 460 张 eval。主分析为各模型命中组；ImageNet n=408/371/363、
  VOC n=387/388/400（ResNet50/DenseNet121/VGG16）。跨模型使用共同命中组 332/353。
- VOC 命中表示冻结单目标为 sigmoid argmax，不是多标签 exact-match 或 mAP。
- 每图五次噪声先汇总，推断单位仍是原图；28 条 Grad-CAM 退化 trace 按冻结协议计零并保留。
- 普通三因素 pooled ANOVA 未直接套用；已采用配对替代检验，不能把图写成三因素交互结论。
- LIME 固定 8000 samples / batch32 的本次比较预算不是预算收敛证明。
- 旧 GPU KernelSHAP 六单元与新 DCU 36 单元是不同运行，不能混合数值或耗时。

## 仍需要成员协作的事

E 整合 PPT、固定 Demo 与典型/失败案例时，若需要完整真实热图，按
[原始交接说明](DCU_TEAM_HANDOFF_20261014.md)获取外部大归档，核对样本 ID、target、
checkpoint、配置及 map hash。不要用当前统计图当成同图归因热图，也不要把 synthetic 示例
标成真实 eval 案例；样例需说明选择规则。

A 统筹最终报告结构及结论采纳；各方法负责人复核本人方法解释。D 代码/图表交付已可通过
GitHub 查看，不要求用户再通过微信重复传输这些小型产物。原始 ZIP、NPY、权重不入 Git。

本轮 LIME 通用门禁修复、测试与历史结果边界见
[PR #17 集成复核与门禁修复记录](D_DCU_INTEGRATION_REVIEW_20261016.md)。
