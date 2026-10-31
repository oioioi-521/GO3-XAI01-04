# D：六方法 DCU 配对统计与方法选择（2026-10-15）

本次已完成**接收到的 36 单元 DCU 表格的受限独立验收、配对检验、效应量、置信区间、
多重比较、正确组敏感性分析、共同正确组跨模型比较、探索性相关和 Pareto**。
没有启动新模型实验，没有更改原始逐图记录；这不是原始归因图/未保存噪声图的独立复现实验。

## 可以直接用于汇报的三条结论

1. 固定预算、模型命中组内，IG 的平均 raw MoRF AUC 在六组中的五组最低；例外是
   ImageNet/DenseNet121（KernelSHAP 0.05013，IG 0.05325）。但该例外的配对均值差 CI
   跨零，且 IG 的配对秩优势为正：**不能把显著秩检验说成 KernelSHAP 均值显著更低**。
2. RISE 平均噪声稳定性六组均最高（0.99568–0.99949），不代表它的忠实性同时最佳。
   KernelSHAP 稳定性在六组均高于 LIME-8000（均值优势 0.21539–0.31639，六组配对秩
   二列效应均为 1，Holm p 均 < 0.05）；其忠实性在五组均值低于 LIME，
   但 ImageNet/ResNet50 相反，不能宣布通用最优。
3. 方法选择应按数据集、模型与目标作取舍：质量前沿在不同单元包含 2–4 个方法。
   Grad-CAM 在本批次 wrapper 计时中六组均最低，但计时来自两轮共享负载，
   **不是受控硬件效率排名或跨 GPU 加速比**。

## 图表和完整机器可读结果

- [六方法均值与 95% CI](../analysis/dcu_20261015/six_method_means.png)
- [质量 Pareto 与记录耗时三目标前沿](../analysis/dcu_20261015/quality_pareto.png)
- [质量指标配对效应量图](../analysis/dcu_20261015/paired_quality_effects.png)
- [方法内忠实性/稳定性相关图](../analysis/dcu_20261015/faithfulness_stability_correlation.png)
- [主分析全部 180 个质量比较及 72 个共同正确组跨模型质量比较](../analysis/dcu_20261015/paired_quality_tables.md)
- [完整派生 JSON](../analysis/dcu_20261015/statistics.json)：432 个描述汇总、54 个方法总体检验、
  810 个方法成对比较、108 条前沿记录、324 个相关结果、36 个跨模型总体检验和 108 个跨模型成对比较。
  JSON 包含三个口径、QC、均值/样本标准差/中位数、CI、效应量和所有 p 值，不含原始逐图记录。

## 1. 输入、验收与口径

受控输入：`dcu_team_handoff_20261014.zip`，13,707,240 字节。
SHA256：`14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`。
包内源码快照提交：`f5311404b254e69ee9adf39c31bcbbfc9f3c5315`；这不是 D 当前分支提交。

独立工具核对 ZIP CRC、720 个 manifest 文件和 721 个 checksum 项、冻结 metadata 字节哈希、
各单元 460 张 eval ID、目标、预测/权重身份、90 份执行配置（30 单元 + 60 LIME 分片）的
配置哈希与样本绑定、配对噪声种子、各状态、逐图汇总与均值/样本标准差/n 重算。
共有 66,240 行逐图指标、144 行单元汇总、82,800 条重复记录，36 单元输入齐备。
包内脚本未被执行；completion 中的“verified/resume”只作为交付证据读取，
**本次没有在服务器复跑 resume，也没有独立验证未交付 NPY 的生成来源**。

28 条 Grad-CAM 退化 trace（both_constant 14、candidate_constant 8、reference_constant 6）
按协议计 0 并保留：有效性率是 QC，不是解释正确率，也没有剔除后重新美化结果。
每图 5 次重复先汇总，推断样本是原始图像，不是 5 倍伪重复。

| 数据集 | 模型 | 全样本 | 模型命中组（主分析） | 三模型共同命中组 |
| --- | --- | --- | --- | --- |
| imagenet | densenet121 | 460 | 371 | 332 |
| imagenet | resnet50 | 460 | 408 | 332 |
| imagenet | vgg16 | 460 | 363 | 332 |
| voc | densenet121 | 460 | 388 | 353 |
| voc | resnet50 | 460 | 387 | 353 |
| voc | vgg16 | 460 | 400 | 353 |

VOC 的“命中”只表示冻结 metadata 单目标是 sigmoid 输出的最高分标签，不是多标签
exact match、mAP 或整图语义正确率。ImageNet softmax 与 VOC sigmoid 分开分析，
不把两数据集 raw AUC 绝对值横向排序；不把不同模型各自命中组当成同一批样本。

## 2. 统计方案与不确定性

- 主分析 `model_correct`；`common_correct` 和 `all` 分别完整重算，不能择优选口径。
  跨模型仅使用三模型共同命中图（ImageNet 332、VOC 353），按同图配对。
- 每个数据集/模型的六方法采用 Friedman；报告 Kendall W 和 **4,999 次块内秩置换**
  plus-one p（最小可报 0.0002，不报 p=0）。没有用小 k 时的渐近 p 作为主判断。
- 事后全 15 对采用双侧 Wilcoxon 正态近似（连续性/并列修正、零差不入秩）；差值仅为
  秩/符号计算四舍五入到 12 位，原始均值不变。正效应统一表示行中 a 方法更好。
  输出配对秩二列效应、配对 Cohen dz、胜/负/平和配对优势概率；**不是独立样本 Cohen d**。
- Wilcoxon 涉及差值分布对称性。另给双侧精确符号检验及其独立 Holm 校正作为敏感性检查，
  不把 Wilcoxon 当均值检验，也不把“不显著”当等效。
- 10,000 次按冻结类别分层的原图 case bootstrap；每次所有配对方法/指标共用采样，
  跨模型比较也共用原图采样。报告均值及配对均值优势的 percentile 95% CI。
  CI 是条件于类别数量和命中组的**边际区间，不是同时区间**；不含模型训练、预算选择、
  新图分布及共享机器负载漂移的不确定性。原图独立性是推断假设，未另做同源图簇抽样。
- 每个口径质量成对比较一族 180 项（2 指标 × 6 单元 × 15 对）；质量总体检验一族 12 项。
  计时成对比较另族 90 项、总体另族 6 项，均只作探索性记录；符号检验另作同规模族。
  跨模型质量比较一族 72 项、总体 24 项；计时 36/12 项。均用 Holm，阈值 0.05。
  这些是事先由分析脚本固定的族，不是全报告单一全局 FWER。
- 相关分析在方法/模型/数据集内做，Pearson 与 Spearman 共 216 个端点/口径一起 Holm。
  使用渐近 p，标为探索性，不因相关而宣称机制或因果关系。
- 没有把重复测量套入普通三因素 ANOVA。改用分层配对检验和共同正确组跨模型比较，
  **不宣称三因素交互作用、混合效应模型或完整因子 ANOVA 已完成**。

方法依据：[SciPy Friedman](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html)、
[SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)、
[Holm 方法说明](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)。

## 3. 主分析测量值

表中为命中组算术均值；逐图样本标准差、中位数和所有 CI 见 JSON，图中误差线为均值 CI。
效率是记录的归因 wrapper 毫秒（不含预测、MoRF 和 I/O），不能误写成整轮 wall time。

| 数据集 | 模型 | 方法 | n | raw AUC ↓ | 噪声 Spearman ↑ | 记录耗时 ms ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| ImageNet | ResNet50 | IG | 408 | 0.04273 | 0.66837 | 131.64 |
| ImageNet | ResNet50 | Grad-CAM | 408 | 0.09663 | 0.99006 | 24.03 |
| ImageNet | ResNet50 | Occlusion | 408 | 0.21531 | 0.91568 | 145.83 |
| ImageNet | ResNet50 | RISE | 408 | 0.11830 | 0.99568 | 2183.80 |
| ImageNet | ResNet50 | KernelSHAP | 408 | 0.12537 | 0.99336 | 2199.49 |
| ImageNet | ResNet50 | LIME-8000 | 408 | 0.07213 | 0.67696 | 4261.89 |
| ImageNet | DenseNet121 | IG | 371 | 0.05325 | 0.75901 | 318.11 |
| ImageNet | DenseNet121 | Grad-CAM | 371 | 0.06737 | 0.99896 | 54.61 |
| ImageNet | DenseNet121 | Occlusion | 371 | 0.09685 | 0.98803 | 335.89 |
| ImageNet | DenseNet121 | RISE | 371 | 0.06351 | 0.99898 | 2909.20 |
| ImageNet | DenseNet121 | KernelSHAP | 371 | 0.05013 | 0.99599 | 3891.07 |
| ImageNet | DenseNet121 | LIME-8000 | 371 | 0.06578 | 0.73189 | 6547.67 |
| ImageNet | VGG16 | IG | 363 | 0.02609 | 0.79278 | 142.99 |
| ImageNet | VGG16 | Grad-CAM | 363 | 0.29175 | 0.93384 | 7.00 |
| ImageNet | VGG16 | Occlusion | 363 | 0.04768 | 0.99089 | 210.00 |
| ImageNet | VGG16 | RISE | 363 | 0.06332 | 0.99765 | 4365.54 |
| ImageNet | VGG16 | KernelSHAP | 363 | 0.03955 | 0.99646 | 3344.52 |
| ImageNet | VGG16 | LIME-8000 | 363 | 0.06136 | 0.72314 | 7812.14 |
| VOC (single target) | ResNet50 | IG | 387 | 0.45611 | 0.56480 | 124.58 |
| VOC (single target) | ResNet50 | Grad-CAM | 387 | 0.56279 | 0.98374 | 24.04 |
| VOC (single target) | ResNet50 | Occlusion | 387 | 0.64880 | 0.91510 | 148.16 |
| VOC (single target) | ResNet50 | RISE | 387 | 0.56723 | 0.99778 | 2180.59 |
| VOC (single target) | ResNet50 | KernelSHAP | 387 | 0.51488 | 0.97990 | 2196.73 |
| VOC (single target) | ResNet50 | LIME-8000 | 387 | 0.52481 | 0.76451 | 4218.50 |
| VOC (single target) | DenseNet121 | IG | 388 | 0.34221 | 0.70443 | 315.20 |
| VOC (single target) | DenseNet121 | Grad-CAM | 388 | 0.46633 | 0.99832 | 54.83 |
| VOC (single target) | DenseNet121 | Occlusion | 388 | 0.48441 | 0.98789 | 336.68 |
| VOC (single target) | DenseNet121 | RISE | 388 | 0.44168 | 0.99949 | 2895.37 |
| VOC (single target) | DenseNet121 | KernelSHAP | 388 | 0.38716 | 0.99507 | 3876.07 |
| VOC (single target) | DenseNet121 | LIME-8000 | 388 | 0.43524 | 0.73353 | 6527.47 |
| VOC (single target) | VGG16 | IG | 400 | 0.12134 | 0.77211 | 143.91 |
| VOC (single target) | VGG16 | Grad-CAM | 400 | 0.37316 | 0.95780 | 7.06 |
| VOC (single target) | VGG16 | Occlusion | 400 | 0.27914 | 0.99211 | 210.01 |
| VOC (single target) | VGG16 | RISE | 400 | 0.29303 | 0.99835 | 4363.48 |
| VOC (single target) | VGG16 | KernelSHAP | 400 | 0.19773 | 0.99609 | 3345.27 |
| VOC (single target) | VGG16 | LIME-8000 | 400 | 0.23179 | 0.70670 | 7774.53 |

质量总体检验（12 项 Holm）：

| 数据集 | 模型 | 指标 | Kendall W | 置换 Holm p |
| --- | --- | --- | --- | --- |
| ImageNet | ResNet50 | AUC | 0.4670 | 0.0024 |
| ImageNet | ResNet50 | 稳定性 | 0.8785 | 0.0024 |
| ImageNet | DenseNet121 | AUC | 0.1399 | 0.0024 |
| ImageNet | DenseNet121 | 稳定性 | 0.9277 | 0.0024 |
| ImageNet | VGG16 | AUC | 0.5517 | 0.0024 |
| ImageNet | VGG16 | 稳定性 | 0.9167 | 0.0024 |
| VOC (single target) | ResNet50 | AUC | 0.5368 | 0.0024 |
| VOC (single target) | ResNet50 | 稳定性 | 0.9489 | 0.0024 |
| VOC (single target) | DenseNet121 | AUC | 0.1691 | 0.0024 |
| VOC (single target) | DenseNet121 | 稳定性 | 0.9375 | 0.0024 |
| VOC (single target) | VGG16 | AUC | 0.2601 | 0.0024 |
| VOC (single target) | VGG16 | 稳定性 | 0.9217 | 0.0024 |

## 4. 配对效应与均值不一致的例外

以下全部为 IG 相对 KernelSHAP；AUC 优势定义 KernelSHAP−IG，稳定性优势定义 IG−KernelSHAP。
正值有利于 IG。CI 针对均值差，p 针对配对秩，两者针对的统计问题不同。

| 数据集 | 模型 | 指标 | IG 均值优势 | 95% CI | 秩二列效应 | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| ImageNet | ResNet50 | AUC | 0.08264 | [0.07667, 0.08861] | 0.866 | 5.68e-50 |
| ImageNet | ResNet50 | 稳定性 | -0.32499 | [-0.33378, -0.31630] | -1.000 | 2.43e-66 |
| ImageNet | DenseNet121 | AUC | -0.00313 | [-0.00842, 0.00207] | 0.312 | 5.16e-06 |
| ImageNet | DenseNet121 | 稳定性 | -0.23698 | [-0.24523, -0.22894] | -1.000 | 2.01e-60 |
| ImageNet | VGG16 | AUC | 0.01346 | [0.01178, 0.01514] | 0.682 | 1.43e-27 |
| ImageNet | VGG16 | 稳定性 | -0.20368 | [-0.21058, -0.19686] | -1.000 | 3.76e-59 |
| VOC (single target) | ResNet50 | AUC | 0.05877 | [0.05097, 0.06642] | 0.655 | 3.53e-27 |
| VOC (single target) | ResNet50 | 稳定性 | -0.41510 | [-0.42847, -0.40199] | -1.000 | 5.44e-63 |
| VOC (single target) | DenseNet121 | AUC | 0.04495 | [0.03467, 0.05511] | 0.261 | 0.000178 |
| VOC (single target) | DenseNet121 | 稳定性 | -0.29064 | [-0.30233, -0.27894] | -1.000 | 3.98e-63 |
| VOC (single target) | VGG16 | AUC | 0.07639 | [0.06652, 0.08627] | 0.452 | 1.94e-13 |
| VOC (single target) | VGG16 | 稳定性 | -0.22398 | [-0.23367, -0.21447] | -1.000 | 4.63e-65 |

特别是 ImageNet/DenseNet121 AUC：IG 均值优势 −0.00313，95% CI [−0.00842, 0.00207]，
秩二列效应 +0.312，Wilcoxon Holm p=5.16e−6；表示秩加权的图像内比较偏向 IG，
不能推出 IG 均值更低，也不能据点估计宣布 KernelSHAP 均值优势已被证实。
该处共同正确组和全样本口径也不应与主分析混为同一总体。

主分析中下列 7 个质量比较的 Wilcoxon/符号检验校正后判断不一致，
结论需明确“依赖差值幅度/秩假设”，不单靠一颗星做最终选型：

| 数据集 | 模型 | 指标 | 方法对 | Wilcoxon Holm p | 符号检验 Holm p |
| --- | --- | --- | --- | --- | --- |
| ImageNet | ResNet50 | AUC | Grad-CAM / KernelSHAP | 1.82e-08 | 0.53 |
| ImageNet | ResNet50 | 稳定性 | Grad-CAM / KernelSHAP | 0.000181 | 0.273 |
| ImageNet | DenseNet121 | AUC | Grad-CAM / RISE | 0.00814 | 0.0997 |
| ImageNet | DenseNet121 | AUC | Occlusion / RISE | 0.0351 | 1 |
| ImageNet | VGG16 | AUC | Occlusion / KernelSHAP | 0.0138 | 0.821 |
| VOC (single target) | DenseNet121 | AUC | Occlusion / RISE | 0.000698 | 0.0566 |
| VOC (single target) | VGG16 | AUC | RISE / LIME-8000 | 0.000131 | 0.203 |

## 5. Pareto：质量与记录计时分开看

用每单元均值的 AUC↓、稳定性↑构建双目标质量前沿；另加本批次 ms↓形成三目标记录前沿。
不归一化加权出总分，不混不同模型/数据集；前沿不是显著性结论。

| 数据集 | 模型 | 质量前沿 | 含记录时间的三目标前沿 |
| --- | --- | --- | --- |
| ImageNet | ResNet50 | IG, Grad-CAM, RISE, LIME-8000 | IG, Grad-CAM, RISE, LIME-8000 |
| ImageNet | DenseNet121 | RISE, KernelSHAP | IG, Grad-CAM, RISE, KernelSHAP |
| ImageNet | VGG16 | IG, RISE, KernelSHAP | IG, Grad-CAM, Occlusion, RISE, KernelSHAP |
| VOC (single target) | ResNet50 | IG, Grad-CAM, RISE, KernelSHAP | IG, Grad-CAM, RISE, KernelSHAP |
| VOC (single target) | DenseNet121 | IG, RISE, KernelSHAP | IG, Grad-CAM, RISE, KernelSHAP |
| VOC (single target) | VGG16 | IG, RISE, KernelSHAP | IG, Grad-CAM, Occlusion, RISE, KernelSHAP |

JSON 附 bootstrap 前沿出现频率；它只是本次条件抽样的稳定性诊断，
**不是某方法属于真实前沿的后验概率**。ImageNet/DenseNet121 的 Grad-CAM 稳定性与 RISE
均值非常接近，质量前沿变化对样本口径敏感；计时三维前沿只可在附带共享负载限制时引用。

## 6. 敏感性、跨模型与相关结论

下表是各指标 90 个方法对中 Holm p < 0.05 的数量，仅作覆盖诊断，不作为结果优劣分数。

| 口径 | AUC / 90 | 稳定性 / 90 | 记录时间 / 90 |
| --- | --- | --- | --- |
| model_correct | 78 | 88 | 90 |
| common_correct | 79 | 88 | 90 |
| all | 78 | 89 | 90 |

| 数据集 | 模型 | 主分析质量前沿 | 共同命中组 | 全样本 |
| --- | --- | --- | --- | --- |
| ImageNet | ResNet50 | IG, Grad-CAM, RISE, LIME-8000 | IG, Grad-CAM, RISE, LIME-8000 | IG, Grad-CAM, RISE, LIME-8000 |
| ImageNet | DenseNet121 | RISE, KernelSHAP | RISE, KernelSHAP | Grad-CAM, RISE, KernelSHAP |
| ImageNet | VGG16 | IG, RISE, KernelSHAP | IG, RISE, KernelSHAP | IG, RISE, KernelSHAP |
| VOC (single target) | ResNet50 | IG, Grad-CAM, RISE, KernelSHAP | IG, Grad-CAM, RISE, KernelSHAP | IG, RISE, KernelSHAP |
| VOC (single target) | DenseNet121 | IG, RISE, KernelSHAP | IG, RISE, KernelSHAP | IG, RISE, KernelSHAP |
| VOC (single target) | VGG16 | IG, RISE, KernelSHAP | IG, RISE, KernelSHAP | IG, RISE, KernelSHAP |

主分析与共同正确组质量前沿相同；全样本时 ImageNet/DenseNet121 增加 Grad-CAM，
VOC/ResNet50 移除 Grad-CAM。**“正确组”与“包含预测错误”的总体不可混称**。

共同正确组的跨模型质量比较 72 对中 69 对 Holm p < 0.05，但不可据此评判分类器准确率：
模型预测尺度和原始置信度不同，raw MoRF AUC 不是归一化的“架构解释质量”。
KernelSHAP 在两个数据集的共同正确组中，VGG16 的平均 AUC 更低、噪声稳定性更高，
相应 VGG16 与另两个模型的配对质量比较均校正后显著；这是固定 checkpoint/指标口径下的观察，
不是控制所有架构因素后的因果结论。另三对不显著比较见完整表，不声称所有模型对均不同。

方法内 AUC/稳定性的 Spearman 相关既有正也有负，例如 ImageNet/DenseNet121 的 LIME
rho=+0.383（Holm p=4.49e−12），VOC/VGG16 的 KernelSHAP rho=−0.392（p=8.89e−14）。
由于 AUC 越低越好，正相关表示较差忠实性与较高稳定性伴随出现；负相关并非“指标冲突”。
这些是方法内图像差异，不能拿 36 个均值混成总体相关，也不证明提高一个指标会改变另一个。

RQ1：报告的是限定预算的逐单元取舍，不是六方法统一冠军。
RQ2：展示共同正确组的跨模型配对差异，不声称已估计数据集/架构/方法三因素交互。
RQ3：给出质量及记录时间前沿和方法内相关，使用限制应随图/报告一起交付。

## 7. 运行与剩余边界

LIME 是交接包中的 VectorizedLIME、8,000 samples、batch 32，包内标记为已审阅固定比较预算，
不是采样收敛证明。交接材料称 720 张 debug 图/R² 与原实现一致；本次只验证其归档哈希，
没有重新执行这项算法等价试验，也没有因此回溯批准 E 旧提交的全部门禁缺口。
旧 E 64d2ffd 工程审查和原始 GPU KernelSHAP 六单元验收仍各自保留历史边界。

五方法和 LIME 分两轮、共享 DCU 并发运行；交付文档记载两轮 wall time 为 4:21:06.2 和
3:52:06.7，合计 8:13:12.9，不是六方法同一受控队列。没有匹配的 2080 Ti LIME-8000
正式计时，不合并旧 GPU 与新 DCU 耗时。噪声图未存盘；完整归因热图另在大归档中，
E 若做同图 Demo 需按需获取，本次 D 统计不以此为阻塞。新增预算实验/受控计时或新训练
需要另行定义，不为了补一句“全部完成”自行重跑。

复现（受控 ZIP 不入 Git；只运行本仓自写模块）：

```powershell
# 在独立 Python 3.12 分析环境安装依赖；不要升级原实验环境。
python -m pip install -r requirements-analysis.txt
python -m analysis.dcu_statistics '本地路径/dcu_team_handoff_20261014.zip' --output analysis/dcu_20261015/statistics.json
python -m analysis.plot_dcu_statistics
python -m analysis.report_dcu_statistics
python -m pytest -q tests/test_dcu_statistics.py
python -m pip check
# 原项目环境另跑完整测试与依赖检查（Windows 临时目录受限时指定新 basetemp）。
.\.venv\Scripts\python.exe -m pytest -q --basetemp=results/.pytest_dcu_recheck
.\.venv\Scripts\python.exe -m pip check
```

随机主种子 20261015，bootstrap 10000 次，
置换 4999 次。分析环境：`{"python": "3.12.14", "numpy": "2.3.5", "pandas": "3.0.1", "scipy": "1.15.3", "PyYAML": "6.0.3"}`。
源码 SHA256：统计模块 `aa94d9777465158a70c615f1af2f813341bd25465a3cd0c3becd13a832d5141e`；
校验器 `ebd24fd916d0ddb7132720d5e96288be98ceb1e6ee4458a273307ffd96b891ff`。
软件回归测试是分析程序验证，不是新模型/服务器实跑的证据；最终复验记录见交付入口。

本次输出可以通过 GitHub 给组员复核；原 ZIP、逐图明细、权重、NPY 和大归档仍受控外部交接。
图表和报告已准备不等于已经代用户在微信群发送、获得组长签收或合并 main。
