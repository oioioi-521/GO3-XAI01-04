# B→D：ImageNet / DenseNet121 的 IG–KernelSHAP 例外核对

2026-10-26 · Refs #19 / #21。状态：本轮已核对来源及逐图描述量，以下结论仍**待 D 回复确认**；未代发评论，未重新运行 bootstrap/Wilcoxon/置换。

证据入口：[D 正式报告](DCU_STATISTICAL_ANALYSIS_20261015.md)、[配对附表](../analysis/dcu_20261015/paired_quality_tables.md)、[statistics.json](../analysis/dcu_20261015/statistics.json) 的 `method_pairs`，以及 [三口径摘录](../analysis/b_m5_20261026/ig_kernelshap_exception.csv)。原图CSV重算样本数、均值优势和胜图数，区间与秩p逐项对照既有JSON/报告，不能把这两种验证写成重新检验。

本表优势定义 `KernelSHAP raw AUC − IG raw AUC`，正值表示 IG 较低；不是 IG−KernelSHAP。均值区间为边际配对bootstrap 95% CI；p为配对Wilcoxon的Holm校正，质量检验族各180项。

| cohort | n | IG均值优势 | 均值差95% CI | IG胜/负 | 秩二列效应 | Wilcoxon Holm p |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| model_correct | 371 | −.0031269003 | [−.0084232795,.0020708717] | 259/112 | +.3115961 | 5.15549e−6 |
| common_correct | 332 | −.0025613582 | [−.0081186469,.0028862960] | 233/99 | +.3057998 | 3.43430e−5 |
| all | 460 | −.0025157057 | [−.0068193179,.0017550938] | 323/137 | +.3267000 | 3.67946e−8 |

主分析 IG AUC均值约.05325，KernelSHAP约.05013；KernelSHAP点估计更低，但均值差区间跨0，不能宣称其均值显著更优。IG在371图中259图AUC更低，秩效应为正且秩检验显著；这也不能转述成IG均值显著更低。均值、胜率和按差值绝对秩加权的量不同，少量大差值可改变均值方向；这是对统计量的解释，不是本轮已证明某种生成机制。

请 D 确认：

1. 报告/PPT保留上述差值方向、三种cohort和各自n，不把371与332混用。
2. 均值CI和Wilcoxon针对不同统计量；边际95%CI不冒充经过Holm校正的同时置信区间。
3. 描述“KernelSHAP均值点估计较低，均值优势未证实；配对秩方向偏向IG”，避免“IG/KernelSHAP显著更低”的无统计量限定句。
4. 签名检验属于敏感性检查，与Wilcoxon p不同，不以其中一个代替均值推断；#21后续交互分析单独登记，不由此例宣称方法×模型交互成立。
5. 当前 `dafb655` 已修正历史报告的RQ2/RQ3对应，A的合稿说明仍描述旧编号状态；请A/C合稿时按最新差异更新。D后续统计若改变表格/图注，保留版本和来源身份。
