# D：IG/KernelSHAP 例外确认与方法选择初稿复核

2026-10-27；接续 B 的 [Issue #21 评论](https://github.com/oioioi-521/GO3-XAI01-04/issues/21#issuecomment-6417577319)
及 [B 三口径核对表](https://github.com/oioioi-521/GO3-XAI01-04/blob/1e71afde70b4ed1222d891646abea90b74ccc376/docs/B_M5_D_STATISTICAL_CHECK.md)。
这是 D 的程序和文档复核，未代表用户发送评论，也不代表教师或其他成员签收。

## 三种样本口径均确认

数据集/模型：ImageNet / DenseNet121。差值方向始终为 **KernelSHAP raw AUC − IG raw AUC**，
正数表示 IG 的 AUC 较低。下表的 95% CI 是边际配对均值差区间；Holm p 来自配对秩检验，
各 cohort 的质量检验族为 180 项，两者不是同一统计量。

| cohort | n | 均值差 | 均值差 95% CI | IG 胜/负 | 秩二列效应 | Wilcoxon Holm p |
|---|---:|---:|---|---|---:|---:|
| model_correct（主分析） | 371 | −0.0031269003 | [−0.0084232795, 0.0020708717] | 259 / 112 | +0.3115961 | 5.1554907e−6 |
| common_correct | 332 | −0.0025613582 | [−0.0081186469, 0.0028862960] | 233 / 99 | +0.3057998 | 3.4342987e−5 |
| all | 460 | −0.0025157057 | [−0.0068193179, 0.0017550938] | 323 / 137 | +0.3267000 | 3.6794614e−8 |

可以直接合稿的表述：

> ImageNet/DenseNet121 的主分析中，KernelSHAP 的平均 raw MoRF AUC 点估计较低，
> 但配对均值差的边际 95% CI 跨零，未证实其均值优势。IG 在 371 图中有 259 图的 AUC 更低，
> 配对秩方向偏向 IG；不能将秩检验显著改写成 IG 均值显著更低。

共同命中组与全 eval 的同向现象作为敏感性结果单列，不能把 n=371 与 n=332 混用。
不同统计量的结果不矛盾；少量大差值可以影响均值方向，但这里不进一步推断其生成机制。
边际 CI 不是 Holm 同时置信区间；符号检验是另一项敏感性检验，也不能替代均值推断。
本例不构成方法 × 模型交互检验，不能用“两个模型的 p 值不同”证明交互。

这次实际从冻结 CSV 重算了描述量、秩效应、Wilcoxon、各 cohort 全部 180 项质量检验及 Holm，
并按原类别分层、配对索引、10,000 resamples 和原 seed 重算 CI，均匹配既有 JSON。
复算来源与结果：[ig_kernelshap_recheck.json](../analysis/dcu_20261027/ig_kernelshap_recheck.json)；
代码：[recheck_dcu_exception.py](../analysis/recheck_dcu_exception.py)。没有运行 GPU 归因或未保存的噪声图。

## 方法选择指南的复核

复核对象为 A [固定版本报告初稿 §6](https://github.com/oioioi-521/GO3-XAI01-04/blob/b3df72b16199e8d785723c421b06f1e3f4f77f8c/docs/FINAL_REPORT_DRAFT.md)。
该版本报告也包含于新分析包；数值取自当前 `statistics.json` 的 `cohort=model_correct`。

| A 的候选规则 | D 数值核对 | 合稿限制 |
|---|---|---|
| 忠实性优先，将 IG 作为比较候选 | 六个单元中五个平均 raw AUC 最低；ImageNet/DenseNet121 例外已复算 | 候选规则是当前固定预算的描述；不是 IG 六组均值显著最优 |
| 输入噪声稳定性优先，将 RISE 作为候选 | 六个单元平均稳定性最高 | 固定方法 seed 下的 RGB 输入噪声稳定性；不是重采样稳定性或忠实性保证 |
| 记录成本优先，将 Grad-CAM 作为候选 | 六个单元的平均记录时间最低 | 共享 DCU 两轮、wrapper 时间；不是受控跨 GPU/独占硬件比较 |
| 多目标选择查质量前沿和配对证据 | 六组质量前沿分别包含 4、2、3、4、3、3 种方法 | 前沿基于点估计；不等于前沿之间均有显著差异 |
| 只可查询输出分数时考察扰动方法 | KernelSHAP 当前实现不需要输入梯度 | 要核对实际响应量；KernelSHAP 接收模型 logit，不能随意替换成只返回概率的 API |
| LIME-8000 先核对局部拟合和前沿位置 | 当前只有 ImageNet/ResNet50 的质量前沿包含 LIME | 8000 是固定预算；没有预算收敛或普遍优势证明 |

质量前沿按 ImageNet/ResNet50、DenseNet121、VGG16，再 VOC 三模型排列：
IG/Grad-CAM/RISE/LIME；RISE/KernelSHAP；IG/RISE/KernelSHAP；
IG/Grad-CAM/RISE/KernelSHAP；IG/RISE/KernelSHAP；IG/RISE/KernelSHAP。

D 确认上述初稿规则与现有数据和限制相容，可供 A 继续合稿；新增交互分析尚未完成，
不能将其省略或声称最终统计版本已经共同冻结。规则流程图没有训练/外部验证，不冒充选做决策树实验。
RQ2/Pareto、RQ3/指标冲突已在 D 分支 `dafb655` 修正；A/C 合稿时更新仍描述旧编号的说明。
