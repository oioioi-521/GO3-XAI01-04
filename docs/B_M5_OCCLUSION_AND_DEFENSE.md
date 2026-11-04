# B：Occlusion 原理、正式结果与答辩准备

2026-10-26 · Refs #19，供报告 §2.3/§8 和 E 展示。这里复用统一 DCU 结果，不是 B 的历史 RTX 门禁/正式稳定性批次。

## 原理与实际执行

Occlusion 逐个将连续矩形区域替换为给定 baseline，以替换前后的目标输出差描述区域影响；一个像素被多个窗口覆盖时，按覆盖的输出差平均。可写成 `A_i = mean_{w contains i}[f_t(x)-f_t(x with w replaced)]`。它衡量特定替换干预下的响应，不自动等于现实世界因果贡献。原始研究为 [Zeiler 与 Fergus, Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)；实现语义和覆盖/边缘规则以 [Captum Occlusion 官方文档](https://captum.ai/api/occlusion.html)为准（2026-10-26 读取）。

本项目冻结 `source/five/release/experiments/attribution/occlusion.py` 直接把返回 logits 的模型传给 Captum；target 来自 metadata 真值，不换成预测类别。正式窗口是 `(3,32,32)`，stride `(3,16,16)`，`perturbations_per_eval=16`；输入为224×224 RGB归一化张量，黑基线为 `-mean/std`。类构造默认的15×15/stride8不是本批次参数。窗口覆盖后得到有符号通道归因，以通道算术均值投影，再空间 min-max；常量返回全零，保存 float32/224×224。

因此 map 的0代表本图最小投影贡献，不能读成“零贡献”；也不能把 min-max 后的值当原始 logit 差。正式 MoRF 则对相同 target 的 ImageNet softmax/VOC sigmoid 分数做21点 raw AUC，解释响应与评价响应须分别写清。参数、源码与输入身份详见 [模型/指标复核](B_M5_MODEL_AND_METRICS.md)及 [源码哈希表](../analysis/b_m5_20261026/frozen_source_inventory.csv)。

## 正式结果与解释

| 数据集 | 模型 | n | raw AUC ↓ | 噪声稳定性 ↑ | 记录时间 ms ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| imagenet | resnet50 | 408 | 0.21531 | 0.91568 | 145.83 |
| imagenet | densenet121 | 371 | 0.09685 | 0.98803 | 335.89 |
| imagenet | vgg16 | 363 | 0.04768 | 0.99089 | 210.00 |
| voc | resnet50 | 387 | 0.64880 | 0.91510 | 148.16 |
| voc | densenet121 | 388 | 0.48441 | 0.98789 | 336.68 |
| voc | vgg16 | 400 | 0.27914 | 0.99211 | 210.01 |

来源：`statistics.json` 的 `summaries`，`cohort=model_correct`、`method=occlusion`；本轮由交接 CSV 的筛选表重算均值/SD/n并核对。表中是均值，不是均值差、显著性、SD或CI。ImageNet/VOC 的 raw AUC 分开解读。Occlusion 具备直接扰动解释、无需梯度的接口，但当前结果没有支持它为全指标冠军。某图 AUC 较高不等于实现失效；要结合干预基线、窗口尺度、输出量和六方法配对证据。

例如215 / VOC / ResNet50，本图 Occlusion AUC=.86079、稳定性=.84553；298 / VOC / VGG16 则是 .13020/.99732。不能用两个个例的差异断言模型架构造成某机制。六组主分析中 Occlusion 均未进入 D 的二维质量前沿；部分 VGG16 单元进入含记录时间的三维前沿，见 [D 报告](DCU_STATISTICAL_ANALYSIS_20261015.md)。点估计前沿不是普适最优或统计支配证明。

局限：遮挡制造分布外输入；窗口/stride和黑基线影响排序；重叠平均可能稀释局部效应；跨通道均值与min-max损失绝对幅度；类别响应不同于概率归因方法；224×224及冻结子集限制推广；共享DCU计时不可推导独占设备速度。没有为这些机制新增正式对照实验。

## 可直接讲述的短稿

我负责 Occlusion。我们用32×32黑色窗口、步长16逐步遮挡，观察冻结真值类别的logit变化，再把覆盖贡献投影成二维排序图。评价时六方法统一使用目标概率的21点删除AUC，以及固定方法seed下五次RGB噪声稳定性。结果需要同时看忠实性、稳定性和耗时。真实案例215与298说明个体差异和退化检查的重要性；不能由图好看或分类置信度高推出解释可靠。正式分析始终使用统一DCU包，旧GPU结果只保留历史运行身份。

## 答辩问答

| 问题 | 回答 |
| --- | --- |
| 为什么不直接解释预测类别？ | metadata target 在同图六方法之间固定；错预测图解释的仍是真值目标。主分析采用命中组，全eval作敏感性分析。 |
| Occlusion 是否需要梯度？ | 核心以替换输入后的输出差归因，不依赖模型梯度；本实现仍有完整 wrapper 与目标有效性检查。 |
| 黑色为什么不是零张量？ | RGB=0归一化后是-mean/std；归一化张量0对应均值颜色，两者干预不同。 |
| 归因和删除评价都用概率吗？ | Occlusion对target logit归因；MoRF评价用softmax/sigmoid目标概率。同目标类别不意味着同响应量。 |
| 稳定性高为什么还可能AUC差？ | 两者问不同问题。固定输入噪声下排序相近不保证删除能降低目标分数；数值、配对证据和案例分别报告。 |
| 五次重复验证了采样收敛吗？ | 没有。这里是固定方法seed的RGB噪声稳定性；Occlusion本身的窗口流程与LIME/RISE随机采样也不同。 |
| 为什么常量图不算完全稳定？ | Spearman在常量时未定义，按协议score=0并报告valid_rate；不能把没有变化误称有意义的稳定解释。 |
| min-max后能比较贡献大小吗？ | 只能在本图空间排序语境使用；零点和幅度已改变，不能跨方法读作相同logit效应。 |
| Occlusion时间包含哪些开销？ | 单次attribute wrapper含校验forward、遮挡构造、批量推理、投影和CPU返回；读图、独立预测、MoRF与写盘在外。 |
| 现有复核等于重新生成实验吗？ | 不等于。本轮读取冻结源码、CSV/trace和归档预览，未启动模型推理、GPU归因或全矩阵。 |

学校 DeepSeek 曾获授权用于简单问答整理；本轮一次有界生成请求超时，未得到可采用文本，以上问答由 B 本轮直接编写和复核，不声称获得第二模型独立审计。
