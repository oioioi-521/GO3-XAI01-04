# B：模型与评价协议复核及报告补充正文

2026-10-26 · B / YaoYang-123456 · Refs #19。本文供 A 按 PR #23 的章节责任合稿，不覆盖 A 的全文。PR #23 `b3df72b16199e8d785723c421b06f1e3f4f77f8c` 和 PR #24 `507981464a770ee92dc43b00567f9b42f81071f4` 均为待审草稿。

现行分析基线为 `dafb65591349d426637855a6d4b6c9b52b979b64`，继承指定的 `29895378e930138e042fd29ad029d0f14e8b9186`。新增提交含案例请求工具及 RQ 文案校正，没有改变本次正式 CSV。执行事实取冻结包源码，不由当前 runner 的注释或函数默认值推定。

## 正式输入与本轮实际核验

| 输入 | SHA-256 | 本轮核验范围 |
| --- | --- | --- |
| `dcu_team_handoff_20261014.zip` | `14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc` | ZIP CRC；720 manifest 文件、721 checksum 项；36 单元；66,240 指标行；82,800 trace；90 份配置及 cohort/target/预测/权重绑定 |
| `dcu_analysis_handoff_20261026.zip` | `a1ae2e245e3a96fd2790c4d61a3bbb8c61c5f25bbc5ee2d06f10436f19c2894a` | CRC；1001 manifest 文件、1002 checksum 项；722 项原轻量包内容逐字节一致；全量资产目录 16,560 行对照逐图指标、预测、cohort 和配置 |

新包 `manifest.scope=analysis_only_no_image_assets`。README 明确没有原图/NPY/权重本体；TEAM_NOTICE 的完整包描述与本包范围不同，`evidence/packaging_validation.json` 是组长构建完整包时的记录，不能转述为 B 本轮重验全部 16,560 张 map。具体本轮回执见 [evidence.json](../analysis/b_m5_20261026/evidence.json)，冻结执行源码逐文件 hash 见 [源码索引](../analysis/b_m5_20261026/frozen_source_inventory.csv)。本轮不执行 ZIP 中的代码。

## §2.1 三种样本口径：复核后可保留原表

| 数据集 | 模型 | 全 eval | 本模型命中 | 三模型共同命中 |
| --- | --- | ---: | ---: | ---: |
| ImageNet | ResNet50 | 460 | 408 | 332 |
| ImageNet | DenseNet121 | 460 | 371 | 332 |
| ImageNet | VGG16 | 460 | 363 | 332 |
| VOC | ResNet50 | 460 | 387 | 353 |
| VOC | DenseNet121 | 460 | 388 | 353 |
| VOC | VGG16 | 460 | 400 | 353 |

主分析的六方法按本模型命中组同图配对；跨模型分析按同数据集三模型共同命中组配对；全 460 张用于敏感性分析。`per_image.csv` 唯一键为 `dataset,model,method,image_id,metric`；预测/cohort 连接键为 `dataset,model,image_id`；跨模型连接键为 `dataset,image_id`。本轮从 metadata 的 eval ID、预测 target 与 predicted_class_id 重新推导标记，并对照两种筛选表、均值/SD/n；不是只抄计数表。

VOC 命中仅表示**冻结的单个 target 是 sigmoid 输出 argmax**。不能写成多标签 exact match、完整标签预测正确、mAP 或所有物体均识别正确。即使命中，解释的也是该冻结目标，不代表图中其他标签。五次输入扰动在图内先聚合；460×5 不是独立样本量。课程冻结子集每数据集共 500 张，40 debug 与 460 eval 分离；历史筛选后的子集结论不外推为完整数据集精度。

## §2.2 模型与预处理：可直接补入正文

使用 ResNet50、DenseNet121、VGG16，推理均为 eval/FP32。ImageNet 使用 torchvision 默认预训练 1000 类输出，其类别索引沿用 torchvision/ImageNet-1k 顺序，以 metadata 数值 class_id 选取输出坐标；不按当前子集类别重编号。VOC 将分类头替换为 20 维，加载 B 的多标签训练 checkpoint（BCEWithLogitsLoss），评价概率为逐类 sigmoid。模型工厂返回原始 logits；`output_activation` 控制预测/评价及使用概率的方法，并没有给整个模型追加统一激活层。

| 数据集 | 模型 | 权重文件 | 冻结 SHA-256 |
| --- | --- | --- | --- |
| ImageNet | ResNet50 | `resnet50-11ad3fa6.pth` | `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| ImageNet | DenseNet121 | `densenet121-a639ec97.pth` | `a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395` |
| ImageNet | VGG16 | `vgg16-397923af.pth` | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` |
| VOC | ResNet50 | `resnet50_voc20.pt` | `1e54867cf18ed02383ba329c944a9143871fc37fe268bec6e769a8d364ae134f` |
| VOC | DenseNet121 | `densenet121_voc20.pt` | `ab61772eaf3459842f4b1e34be9ec15062e02a940d521390d80df0371168feca` |
| VOC | VGG16 | `vgg16_voc20.pt` | `61640717641bfc1e9ae5d6172de667ce45755e6f046f458c848f70cb4a6c9d58` |

权重身份来源：正式 `predictions.csv`、`method_predictions.csv`、配置 prediction_context，以及新包 `indexes/checkpoints.json`。本轮逐方法与共享预测表绑定一致；本轮未重新读取权重二进制或执行模型推理，故上述是冻结身份复核，不是重新验算模型输出。VOC 类别从索引 0 起依次为：aeroplane、bicycle、bird、boat、bottle、bus、car、cat、chair、cow、diningtable、dog、horse、motorbike、person、pottedplant、sheep、sofa、train、tvmonitor。类别顺序须与 [checkpoint 清单](VOC20_CHECKPOINT_MANIFEST.json)一致。

VOC 训练 manifest 4037 张、验证 manifest 474 张；本轮按数值化 VOC ID 核对二者互斥，且均排除冻结的全部 500 张（包括 debug）。这证明已交 manifest 的隔离，不替代重新审计每次训练的现场数据读取。旧 checkpoint 清单的训练 CSV/版本 JSON hash 对应 CRLF 字节，当前 `.gitattributes` 导致 checkout 为 LF；本轮分别记录原字节 hash 和转为 CRLF 后的匹配，不把换行后的匹配叫作原字节相同，也不回写清单。排除清单原字节直接匹配。详见回执 `voc_isolation`。

所有输入转 RGB，直接 `Resize((224,224))`、ToTensor 到 [0,1]，再用 mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225) 归一化；不使用权重默认 center-crop 管线，因此当前命中率不宣称复现 torchvision 官方精度。统一黑色 baseline 是原 RGB=0，对应归一化后的 `-mean/std`，不是归一化张量全零。数据版本见 [DATA_VERSION](../data/DATA_VERSION.json)与 [metadata](../data/metadata.csv)。

## §2.3 补充：同一 target 不等于同一响应量

以下逐项读取 `source/five/release/` 与 LIME 的实际 overlay；不能仅看配置的激活字段。

| 方法 | 冻结执行调用与响应 | 正式预算 / 输出处理 |
| --- | --- | --- |
| IG | `CaptumIntegratedGradients(self.model)`，选择 target logit | 50 步 Gauss-Legendre，internal batch10；有符号通道均值后 min-max |
| Grad-CAM | `LayerGradCam(self.model,target_layer)`，对 target logit 求梯度 | ReLU、通道求和、双线性上采样，再除最大值；三层为 layer4.2 / features.norm5 / features.28 |
| Occlusion | `CaptumOcclusion(self.model)`，替换窗口后的 target logit 差 | 32×32，stride16，扰动 batch16；有符号通道均值后 min-max |
| KernelSHAP | `KernelShap(self.model)`，拟合 target logit | 2048 samples，7×7 特征网格，扰动 batch16，seed42；通道均值后 min-max |
| RISE | 在 masked logits 后显式 softmax/sigmoid，再取 target 分数加权 mask | 4000 masks，7×7，p=.5，batch32，seed42；最后 min-max |
| LIME | 实际 `VectorizedLIME` 在 neighborhood 内对 logits 激活，Ridge 拟合目标概率 | 8000 samples，batch32，quickshift，采样/分割 seed42，cosine kernel width .25，Ridge alpha1 带截距；有符号超像素系数投影后 min-max |

四种 Captum 方法没有通过 predict 概率接口归因；RISE/LIME 的响应则是 ImageNet softmax 或 VOC sigmoid 概率。模型、图片、目标类别和评价协议一致，但响应量、特征粒度、归一化仍不同，属于本次被比较实现的一部分。保存的二维 [0,1] 图主要用于排序与展示，不能跨方法比较绝对贡献幅度。min-max 后零表示本图最小值，不必然是原贡献为零；全常量图另按退化规则处理。LIME overlay 文件头保留早期 prototype 注释，正式接入由 `run_lime_vectorized_full.py` 的 `execution_partition(cfg,VectorizedLIME)`、执行配置 `implementation` 与 overlay manifest 共同证实，不能只看文件头断言未使用。

## §2.4 指标与测量边界：可直接补入正文

**raw MoRF。** 对保存的二维图按值降序排序；在比例 f=0,.05,…,1 共 21 点，删除 `round(f×224×224)` 个空间位置，三个 RGB 通道一起替换为黑色 baseline。`faithfulness_morf_auc_raw` 计算 metadata 目标概率：ImageNet 为 softmax，VOC 为 sigmoid；按比例作梯形积分，数值越低越好。它不除以原图目标分数，也不是准确率下降率。归因响应为 logit 的方法仍统一用上述概率评价。低 AUC 仅表示该替换/排序协议下目标分数较低；初始分数、baseline、删除后分布变化及并列排序都会影响结果。常量图没有有意义的排序，不能由其 AUC 推断定位正确。ImageNet/VOC raw AUC 分开解释，不能把绝对值混合排名或直接作同尺度三因素模型。正式逐点曲线未保存，当前包支持 AUC 核对，不能由 AUC 还原整条曲线。

**输入噪声稳定性。** 把预处理后的张量还原到 RGB [0,1]，加 sigma=.005 的高斯噪声、裁剪，再归一化。每原图五次重复；seed 由 dataset、image_id、repeat 的 SHA-256 派生，同图跨方法/模型配对。各随机方法使用固定方法 seed；比较原图参考 map 与扰动图 map 全像素 Spearman，不是重新采样方法 seed 的方差。先按图平均五个 score，然后统计图间均值与样本 SD（ddof=1）。本轮验证 82,800 条 trace、4600 个配对 seed 条目与逐图聚合。

**退化与 valid_rate。** 常量/未定义相关的 spearman 为空，score 按冻结协议为 0，状态保留；每图 valid_rate=有效重复数/5。全矩阵 28 条退化均为 Grad-CAM，没有删掉后重算更好看的均值；valid_rate 是相关系数定义率，不是解释正确率。原图与五次噪声 map 并非全部归档，故本轮验证 trace/seed/聚合一致性，未独立重算噪声 map 的相关系数。

**效率。** 冻结 runner 先进行一次不计时归因预热；逐图独立预测在计时外。同步设备后计时，调用一次完整 `attribute` wrapper，再同步后停止；包括 wrapper 内的实际校验 forward、mask/特征/分割构建、拟合、归一化与 CPU 返回等开销，不是纯网络 forward/backward 时间。读图、模型加载、写 NPY/PNG/CSV、MoRF 与稳定性阶段在此单次原图计时之外。LIME 60 分片每片单独预热，并非全实验只有一次预热。

前五方法和 LIME 来自两轮共享八 DCU 运行（墙钟分别 4:21:06.2、3:52:06.7）；相加 8:13:12.9 不代表六方法同队列一次运行。并发和负载随时间变化，逐图毫秒不能直接相加当队列墙钟，也不能外推独占硬件速度保证。历史 RTX/Windows 是独立运行身份，不拼入当前 36 单元或合算同机速度比。

## 给 A 的精确合稿建议

- §2.1 的 n 表与单目标口径正确，可保留；在版本注释中加入本轮连接键及隔离复核入口。
- §2.2 保留现有段落，增加上述六权重表、VOC 完整类别顺序、模型工厂 logits 说明及训练清单换行边界。
- §2.3 原响应量表经冻结执行链复核正确，不改成“六方法解释相同概率”；注明 LIME 实际 overlay。
- §2.4 保留已有定义，补 `round` 的空间删除、并列/常量排序限制、分片预热和 wrapper 内开销；不把未存曲线/噪声图写成可独立重算。
- §8 追加 [真实案例初稿](B_M5_CASE_ANALYSIS.md)；Occlusion 详述采用 [方法与答辩材料](B_M5_OCCLUSION_AND_DEFENSE.md)。本轮不重写报告其他章节。
