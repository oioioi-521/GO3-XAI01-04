# 八卡 DCU 前五种方法全量结果（2026-10-13）

**30/30 组正式计算及服务器端完整产物复核均通过。** 本轮不含 LIME。15:23:14 开始，19:44:20 完成，正式队列墙钟 **4 小时 21 分 6 秒**（15,666.17 秒）。这一时间包含正式任务加载、预热、原图归因、21 点 raw MoRF、五次噪声归因、结果保存、单元校验、续跑检查和合并；不含前期部署/debug 检查，也不含结束后的独立复核及传输。

## 完整性与溯源

- 30 组均有成功完成标记和续跑检查；控制器退出码 0，没有控制器或单元 failure 文件。
- 每组 460 张 eval；原图组合 13,800，五次噪声重复 69,000，总计 82,800 次正式归因。
- 55,200 行逐图指标、120 行单元汇总、69,000 行稳定性 trace；逐图键和重复编号完整、无重复，合并数据与各单元一致。
- 核对 85 个发布源文件/配置、1,000 张冻结数据、六个模型权重的 SHA-256；13,800 个 float32 224×224 NPY 的有限值、范围、哈希和溯源绑定通过，13,800 个 PNG 可读取。
- 核对真值 target、稳定性种子与 sigma、均值及样本标准差、valid rate、逐图五次重复均值；同一模型/数据集在五个方法下预测一致。
- Grad-CAM 有 28/13,800 条噪声重复遇到常量图：DenseNet/VOC 10 条，VGG/ImageNet 13 条，VGG/VOC 5 条。按冻结协议保留 undefined correlation 的空值、score=0 和 valid rate；不是丢样本，也未人工修图。其余四方法均为有效重复。

具体配置、部署位置和恢复流程见 [部署记录](DCU_FULL_RUN_20261013.md)。完整复核记录见 `results/formal-final-evidence/run/final_audit.json`。

## 用时与稳定性概览

下表范围是六个模型/数据集单元的平均值范围；累计归因时长是并发任务的测量区间求和，**不是实际墙钟，也不是一张卡完整串行重跑的实测值**。该项不含 MoRF、I/O、模型加载等开销。

| 方法 | 单次原图归因均值范围（ms） | 稳定性均值范围 | 原图+噪声累计归因时长（h） |
| --- | ---: | ---: | ---: |
| Grad-CAM | 6.99–54.80 | 0.9318–0.9989 | 0.132 |
| IG | 124.56–318.12 | 0.5726–0.7964 | 0.914 |
| Occlusion | 145.79–336.82 | 0.9154–0.9921 | 1.065 |
| RISE | 2180.64–4366.03 | 0.9952–0.9993 | 14.496 |
| KernelSHAP | 2194.65–3891.30 | 0.9788–0.9959 | 14.486 |

归因测量区间合计 31.093 小时，其中 RISE 和 KernelSHAP 占 93.2%；噪声归因占 83.4%。这解释了全量稳定性最耗时，也说明把长任务分散到多张卡的价值。

IG 在六个模型/数据集组合中有五组 raw MoRF AUC 均值最低；DenseNet/ImageNet 的最低均值来自 KernelSHAP。Grad-CAM 在六组中均最快。RISE 的固定采样种子输入噪声稳定性很高，IG 较低。这里是描述性结果，没有据此声明统计显著性或唯一最优方法；应在每个模型/数据集内部做配对检验和权衡。

## 与桌面 RTX 2080 Ti 的已有正式运行对照

来源：[A 正式运行报告](A_IG_GRADCAM_RESULTS.md) 与 `analysis/a_ig_gradcam_summary.csv`。这 12 组使用同一冻结 eval、相同权重 SHA-256、IG/Grad-CAM 主要算法参数及计时边界。RTX 2080 Ti 为 PyTorch 2.8 / torchvision 0.23 / CUDA 12.8，本次 DCU 为厂商 PyTorch 2.10 / torchvision 0.25 / HIP 6.3；CPU、软件版本与并发条件不同。因此下列为历史运行耗时对照，不是隔离硬件变量的加速实验。

| 方法 | 模型 | 数据集 | RTX 2080 Ti（ms） | 本次 DCU（ms） | 旧耗时 / 新耗时 |
| --- | --- | --- | ---: | ---: | ---: |
| Grad-CAM | densenet121 | imagenet | 19.75 | 54.60 | 0.36 |
| IG | densenet121 | imagenet | 191.83 | 318.12 | 0.60 |
| Grad-CAM | resnet50 | imagenet | 8.11 | 24.03 | 0.34 |
| IG | resnet50 | imagenet | 168.76 | 131.47 | 1.28 |
| Grad-CAM | vgg16 | imagenet | 10.09 | 6.99 | 1.44 |
| IG | vgg16 | imagenet | 298.09 | 142.99 | 2.08 |
| Grad-CAM | densenet121 | voc | 19.23 | 54.80 | 0.35 |
| IG | densenet121 | voc | 191.64 | 315.77 | 0.61 |
| Grad-CAM | resnet50 | voc | 7.81 | 24.02 | 0.33 |
| IG | resnet50 | voc | 168.39 | 124.56 | 1.35 |
| Grad-CAM | vgg16 | voc | 10.00 | 7.06 | 1.42 |
| IG | vgg16 | voc | 297.53 | 143.94 | 2.07 |

比值大于 1 表示本次 DCU 运行更快：VGG16/IG 约 2.07–2.08，ResNet50/IG 约 1.28–1.35；DenseNet/IG 约 0.60。Grad-CAM 在 VGG16 上约 1.42–1.44，在 ResNet/DenseNet 上约 0.33–0.36。不能声称单张 DCU 全面超过 2080 Ti。八卡优势体现在独立任务并发；没有同矩阵的 2080 Ti 完整墙钟，不能给整矩阵硬件加速比。

模型预测也存在少量跨运行差异：ImageNet/ResNet 命中从 409 变为 408；VOC/ResNet 从 385 变为 387，DenseNet 从 387 变为 388，VGG 从 402 变为 400。权重哈希相同，但不能假定跨软件栈逐图数值完全一致；原因尚未隔离。VOC 这里是冻结单标签 target 与 sigmoid argmax 的命中数，不是多标签 mAP。

## 与服务器试跑的关系

本次全部采用正式参数。此前 DCU 冒烟使用 debug 图片，而本次是 460 张 eval；同参数的 11 组 ImageNet 归因均值相对此前一图试跑约变化 -6.5% 至 +8.3%，仅说明量级相近，不构成统计检验。Occlusion 的试跑窗口 15/stride 8 与本次 32/16 不同，DenseNet Grad-CAM 的层也从试跑 denseblock4 改为正式 norm5，均排除在上述范围之外。

你的 2080 Ti KernelSHAP 7×7/2,048 样本试跑均值约 3.067 秒；本次 DCU ResNet/ImageNet 正式均值约 2.200 秒。前者是 10 张 debug，后者是 460 张 eval，样本与软件栈不同，只作为成本参考，不写成受控硬件加速比。

## 30 组正式均值

每行 n=460；完整标准差保存在原始 `merged/units.csv` 和机器可读汇总。raw MoRF 越低越好；稳定性越高越好；valid rate 单列。效率只含归因 wrapper，不含 prediction、MoRF 和文件 I/O。稳定性固定方法 seed，因此不代表重新抽取 RISE/KernelSHAP 样本时的稳定性。

| 方法 | 模型 | 数据集 | 归因 ms | raw MoRF AUC | 稳定性 | 有效率 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Grad-CAM | resnet50 | imagenet | 24.029 | 0.087625 | 0.989558 | 100.000% |
| Grad-CAM | resnet50 | voc | 24.015 | 0.538453 | 0.976489 | 100.000% |
| Grad-CAM | densenet121 | imagenet | 54.602 | 0.056883 | 0.998853 | 100.000% |
| Grad-CAM | densenet121 | voc | 54.796 | 0.436197 | 0.992264 | 99.565% |
| Grad-CAM | vgg16 | imagenet | 6.990 | 0.243221 | 0.931802 | 99.435% |
| Grad-CAM | vgg16 | voc | 7.059 | 0.337817 | 0.959116 | 99.783% |
| IG | resnet50 | imagenet | 131.473 | 0.038469 | 0.673544 | 100.000% |
| IG | resnet50 | voc | 124.559 | 0.443193 | 0.572564 | 100.000% |
| IG | densenet121 | imagenet | 318.119 | 0.044523 | 0.761713 | 100.000% |
| IG | densenet121 | voc | 315.771 | 0.327278 | 0.712287 | 100.000% |
| IG | vgg16 | imagenet | 142.989 | 0.021618 | 0.796447 | 100.000% |
| IG | vgg16 | voc | 143.940 | 0.113455 | 0.776793 | 100.000% |
| Occlusion | resnet50 | imagenet | 145.789 | 0.193134 | 0.916653 | 100.000% |
| Occlusion | resnet50 | voc | 148.157 | 0.610909 | 0.915388 | 100.000% |
| Occlusion | densenet121 | imagenet | 335.796 | 0.081488 | 0.987803 | 100.000% |
| Occlusion | densenet121 | voc | 336.820 | 0.453094 | 0.987863 | 100.000% |
| Occlusion | vgg16 | imagenet | 210.001 | 0.039638 | 0.990122 | 100.000% |
| Occlusion | vgg16 | voc | 210.012 | 0.256511 | 0.992078 | 100.000% |
| RISE | resnet50 | imagenet | 2183.741 | 0.107078 | 0.995154 | 100.000% |
| RISE | resnet50 | voc | 2180.640 | 0.541116 | 0.997432 | 100.000% |
| RISE | densenet121 | imagenet | 2909.603 | 0.054209 | 0.998162 | 100.000% |
| RISE | densenet121 | voc | 2895.508 | 0.416223 | 0.999316 | 100.000% |
| RISE | vgg16 | imagenet | 4366.031 | 0.052458 | 0.996804 | 100.000% |
| RISE | vgg16 | voc | 4363.539 | 0.268402 | 0.997936 | 100.000% |
| KernelSHAP | resnet50 | imagenet | 2200.377 | 0.112884 | 0.992847 | 100.000% |
| KernelSHAP | resnet50 | voc | 2194.649 | 0.489404 | 0.978835 | 100.000% |
| KernelSHAP | densenet121 | imagenet | 3891.302 | 0.042007 | 0.995669 | 100.000% |
| KernelSHAP | densenet121 | voc | 3874.118 | 0.363608 | 0.994786 | 100.000% |
| KernelSHAP | vgg16 | imagenet | 3343.251 | 0.032818 | 0.995943 | 100.000% |
| KernelSHAP | vgg16 | voc | 3346.603 | 0.181658 | 0.995845 | 100.000% |

## 产物与归档

- 机器可读汇总：`analysis/dcu_full_results_20261013.json`。
- 指标、日志、参数、审计及部署源文件：`results/formal-final-evidence/dcu_full_five_20261013_tables.tar.gz`，14,585,000 字节。
- 完整包还包含所有 PNG / float32 NPY / 溯源 sidecar：`dcu_full_five_20261013_complete.tar.gz`，1,428,278,650 字节。传输与本地最终核验状态见下方收尾记录。
- 服务端原目录保持不变：`/workspace/xai-formal-five-20261013/results/dcu-full-five-20261013`；宿主前缀 `/opt/dapjobs-dev/cas_mfwzan/dev-59d5af49/`。

| 归档 | SHA-256 |
| --- | --- |
| tables | `39de02322ff697f12a69cd58842b7591f6c3c76f1805c938101f258eab789a86` |
| complete | `852eb7c8c4eb74124b6c599d2fd54bf1f067f9c0579ce9a0542dfada040dbede` |

数据包不重复包含模型权重和原始数据图片，其版本与 SHA-256 已在发布/数据清单中记录；这些数据和权重已存在于此前部署材料。该包包含复现实验所需的实现、正式配置、元数据及全部本轮结果。

## 收尾记录

完整包与精简包均已下载至本地 `results/formal-final-evidence/`，大小及 SHA-256 与服务器一致。完整包已安全解压到该目录，确认包含 13,800 个 NPY 和 13,800 个 PNG；三份合并 CSV 的本地哈希均与服务器审计一致。完整审计记录位于 `results/formal-final-evidence/run/final_audit.json`。

首次整包下载的工具等待超时，但浏览器在后台继续并完成了下载；核对本地文件后停止后续分片下载，采用已通过校验的完整包。工具等待超时不能单独作为下载失败的依据。

确认计算与审计进程均已退出后，通过 DAPJobs 页面停止 #78，恢复此前临时暂停的单卡实例 #75；页面已核实 #78 为“已停止”、#75 为“运行中”。实例及结果均保留。状态截图位于 `results/formal-final-evidence/instances-after-completion.png`。

归档和汇总完成后，自动检查 `dcu` 已暂停。未启动 LIME。
