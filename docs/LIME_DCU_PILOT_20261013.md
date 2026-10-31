# LIME 八卡 DCU 实际耗时试跑

日期：2026-10-13（Asia/Shanghai）。分支：`feat/dcu-throughput-20261013`。

## 结论及范围

在 node62 的官方 Docker 实例中，完成三模型 × 两数据集的真实 LIME 试跑。
2,000 个扰动样本、batch 32、FP32 下，每次解释平均 6.955–8.146 秒。
相同 16 张图，一卡计算阶段 110.585 秒，八卡 14.870 秒，加速 7.437 倍。
六单元各 460 图、每图原图加五次稳定性解释，共 16,560 次解释，按本次八卡
墙钟时间外推约 4.520 小时；加 25% 余量为 5.650 小时，建议预留约 6 小时。

这是 **官方 LIME 的独立容量试跑**。尚未取得 E 可独立复核的正式 adapter 和六份
冻结配置，不能称为对 E 实现的验收，也不能将本次 debug 结果混入正式方法排名。
2000 是本次明确选定的候选预算，并未代替 E 冻结正式预算。已有材料的限制见
[D 的 LIME 接收记录](D_LIME_MIDTERM_RECEIPT.md)。

## 环境和实验协议

- DAPJobs 网页控制台及文件面板接入 node62，复用实例 #78 `dev-59d5af49`。
- 56 CPU / 256 GiB 主存 / 8 张 BW DCU，每卡约 64 GiB；每卡一个进程，
  每个进程只看到一张卡，CPU/BLAS 线程数为 2。没有 CPU 模型推理回退。
- 镜像：`harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-sf_b075`。
- PyTorch 2.10.0、torchvision 0.25.0、HIP 6.3.26113；保留厂商版本。
- 新依赖装入 `.lime-deps`：lime 0.2.0.1、scikit-image 0.25.2、imageio 2.37.0、
  tifffile 2024.9.20、lazy_loader 0.4；NumPy 1.25.0、SciPy 1.15.3、sklearn 1.7.2。
- 三个 ImageNet 模型使用 torchvision 官方预训练权重；VOC 使用项目冻结的
  VOC20 权重，传输前后验证 SHA-256，与 `VOC20_CHECKPOINT_MANIFEST.json` 一致。
- 输入来自冻结 `metadata.csv` 的 debug 子集，224×224、项目预处理、metadata 真值
  目标；ImageNet 用 softmax，VOC20 用 sigmoid，FP32，无混合精度。
- 官方 `LimeImageExplainer`：2000 samples、batch 32、kernel width 0.25、
  feature selection auto、num_features 100000、seed 42、hide_color 0（RGB 黑色）。
- quickshift：kernel_size 4、max_dist 200、ratio 0.2、convert2lab=True、rng 42。
  每次原图或噪声图重新分割，不跨噪声重复复用分割结果。仅拟合指定真值类别。
- 将全部超像素的有符号系数展开为 224×224 图，再 min-max 归一化至 [0,1]。
- 每个组合 8 张图片；每图原图解释一次，再按冻结协议做五次 RGB Gaussian
  噪声解释（sigma 0.005、固定 dataset/image_id/repeat seed），并计算 Spearman。
- 每个原图做 21 点 raw MoRF，归一化后的 RGB 黑色基线，采用对应数据集概率语义。

实现参考：[官方 LIME 图像解释源码](https://github.com/marcotcr/lime/blob/master/lime/lime_image.py)、
[scikit-image quickshift 文档](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.quickshift)。

每个 worker 先完成一次完整解释预热；全部就绪后共同开始测量。解释计时包括 RGB
转换、分割、扰动构造、CPU/DCU 传输、前向推理、加权回归和解释图展开，前后同步设备。
下表“计算墙钟”还包括测量期内 NPY 保存、稳定性相关性和 MoRF；“含启动总时长”
另含解释器启动、导入、权重加载、校验、预热和协调。预加载图片的读取不在计算墙钟内。

## 六组合实测

每行均为八卡并发、8 张原图 × 6 次解释 = 48 次测量解释，另含 8 次 MoRF。
P95 是这 48 次解释的经验分位数，不能当作正式数据集的置信界限。

| 模型 / 数据集 | 每次解释均值 s | P95 s | 计算墙钟 s | 含启动总时长 s | 外推 460 图全流程 h |
|---|---:|---:|---:|---:|---:|
| ResNet50 / ImageNet | 6.978 | 7.241 | 43.983 | 80.510 | 0.702 |
| ResNet50 / VOC | 6.955 | 7.190 | 43.774 | 79.507 | 0.699 |
| DenseNet121 / ImageNet | 7.517 | 7.851 | 46.974 | 84.114 | 0.750 |
| DenseNet121 / VOC | 7.368 | 7.590 | 45.979 | 83.114 | 0.734 |
| VGG16 / ImageNet | 8.140 | 8.345 | 50.366 | 89.319 | 0.804 |
| VGG16 / VOC | 8.146 | 8.657 | 51.893 | 91.926 | 0.829 |

原图与噪声图的平均解释成本接近。稳定性需要额外五次完整解释，因此约占归因
计算量的 5/6；并非稳定性相关系数本身很慢。

分段计时显示，分割约 0.55 秒，推理回调（含数据转换/传输）约 1.48–2.64 秒，
回归约 0.006 秒，“扰动生成及其余处理”的剩余时间约 4.89–5.00 秒。
后者约占解释时间 61%–71%，包含官方 LIME 的 Python/NumPy 扰动构造及未单独计时的
距离、图展开等操作，不能全部标成纯 GPU 或纯 CPU 内核时间。后续若优化，应优先
检查这部分；仅增大 GPU batch 的收益需要另测。本次没有重写 LIME 的采样算法。

最大单卡 PyTorch allocated 峰值约 1.81 GiB（VGG16），没有出现显存压力。
一份运行期硬件快照中 junction 温度为 56–59°C，各卡瞬时利用率不同；这不是持续
监控的最高温度，也不能声称八卡始终满利用率。

## 相同工作量的一卡 / 八卡对照

ResNet50 / ImageNet，同一批 16 张冻结 debug 图，2000 samples、batch 32。
此对照只比较解释和输出保存，不包含五次噪声重复或 MoRF。

| 并发方式 | 计算墙钟 | 含启动总时长 | 计算吞吐 |
|---|---:|---:|---:|
| 一卡，一个进程 | 110.585 s | 144.560 s | 0.145 次/s |
| 八卡，八个进程 | 14.870 s | 51.277 s | 1.076 次/s |

计算阶段加速 **7.437×**；端到端加速 **2.819×**。所有 16 张解释图逐元素完全一致，
最大绝对差为 0；最小 Spearman 约为 1。正式任务应让 worker 持续处理分片，避免
每张图重启进程。这里比较的是同一台 DCU 的一卡/八卡，不是 RTX 2080 Ti 对照。

## 服务器时间预算

直接采用本次实测的每组八卡墙钟，而不是把单卡均值简单除以八：

```text
全量 LIME 时间 = Σ(该组合的 8 图计算墙钟 × 460 / 8)
              = 4.519641 小时
加 25% 规划余量 = 5.649551 小时
```

- 六单元基础解释加 MoRF、不做五次稳定性：理想均匀分工约 0.772 小时，
  可先按 45–60 分钟估算。
- 六单元基础加五次稳定性：16,560 次解释，外推约 4.5 小时，预留约 6 小时。
- 若沿用此前其他五种方法约 6–8 小时的规划估算，36 单元完整重跑合计约
  12–14 小时，可预约 16 小时窗口。其他五种方法此次没有重测，这不是全矩阵已跑完的时间。

本次每组合仅 8 张不同的 debug 图，不能覆盖 460 张 eval 的内容复杂度、重试、I/O
和平台排队；25% 是排期余量，不是统计置信区间。假设八卡独占、常驻进程、模型及
数据已就位、预算与本次一致。若 E 的最终分割器、samples 或实现改变，应重新校准；
本次不包含科研门禁、代码集成及首次部署的人工时间。

## 加入 LIME 后的 36 单元总矩阵时间表

本节于同日按两份已核验的 DCU 结果重新合并计算。口径为 6 方法 × 3 模型 × 2 数据集，
每单元 460 图，每图 1 次原图解释、5 次噪声解释，原图另做 MoRF。共 16,560 个
方法/模型/图片记录、99,360 次解释。**下面是容量外推，不是 36 单元已实际重跑。**

每个单元都按占用整台八卡服务器计算，表中各格可以相加，合计后不能再除以 8。
单位为分钟，含五次稳定性；暂不含进程启动、准备、重试等规划余量。

| 方法 | ResNet / ImageNet | ResNet / VOC | DenseNet / ImageNet | DenseNet / VOC | VGG / ImageNet | VGG / VOC | 六单元合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Grad-CAM | 0.4 | 0.4* | 0.8 | 0.8* | 0.1 | 0.1* | 2.5 |
| IG | 0.9 | 0.9* | 2.2 | 2.2* | 0.9 | 0.9* | 8.0 |
| Occlusion | 3.4 | 3.4* | 7.6 | 7.6* | 5.5 | 5.5* | 33.1 |
| RISE | 12.9 | 12.9* | 17.1 | 17.1* | 25.5 | 25.5* | 111.0 |
| KernelSHAP | 12.8 | 12.8* | 22.8 | 22.8* | 19.5 | 19.5* | 110.2 |
| LIME | 42.1 | 42.0 | 45.0 | 44.1 | 48.3 | 49.7 | 271.2 |
| **合计** | | | | | | | **535.9 分钟，约 8.93 小时** |

带 * 的 VOC 格子借用同模型 ImageNet 耗时，仅用于预算占位。第一至第五种方法
均主要依据先前各模型一图兼容性检查，假设噪声解释成本与原图相同、八卡理想分片；
ResNet 的 RISE/KernelSHAP 有额外的真实八卡扩展性佐证，其他格子没有同等强度的验证。
LIME 六格均使用本次对应模型/数据集八卡 8 图的全流程墙钟外推。
小于一分钟的格子尤其不能当成端到端承诺；启动、相关性计算与 I/O 对快速方法影响更大。
合计由未四舍五入数值计算。

第一至第五种方法的公式为：

```text
单元分钟 = 460 / 8 × (一图计算墙钟秒 + 5 × 原图解释秒) / 60
```

一图计算墙钟已经包含原图解释、MoRF 和保存。此公式尚未完整计入五次噪声的 CPU
相关性与输出开销，须使用下表排期余量。KernelSHAP 保留 batch 16 依据，没有将
仅在 ResNet 验证的 batch 128 收益推广到全部模型。Occlusion 采用本次更密的
15/8 扫描作为容量代理，正式 32/16 尚未在 DCU 校准，不预先扣减时间。
DenseNet Grad-CAM 本次层为 denseblock4，正式 norm5 也待校准；VOC RISE 需要先
统一 sigmoid 语义。第一至第五种方法来源为 `analysis/dcu_benchmark_20261013.json`。

| 方法（各六单元） | 基础解释及 MoRF | 加五次稳定性后的完整外推 | 建议预留 |
|---|---:|---:|---:|
| Grad-CAM | 1.7 分钟 | 2.5 分钟 | 10–15 分钟 |
| IG | 2.6 分钟 | 8.0 分钟 | 15–25 分钟 |
| Occlusion | 6.8 分钟 | 33.1 分钟 | 45–60 分钟 |
| RISE | 21.1 分钟 | 111.0 分钟 | 2.5–3 小时 |
| KernelSHAP | 20.9 分钟 | 110.2 分钟 | 2.5–3 小时 |
| LIME | 46.3 分钟 | 271.2 分钟 | 6 小时 |
| **全矩阵** | **约 1.66 小时** | **约 8.93 小时** | **约 12.2–13.7 小时** |

建议对外排期表述为 **12–14 小时，预约 16 小时窗口**。这些预留范围是运行计划，
不是统计置信区间。基础评估的 1.66 小时也是假设理想分片的计算量参考。
LIME 基础耗时由原图及 MoRF 均值理想分片估计；完整耗时则来自实测八卡墙钟，
两者相减包含负载不均衡等剩余开销，不能全称为稳定性纯计算。

全矩阵外推中，LIME 约占 51%，RISE 和 KernelSHAP 合计约 41%，这三种方法约占 92%。
机时主要花在这三种方法，Grad-CAM / IG 的重复运行成本较小。本节不含重新训练模型、
科学门禁、修改实现或排队等待，也不据此推导与 RTX 2080 Ti 的速度比。

逐格输入、公式、证据强度和未舍入结果见
[`analysis/dcu_full_matrix_estimate_20261013.json`](../analysis/dcu_full_matrix_estimate_20261013.json)。

## 验证与证据

主试跑 8 组全部成功：六组共 288 次解释，加扩展性对照 32 次，共 **320 次测量**。
另有 1 组单图冒烟检查，含原图和一次噪声解释；预热解释均不计入上述测量数量。
主试跑含启动累计 704.326 秒（约 11 分 44 秒），不含部署与上传准备。

下载后用独立核验器检查每个 worker、冻结图片/目标、噪声 seed、设备可见性、
VOC 权重哈希、NPY 哈希、224×224 float32、有限值、[0,1]、每图 2000 次预测、
无遗漏/重复、各计时阶段聚合、墙钟吞吐，并重算所有稳定性分数，全部通过。
320 张测量解释图均非常量。脚本通过 Python 语法编译；没有新增正式结果 CSV。

- 试跑入口：[`scripts/benchmark_lime_dcu.py`](../scripts/benchmark_lime_dcu.py)。
- 独立核验：[`scripts/audit_lime_dcu.py`](../scripts/audit_lime_dcu.py)。
- 可提交的数值摘要：[`analysis/lime_dcu_pilot_audit.json`](../analysis/lime_dcu_pilot_audit.json)。
- 本地完整原始证据（忽略目录）：`results/lime-server-evidence/`，包含所有 maps、
  worker 日志、环境、原始 manifest/summary、执行脚本和服务器归档。
- 归档 `lime_dcu_evidence_20261013.tar.gz`，1,429,747 字节，服务器/本地 SHA-256 一致：
  `949bbd2d0fd3777c6528a26f02cc9fdd4558a527358e3b08c2d21422e285408e`。

下载采用“预先等待 download → 同一标签页导航到已确认 URL → 读取最终 path →
验证哈希”流程，正常得到最终 `.tar.gz`，没有停在临时文件。

本次已停止 #78 八卡实例，恢复之前暂时停用的 #75 一卡终端；#76/#77 保持停止。
实例及实验文件保留供复用，没有删除。恢复状态：

![实验结束后的实例状态](../results/lime-server-evidence/instances-restored.png)

## 复现示例

在相同镜像中准备好项目冻结图片、三份官方 ImageNet 权重和 VOC20 checkpoint，
以及上述额外依赖，使用一个新的输出目录：

```bash
PYTHONPATH=.lime-deps:.deps:. python scripts/benchmark_lime_dcu.py \
  --model resnet50 --dataset imagenet --images 8 \
  --workers 8 --devices 0,1,2,3,4,5,6,7 \
  --samples 2000 --batch 32 --noise-repeats 5 --morf \
  --output results/lime-new-pilot/resnet50-imagenet-8card
```

整批八组的确切命令位于证据包的 `results/run_lime_pilot_20261013.sh`。
独立校验下载后的整批结果：

```bash
python scripts/audit_lime_dcu.py \
  results/lime-server-evidence/results/lime-matrix-20261013
```
