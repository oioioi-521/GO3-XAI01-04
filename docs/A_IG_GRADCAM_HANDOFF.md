# 成员 A 交接说明：Integrated Gradients + Grad-CAM

更新时间：2026-09-20。

## 当前完成状态

- 已实现 Integrated Gradients（结果中的方法名为 `ig`）。
- 已实现 Grad-CAM（结果中的方法名为 `gradcam`）。
- 已将两种方法接入 `run_unit.py` 的统一归因接口、断点续跑、配置快照、预测表、逐图 CSV、单元汇总和日志流程。
- 已提供 **2 方法 × 3 模型 × 2 数据集 = 12** 个正式单元 YAML。
- 已提供算法、非法参数、目标类别、目标层解析、runner 执行/续跑、预热隔离和 12 配置矩阵测试。
- 已统一使用 `faithfulness_morf_auc_raw`、21 点删除曲线、1 次不计时预热，并同时保存 PNG 与 float32 NPY。
- **ImageNet 与 VOC 共 12 个单元的忠实性、效率正式运行已完成**：每单元 460 张、`processed=460, skipped=0`；稳定性尚未运行，所以按三指标口径仍未完全闭环。
- 完整测试为 **58 passed**；正式结果、验收、哈希与解释见 [`A_IG_GRADCAM_RESULTS.md`](A_IG_GRADCAM_RESULTS.md)。

2026-09-20 在台式机 RTX 2080 Ti 上使用 Python 3.13.9、PyTorch 2.8.0、torchvision 0.23.0 和 Captum 0.9.0 完成正式运行。结果同时保存在台式机与本地集成工作树的 `results/`；该目录由 Git 忽略，不随代码提交。

## 统一接口与方法约定

所有方法遵循：

```python
attribute(image, target, baseline) -> Tensor[H, W]
```

输入是单张归一化 RGB 图像，`target` 固定使用 metadata 中的真值类别；输出是 CPU 上有限的 `[0,1]` 空间热力图。runner 统一传入“归一化后的 RGB 黑色”作为 baseline，即原始像素 `0` 经 ImageNet mean/std 变换后的值，而不是归一化空间中的数值零。

### Integrated Gradients

- Captum `IntegratedGradients`，默认 50 步 Gauss-Legendre 积分。
- 以归一化 RGB 黑色图像为路径起点。
- 三通道原始归因作有符号算术平均，再作逐图 min-max 归一化；不取绝对值。
- 此变换保留当前 MoRF 指标使用的降序空间排名。返回值 `0` 表示该图的最低有符号归因，不一定表示原始零贡献。

### Grad-CAM

- Captum `LayerGradCam`，启用原论文使用的最终 ReLU，再双线性上采样到输入分辨率。
- 每份配置显式记录目标层，防止模型升级后静默换层：
  - VGG-16：`features.28`（最后一个卷积层）
  - ResNet-50：`layer4.2`（最后一个 bottleneck 输出）
  - DenseNet-121：`features.norm5`（最终归一化特征图）
- 非负 CAM 除以逐图最大值；全零 CAM 保持全零。

## 12 个正式配置

配置命名为：

```text
configs/ig_{vgg16,resnet50,densenet121}_{imagenet,voc}.yaml
configs/gradcam_{vgg16,resnet50,densenet121}_{imagenet,voc}.yaml
```

这些配置已经是正式矩阵口径：`split: eval`、`max_images: null`。不要在无意间直接启动；开发验证应使用测试套件，或在有权限后显式加 `--max-images 1` 做 GPU 首图检查。

## 正式跑批结果

2026-09-19 的 ImageNet ratio MoRF 与未预热计时已被 2026-09-20 的统一 raw AUC 正式结果取代，不得混入排名、ANOVA 或 Pareto。ImageNet 与 VOC 的完整表格、图、预测命中率、常量图记录和结果树校验和统一见 [`A_IG_GRADCAM_RESULTS.md`](A_IG_GRADCAM_RESULTS.md)。

## 正式跑批状态与剩余条件

1. 台式机访问和 CUDA/PyTorch/Captum 环境：已通过。
2. `preprocessing/verify_data_version.py` 和冻结的 460 张 ImageNet eval 图片：已通过；VOC 图片也已就位。
3. 三种 ImageNet 官方权重：已缓存并校验。
4. B 提供并冻结的下列 VOC-20 checkpoint 已完成 SHA256、strict-load、`(1,20)` 有限前向与正式实验验收：
   - `models/checkpoints/vgg16_voc20.pt`
   - `models/checkpoints/resnet50_voc20.pt`
   - `models/checkpoints/densenet121_voc20.pt`
5. 三种模型的一张首图检查、每单元 40 张 debug 及每单元 460 张 eval：均已通过。

## 复现与续跑顺序

```bash
python preprocessing/verify_data_version.py
python -m pytest -q

# 每个模型先做一张 eval 首图检查；该 CLI 限制会进入有效配置哈希。
python experiments/run_unit.py --config configs/ig_resnet50_imagenet.yaml --max-images 1
python experiments/run_unit.py --config configs/gradcam_resnet50_imagenet.yaml --max-images 1

# 人工确认热力图、真值类别、预测类别和输出目录后，再运行不带限制的正式配置。
python experiments/run_unit.py --config configs/ig_resnet50_imagenet.yaml
python experiments/run_unit.py --config configs/gradcam_resnet50_imagenet.yaml

# VOC 使用对应的 *_voc.yaml；模型输出激活由配置显式设为 sigmoid。
python experiments/run_unit.py --config configs/ig_resnet50_voc.yaml
python experiments/run_unit.py --config configs/gradcam_resnet50_voc.yaml
```

runner 会在配置哈希变化时清理同一方法/模型/数据集的旧单元行，所以首图检查不会被错误复用为全量结果。正式跑批支持逐图续跑，不需要为中断手工删除完整结果。

## 尚未完成及外部依赖

- A 的 IG / Grad-CAM 实现，以及 12 个主体单元的忠实性、效率、结果解释、汇总图和可复现产物已完成；稳定性不包含在本次完成口径内。
- 全项目稳定性指标仍缺少冻结的可执行协议，因此当前不能完成三维 Pareto 与 RQ2/RQ3；详见结果说明的限制章节及对应跟踪 Issue。
- 对 D 的 KernelSHAP 调试仍可复用本次固定的统一接口、真值 target、baseline、结果 schema、raw MoRF 和配置快照约定。
