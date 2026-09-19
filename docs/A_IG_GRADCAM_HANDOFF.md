# 成员 A 交接说明：Integrated Gradients + Grad-CAM

更新时间：2026-09-19。

## 当前完成状态

- 已实现 Integrated Gradients（结果中的方法名为 `ig`）。
- 已实现 Grad-CAM（结果中的方法名为 `gradcam`）。
- 已将两种方法接入 `run_unit.py` 的统一归因接口、断点续跑、配置快照、预测表、逐图 CSV、单元汇总和日志流程。
- 已提供 **2 方法 × 3 模型 × 2 数据集 = 12** 个正式单元 YAML。
- 已提供算法、非法参数、目标类别、目标层解析、runner 执行/续跑和 12 配置矩阵测试。
- **ImageNet 六个正式单元已完成**：每单元 460 张，共生成 5,520 条逐图指标、1,380 条预测和 2,760 张热力图。
- **VOC 六个正式单元尚未运行**：仍等待三种模型的 VOC-20 checkpoint。

本地验证环境为 Python 3.12.14、PyTorch 2.14.0+cpu、torchvision 0.29.0+cpu、Captum 0.9.0。执行 `python -m pytest -q -ra` 的结果为 **29 passed**，`uv pip check` 无依赖冲突；三种 torchvision 模型的目标层路径也已用随机初始化结构逐一解析。该验证只证明代码和合成链路，不是课程实验结果。

2026-09-19 在台式机 RTX 2080 Ti 上使用 Python 3.13.9、PyTorch 2.8.0、torchvision 0.23.0 和 Captum 0.9.0 复验，结果为 **31 passed**，`pip check` 无依赖冲突。正式结果保存在台式机工作树 `/home/hycx233/Courses/machine-learning/GO3-XAI01-04-a-ig-gradcam/results/`；该目录由 Git 忽略，不随代码提交。

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

## ImageNet 正式跑批结果

六个 ImageNet 单元均使用冻结的 `eval` 划分和 YAML 原始配置运行；日志均为 `processed=460, skipped=0`，逐图指标无重复、NaN 或 Inf。汇总如下（均为 460 张的均值 ± 样本标准差）：

| 方法 | 模型 | 归因耗时（ms） | MoRF AUC（越低越好） |
|---|---|---:|---:|
| IG | VGG16 | 256.284 ± 22.046 | 0.172238 ± 0.786286 |
| IG | ResNet50 | 145.028 ± 24.059 | 0.167405 ± 0.092978 |
| IG | DenseNet121 | 166.459 ± 26.919 | 0.174114 ± 0.374724 |
| Grad-CAM | VGG16 | 9.680 ± 20.214 | 0.878437 ± 6.389760 |
| Grad-CAM | ResNet50 | 9.181 ± 22.976 | 0.260079 ± 0.244595 |
| Grad-CAM | DenseNet121 | 20.484 ± 22.189 | 0.224747 ± 1.378958 |

预测 Top-1 准确率分别为 VGG16 363/460（78.91%）、ResNet50 409/460（88.91%）、DenseNet121 371/460（80.65%）。当前 MoRF 实现以真值类别的原始概率归一化；当模型误分类且真值概率很小时，扰动后的比值可能大于 1，因此少量大值会显著抬高均值和标准差，这不是非有限数值。VGG16 Grad-CAM 有 2/460 张全零图，分别是一张误分类图和一张高置信正确分类图；这是 ReLU 后无正向 CAM 的合法输出，结果按原始算法保留，未人工替换。

## 正式跑批状态与剩余条件

1. 台式机访问和 CUDA/PyTorch/Captum 环境：已通过。
2. `preprocessing/verify_data_version.py` 和冻结的 460 张 ImageNet eval 图片：已通过；VOC 图片也已就位。
3. 三种 ImageNet 官方权重：已缓存并校验。
4. B 提供并冻结下列 VOC-20 checkpoint，类别顺序必须与 `VOC_CLASSES` 一致：
   - `models/checkpoints/vgg16_voc20.pt`
   - `models/checkpoints/resnet50_voc20.pt`
   - `models/checkpoints/densenet121_voc20.pt`
5. 三种模型的一张 IG/Grad-CAM 热力图、target/prediction 和 MoRF 数值检查：已通过；另完成每单元 40 张 debug 批次检查。

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
```

runner 会在配置哈希变化时清理同一方法/模型/数据集的旧单元行，所以首图检查不会被错误复用为全量结果。正式跑批支持逐图续跑，不需要为中断手工删除完整结果。

## 尚未完成及外部依赖

- ImageNet 六个单元已完成；台式机上的 `results/` 包含正式日志、配置快照、热力图、逐图 CSV、预测 CSV 和单元汇总 CSV。
- VOC 六个单元：等待三个 20 类 checkpoint；在 checkpoint 到位并校验类别顺序前不要启动。
- 对 D 的 KernelSHAP 调试：截至本说明更新时仓库没有 D 的实现分支；其实现应复用本次固定的统一接口、真值 target、baseline、结果 schema 和配置快照约定。
