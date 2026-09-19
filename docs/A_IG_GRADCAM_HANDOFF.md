# 成员 A 交接说明：Integrated Gradients + Grad-CAM

更新时间：2026-09-19。

## 当前完成状态

- 已实现 Integrated Gradients（结果中的方法名为 `ig`）。
- 已实现 Grad-CAM（结果中的方法名为 `gradcam`）。
- 已将两种方法接入 `run_unit.py` 的统一归因接口、断点续跑、配置快照、预测表、逐图 CSV、单元汇总和日志流程。
- 已提供 **2 方法 × 3 模型 × 2 数据集 = 12** 个正式单元 YAML。
- 已提供算法、非法参数、目标类别、目标层解析、runner 执行/续跑和 12 配置矩阵测试。
- **尚未正式跑批**：等待台式机 GPU 访问权限、冻结评估图片和 VOC-20 checkpoint。

本地验证环境为 Python 3.12.14、PyTorch 2.14.0+cpu、torchvision 0.29.0+cpu、Captum 0.9.0。执行 `python -m pytest -q -ra` 的结果为 **29 passed**，`uv pip check` 无依赖冲突；三种 torchvision 模型的目标层路径也已用随机初始化结构逐一解析。该验证只证明代码和合成链路，不是课程实验结果。

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

## 正式开跑前置条件

1. 台式机可访问，CUDA/PyTorch/Captum 环境可用。
2. `preprocessing/verify_data_version.py` 通过，且 metadata 指向的 460 张 ImageNet eval 与 460 张 VOC eval 图片都存在；当前笔记本仓库只有 metadata，没有这些原图。
3. ImageNet 权重可下载或已经缓存。
4. B 提供并冻结下列 VOC-20 checkpoint，类别顺序必须与 `VOC_CLASSES` 一致：
   - `models/checkpoints/vgg16_voc20.pt`
   - `models/checkpoints/resnet50_voc20.pt`
   - `models/checkpoints/densenet121_voc20.pt`
5. 先分别检查三种模型的一张 IG/Grad-CAM 热力图、target/prediction 和 MoRF 数值，再启动全量。

## 待台式机授权后的执行顺序

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

- 12 单元的正式日志、热力图、逐图 CSV 和单元 CSV：等待台式机权限后生成。
- VOC 六个单元：额外等待三个 20 类 checkpoint。
- 对 D 的 KernelSHAP 调试：截至本说明更新时仓库没有 D 的实现分支；其实现应复用本次固定的统一接口、真值 target、baseline、结果 schema 和配置快照约定。
