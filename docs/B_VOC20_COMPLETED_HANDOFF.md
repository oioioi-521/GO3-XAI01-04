# 成员 B：VOC20 正式 checkpoint 交接

对应 PR #7，训练代码提交为 `f791d1e8a346b66d21c807caabeb1a293cce84b3`。
机器可读的文件名、完整 SHA-256、配置与数据哈希见
[`VOC20_CHECKPOINT_MANIFEST.json`](VOC20_CHECKPOINT_MANIFEST.json)。权重二进制均在
`models/checkpoints/`，该目录被 Git 忽略，**不得**加入 Git。

## 协议与数据隔离

- 任务是 VOC20 多标签分类：20 维 multi-hot target、`BCEWithLogitsLoss`、sigmoid 推理；
  不使用 ImageNet softmax/top-1 语义。
- 训练候选来自 VOC2007 trainval；冻结的 500 张 debug/eval 图在建立 manifest 前排除。
  最终 train/validation 是 4,037/474 张，分别由
  `data/voc20_multilabel_train.csv`、`data/voc20_multilabel_validation.csv` 指定。
- 冻结排除清单的 SHA-256 是
  `e1fd56c2101a1f1240dca631cf5ace0de7e5647cf83d72be99e314fe52fcbdfa`。
  冻结图像未参与训练、验证、早停、阈值选择或调参。最终仅用于每个模型一次的加载/前向工程核验。

## 环境和训练配置

- Windows，`.venv-gpu`，PyTorch `2.11.0+cu128`、torchvision `0.26.0+cu128`、CUDA 12.8；
  RTX 5080 Laptop GPU。
- 三个模型均以 torchvision 官方 `DEFAULT` ImageNet 权重初始化，再替换为 20 维 head：
  VGG16、ResNet50、DenseNet121。
- 优化器为 SGD（learning rate `1e-4`、momentum `0.9`、weight decay `1e-4`）；
  `BCEWithLogitsLoss` 的 `pos_weight` 从训练 manifest 计算并上限截断为 20；CUDA AMP 开启。
- batch size：VGG16=16、ResNet50=32、DenseNet121=24。F1 sigmoid threshold 固定为 0.5。
- 早停监控 validation mAP，`patience=6`、`min_delta=0.0`。每个成功 epoch 用临时文件和
  `os.replace` 原子更新独立的 `.last.pt`；恢复会校验模型、类别顺序、manifest hash 和关键配置。

## 正式结果

| 模型 | 正式文件 | 最佳 epoch | mAP | macro-F1 | micro-F1 |
|---|---|---:|---:|---:|---:|
| ResNet50 | `models/checkpoints/resnet50_voc20.pt` | 30 | 0.7727 | 0.6033 | 0.6185 |
| DenseNet121 | `models/checkpoints/densenet121_voc20.pt` | 29 | 0.7883 | 0.6205 | 0.6373 |
| VGG16 | `models/checkpoints/vgg16_voc20.pt` | 24 | 0.8182 | 0.7378 | 0.7551 |

三个正式文件均已用 `models.factory.load_model(..., num_classes=20,
output_activation="sigmoid", checkpoint=...)` 的 `strict=True` 加载，并在一张冻结 debug 图像上得到
有限的 `(1, 20)` logits 和 sigmoid 概率。旧的
`resnet50_voc20_multilabel.pt` 和 `smoke/` 目录仅是工程 smoke 产物，不是正式交付。

## 时间与恢复说明

- DenseNet121 的单进程完整训练记录为 787.87 秒；VGG16 为 831.42 秒。
- ResNet50 历史记录的 `elapsed_seconds=202.94` **仅是最后一次恢复进程耗时，不是累计总训练耗时**。
  不得将该值用作完整训练时间，也不得从 epoch、时间戳或日志反推总时长。
- 本机训练过程中曾遇到一次 CUDA `unknown error` / Windows TDR 中断；从上一个完整的原子
  `last.pt` 成功恢复并完成训练。尚未确定 TDR 根因，未修改驱动或 Windows TDR 注册表。
- 后续训练器版本将 `elapsed_seconds_total` 写入 rolling checkpoint，并在记录中另列当前
  `elapsed_seconds_process`；它只适用于未来训练，不能追溯修复本次 ResNet50 的总时长。

## 给 A 的使用边界

可将正式 checkpoint 用于后续“单图 → 40 张 debug → 460 张 eval”验收；结果需继续遵守 VOC
multi-label sigmoid 契约，选择 MoRF target 时应显式记录 target 类别。成员 B 不会把权重上传到
Git、GitHub Release 或第三方网盘。若 A 无法读取本机
`models/checkpoints/`，当前阻塞项是**缺少已批准的二进制传输渠道**。
