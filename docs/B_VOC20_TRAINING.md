# 成员 B：VOC20 多标签训练与恢复

VOC2007 trainval 的训练和验证只读取 `data/voc20_multilabel_train.csv` 与
`data/voc20_multilabel_validation.csv`。两份 manifest 在生成前已经排除了
冻结的 500 张 debug/eval 图像；冻结图像不得用于训练、验证、早停、阈值选择或调参。

## 训练和恢复

在项目根目录使用 `.venv-gpu`，每次只启动一个模型：

```powershell
.\.venv-gpu\Scripts\python.exe training\train_voc20.py --config configs\train_voc20_resnet50.yaml
```

每个**成功完成**的 epoch 后，训练器以临时文件加 `os.replace` 原子更新
`last.pt`。中断后以同一配置继续：

```powershell
.\.venv-gpu\Scripts\python.exe training\train_voc20.py `
  --config configs\train_voc20_resnet50.yaml `
  --resume models\checkpoints\resume\resnet50_voc20.last.pt
```

恢复会校验模型名、VOC20 类别顺序、训练 manifest SHA-256、batch size、学习率、早停参数及
sigmoid 阈值。身份不匹配时会拒绝恢复。滚动恢复文件保存模型、优化器、调度器、AMP scaler、
当前 epoch、最佳 mAP/epoch、patience 计数和随机状态；它不是正式模型。

早停监控 validation mAP，默认 `patience: 6`、`min_delta: 0.0`。改善时才替换正式最佳
checkpoint；恢复后继续使用已保存的计数，不会覆盖历史最佳模型。

## 验证口径

VOC 是 20 维 multi-hot 标签，损失为 `BCEWithLogitsLoss`，推理/验证使用 sigmoid。
配置中的 `metrics.threshold` 显式指定 F1 阈值，默认 0.5。每轮验证报告：

- mAP 和每类 AP；
- macro-F1；
- micro-F1；
- 使用的 sigmoid threshold。

这些指标对零分母场景返回有限的零值，不能将任何 NaN/Inf 写入训练记录。

## 产物边界

Issue #4 约定的正式最佳权重（均被 Git 忽略）为：

- `models/checkpoints/resnet50_voc20.pt`
- `models/checkpoints/densenet121_voc20.pt`
- `models/checkpoints/vgg16_voc20.pt`

其滚动恢复文件在 `models/checkpoints/resume/`，名称以 `.last.pt` 结尾。工程恢复烟雾测试
使用 `configs/train_voc20_resnet50_resume_smoke.yaml`，所有输出位于
`models/checkpoints/smoke/` 与 `results/training/smoke/`，不能作为正式模型或指标。
旧的 `resnet50_voc20_multilabel.pt` 同样仅是早期一 epoch smoke 产物，必须保留但不得交付为正式
checkpoint。

正式 checkpoint 是带元数据的包装，包含 `state_dict`、`VOC_CLASSES`、`task_type`、
`output_activation`、manifest hash、最佳 epoch 和验证指标；`models.factory.load_model(...,
checkpoint=..., num_classes=20, output_activation="sigmoid")` 用 `strict=True` 读取它。
