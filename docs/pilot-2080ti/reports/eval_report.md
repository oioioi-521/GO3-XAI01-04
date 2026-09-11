# XAI01-04 试跑 P2：小规模端到端评价

生成时间（UTC）：`2026-09-11T13:48:04.533868+00:00`

VOC2007 与 CHNCXR 上的 ImageNet 预训练分类头均属域外，因此这些行是
`ood_pipeline_diagnostic`（域外管线诊断）的测量，不是任务忠实性结论或医学结论。

## 环境

- 设备：`cuda`；GPU：`NVIDIA GeForce RTX 2080 Ti`
- PyTorch / torchvision / CUDA runtime：`2.8.0` / `0.23.0` / `12.8`
- 每数据集图像数：`16`；稳定性 K：`2`；噪声 sigma：`0.01`
- 记录单元数：`576`

## 数据集 × 模型 × 方法汇总

| 数据集 | 模型 | 方法 | 成功/总计 | 归因耗时均值(s) | Insertion AUC | Deletion AUC | 稳定性 cosine | 归因前向样本数 | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| voc2007 | resnet50 | gradcam | 16/16 | 0.0137 | 0.1689 | 0.1436 | 0.9533 | 1.0000 | {'ok': 16} |
| voc2007 | resnet50 | kernelshap | 16/16 | 0.7779 | 0.2708 | 0.1154 | 0.9753 | 512.0000 | {'ok': 16} |
| voc2007 | resnet50 | ig | 16/16 | 0.0902 | 0.0598 | 0.0203 | 0.4886 | 32.0000 | {'ok': 16} |
| voc2007 | resnet50 | rise | 16/16 | 0.6281 | 0.2898 | 0.0854 | 0.9998 | 512.0000 | {'ok': 16} |
| voc2007 | resnet50 | ablation | 16/16 | 0.0705 | 0.2320 | 0.1565 | 0.8330 | 50.0000 | {'ok': 16} |
| voc2007 | resnet50 | lime | 16/16 | 0.7738 | 0.2985 | 0.0812 | 0.9482 | 512.0000 | {'ok': 16} |
| voc2007 | vgg16 | gradcam | 16/16 | 0.0079 | 0.1143 | 0.0680 | 0.9634 | 1.0000 | {'ok': 16} |
| voc2007 | vgg16 | kernelshap | 16/16 | 1.1404 | 0.2525 | 0.0518 | 0.9890 | 512.0000 | {'ok': 16} |
| voc2007 | vgg16 | ig | 16/16 | 0.1327 | 0.0320 | 0.0164 | 0.6271 | 32.0000 | {'ok': 16} |
| voc2007 | vgg16 | rise | 16/16 | 1.0316 | 0.2273 | 0.0335 | 0.9996 | 512.0000 | {'ok': 16} |
| voc2007 | vgg16 | ablation | 16/16 | 0.1082 | 0.2708 | 0.0541 | 0.9619 | 50.0000 | {'ok': 16} |
| voc2007 | vgg16 | lime | 16/16 | 1.1793 | 0.3380 | 0.0379 | 0.9848 | 512.0000 | {'ok': 16} |
| voc2007 | densenet121 | gradcam | 16/16 | 0.0190 | 0.1600 | 0.0867 | 0.9871 | 1.0000 | {'ok': 16} |
| voc2007 | densenet121 | kernelshap | 16/16 | 0.8260 | 0.1653 | 0.0307 | 0.9880 | 512.0000 | {'ok': 16} |
| voc2007 | densenet121 | ig | 16/16 | 0.1030 | 0.0766 | 0.0219 | 0.5902 | 32.0000 | {'ok': 16} |
| voc2007 | densenet121 | rise | 16/16 | 0.7116 | 0.2010 | 0.0282 | 0.9998 | 512.0000 | {'ok': 16} |
| voc2007 | densenet121 | ablation | 16/16 | 0.0846 | 0.1925 | 0.0567 | 0.9682 | 50.0000 | {'ok': 16} |
| voc2007 | densenet121 | lime | 16/16 | 0.8480 | 0.2829 | 0.0282 | 0.9925 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | resnet50 | gradcam | 16/16 | 0.0096 | 0.0242 | 0.0311 | 0.8844 | 1.0000 | {'ok': 16} |
| chncxr_shenzhen | resnet50 | kernelshap | 16/16 | 0.7098 | 0.0946 | 0.0167 | 0.8854 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | resnet50 | ig | 16/16 | 0.0875 | 0.0064 | 0.0085 | 0.3522 | 32.0000 | {'ok': 16} |
| chncxr_shenzhen | resnet50 | rise | 16/16 | 0.6057 | 0.1012 | 0.0116 | 0.9991 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | resnet50 | ablation | 16/16 | 0.0675 | 0.0370 | 0.0387 | 0.6685 | 50.0000 | {'ok': 16} |
| chncxr_shenzhen | resnet50 | lime | 16/16 | 0.7548 | 0.1292 | 0.0127 | 0.8817 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | vgg16 | gradcam | 16/16 | 0.0088 | 0.0397 | 0.0210 | 0.8664 | 1.0000 | {'ok': 16} |
| chncxr_shenzhen | vgg16 | kernelshap | 16/16 | 1.1062 | 0.0264 | 0.0144 | 0.9525 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | vgg16 | ig | 16/16 | 0.1262 | 0.0174 | 0.0107 | 0.4872 | 32.0000 | {'ok': 16} |
| chncxr_shenzhen | vgg16 | rise | 16/16 | 0.9882 | 0.0720 | 0.0214 | 0.9987 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | vgg16 | ablation | 16/16 | 0.1040 | 0.0598 | 0.0290 | 0.9012 | 50.0000 | {'ok': 16} |
| chncxr_shenzhen | vgg16 | lime | 16/16 | 1.1324 | 0.0482 | 0.0205 | 0.9501 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | densenet121 | gradcam | 16/16 | 0.0173 | 0.0535 | 0.0281 | 0.9037 | 1.0000 | {'ok': 16} |
| chncxr_shenzhen | densenet121 | kernelshap | 16/16 | 0.7867 | 0.0593 | 0.0126 | 0.9781 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | densenet121 | ig | 16/16 | 0.0976 | 0.0683 | 0.0356 | 0.5087 | 32.0000 | {'ok': 16} |
| chncxr_shenzhen | densenet121 | rise | 16/16 | 0.6812 | 0.0684 | 0.0218 | 0.9985 | 512.0000 | {'ok': 16} |
| chncxr_shenzhen | densenet121 | ablation | 16/16 | 0.0813 | 0.0556 | 0.0223 | 0.9347 | 50.0000 | {'ok': 16} |
| chncxr_shenzhen | densenet121 | lime | 16/16 | 0.8334 | 0.0915 | 0.0108 | 0.9528 | 512.0000 | {'ok': 16} |

## 协议

- 归因目标取模型在原图上的 ImageNet top-1 预测类别，逐单元记录。
- 忠实性使用缓存热图：20 步 / 21 个节点，梯形积分 AUC，替换基线为归一化黑图。
- 稳定性为 K=2 高斯噪声（原始 [0,1] 空间下的 sigma）重新归因，同一图像共享扰动；cosine 在带符号热图上计算。
- 随机方法在原图与扰动图上复用同一归因种子，使 cosine 只反映输入变化，而非 Monte-Carlo 噪声（P4 单独覆盖后者）。

## 复现命令

```bash
cd /home/hycx233/Courses/machine-learning/xai01-04-pilot-20260911
.venv/bin/python pilot_eval.py --evaluate --resume --max-images 16 --global-deadline-sec 2400 --task-timeout-sec 180
```

原始记录：`eval_records.jsonl`；聚合 JSON：`eval_summary.json`；CSV：`eval_per_image.csv`、`eval_units.csv`。
