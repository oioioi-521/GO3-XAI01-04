# RISE 采样预算门禁结果

更新时间：2026-09-23。环境为 Python 3.12.14、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、RTX 4060 Laptop GPU 8GB。模型为 ImageNet ResNet50 默认权重，target 为冻结 metadata 真值类别，`batch_size=32`。每个预算先执行一次不计时 warmup。

原始记录：

- `analysis/pilots/rise_budget_smoke.csv`：冻结 debug 集首图；
- `analysis/pilots/rise_budget_40.csv`：冻结 40 张 debug 集，共 120 行。

## 单图 smoke

| masks | RISE 耗时 | 相对 4000 masks Pearson | 相对 4000 masks MAE |
| ---: | ---: | ---: | ---: |
| 1000 | 3.263 s | 0.893 | 0.113 |
| 2000 | 5.464 s | 0.966 | 0.045 |
| 4000 | 11.028 s | 1.000 | 0.000 |

GPU 显存占用约 2.0 GiB，`batch_size=32` 在 8GB 显卡上有充分余量。

## 冻结 40 张门禁

| masks | n | 耗时均值 | 耗时范围 | Pearson 均值 | Pearson 最低 | MAE 均值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 40 | 2.720 s | 2.679–2.782 s | 0.877 | 0.630 | 0.0828 |
| 2000 | 40 | 5.440 s | 5.366–5.562 s | 0.942 | 0.745 | 0.0539 |
| 4000 | 40 | 10.889 s | 10.763–11.075 s | 1.000 | 1.000 | 0.0000 |

Pearson 与 MAE 均以同图 4000-mask 结果为参考。三个预算使用相同 seed；较小预算的 mask 流是较大预算的前缀，因此差异可配对解释。

## 决策与成本

冻结正式预算为 **4000 masks**。原因：2000 masks 虽然平均相关已达到 0.942，但最差样本仅 0.745，仍有明显 Monte-Carlo 波动；4000 是原交接建议的最低正式预算，当前 GPU 又不存在显存瓶颈。

按 ResNet50-ImageNet 的 40 张均时线性外推，单个 460 张单元的纯 RISE attribution 约 83.5 分钟。该数字不包含模型加载、21 点 raw MoRF、PNG/NPY 落盘和稳定性补跑；VGG16、DenseNet121 与 VOC checkpoint 必须分别以真实日志报告，不用本外推冒充实测。

正式跑批继续使用 6 份 YAML 中统一的 `num_masks: 4000`。若更换 GPU，必须重新做单图成本 smoke；除非重新提交 40 张收敛证据，不得自行降低预算。
