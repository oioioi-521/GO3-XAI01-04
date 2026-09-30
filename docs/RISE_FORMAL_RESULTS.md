# RISE 正式评测结果

更新时间：2026-09-30。

## 当前进度

| 模型 | 数据集 | 状态 | 配置哈希 |
| --- | --- | --- | --- |
| ResNet50 | ImageNet | 已完成 460/460，并通过立即续跑 | `c035d8b2d1bd` |
| DenseNet121 | ImageNet | 待跑 | — |
| VGG16 | ImageNet | 待跑 | — |
| ResNet50 | VOC | 待本机接收并验收 checkpoint | — |
| DenseNet121 | VOC | 待本机接收并验收 checkpoint | — |
| VGG16 | VOC | 待本机接收并验收 checkpoint | — |

稳定性尚未补跑；Issue #8 和 PR #10 的候选协议通过 C 侧代码、测试与 RISE
接口复核后，仍需完成组内冻结/合并流程。当前结果不能用于三维 Pareto 或完整排名。

## ResNet50 / ImageNet

运行环境为 Python 3.12.14、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、
RTX 4060 Laptop GPU 8GB。配置使用 4000 masks、`batch_size=32`、21 点 raw
MoRF、一次不计时 warmup，同时保存 PNG 和 float32 NPY。

首轮日志为 `processed=460, skipped=0, warmup_runs=1`，完成时间
`2026-09-30T13:34:24+0800`；控制台墙钟约 87 分 59 秒。立即续跑日志为
`processed=0, skipped=460, warmup_runs=1`，完成时间
`2026-09-30T13:35:48+0800`。

| 指标 | mean | std | n | 方向 |
| --- | ---: | ---: | ---: | --- |
| `efficiency_time_ms` | 11235.028828 | 473.701526 | 460 | 越低越好 |
| `faithfulness_morf_auc_raw`（全部样本，敏感性分析） | 0.107078 | 0.108800 | 460 | 越低越好 |
| `faithfulness_morf_auc_raw`（top-1 正确，主分析） | 0.118306 | 0.110132 | 408 | 越低越好 |
| `faithfulness_morf_auc_raw`（top-1 错误，诊断） | 0.018982 | 0.028657 | 52 | 不进入主排名 |

正确/错误分组遵循 D 已冻结的 Occlusion 验收口径：单模型 faithfulness 主分析只使用
`top1_correct` 样本，全部 460 张仅作敏感性分析。错误组的低 raw AUC 主要受目标分数
较低影响，不能解释为归因更忠实。

## 完整性与哈希

- 460 个唯一 eval `image_id` 与冻结 metadata 完全一致；逐图指标 920 行，每图恰有
  `efficiency_time_ms` 和 `faithfulness_morf_auc_raw` 两行；预测表 eval 行为 460 条。
- 460 张 float32 NPY 均为有限的 `224×224`、值域 `[0,1]`，合计 92,382,720
  字节。按“相对路径、字节数、文件 SHA-256”排序串联后的聚合 SHA-256 为
  `b04f110827fac654eecf693acaec394c8c4b47b53c0acc8910413a052f4ff0ee`。
- 对应 460 张 eval PNG 合计 4,981,884 字节，按相同规则计算的聚合 SHA-256 为
  `e0833bdc6f528691ddda76f740647a83ea6a7ba438cd000b00a9e388755bc5a1`。
- 配置快照 SHA-256：
  `e00bc1b564b3d24f236f308f9887d72f76b71564615a2b242362e1a0ed621d16`。
- 续跑状态 SHA-256：
  `fb987c74d95be92429c7a17e0795e8893289f4bd231e7a88c6124017070fc26a`。

共享 `predictions.csv` 和 PNG 目录仍保留 2026-09-16 的历史 debug 记录；本次验收严格
按 `split=eval` 及冻结 eval ID 集合筛选，不把历史 debug 条目计入上述数量或哈希。
大体积结果继续由 Git 忽略并保存在本机，本文只提交可审计摘要和聚合身份。
