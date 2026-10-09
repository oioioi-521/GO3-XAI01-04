# RISE 正式评测结果

更新时间：2026-10-09。

## 当前进度

| 模型 | 数据集 | 状态 | 配置哈希 |
| --- | --- | --- | --- |
| ResNet50 | ImageNet | 已完成 460/460，并通过立即续跑 | `c035d8b2d1bd` |
| DenseNet121 | ImageNet | 已完成 460/460，并通过续跑 | `b20cd6be154f` |
| VGG16 | ImageNet | 已完成 460/460，并通过立即续跑 | `4412824305f1` |
| ResNet50 | VOC | checkpoint 验收通过，正式批次运行中 | `c52269f4cda3` |
| DenseNet121 | VOC | checkpoint 验收通过，等待顺序跑批 | — |
| VGG16 | VOC | checkpoint 验收通过，等待顺序跑批 | — |

稳定性尚未补跑；PR #10 已合入 `integrate/a-voc-eval`，Issue #8 已记录协议冻结。
C 的独立稳定性集成分支 `feat/rise-stability-eval` 已通过 72 项测试。
VOC checkpoint 接收证据见 [`C_VOC20_RECEIPT.md`](C_VOC20_RECEIPT.md)。
当前结果不能用于三维 Pareto 或完整排名。

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

## DenseNet121 / ImageNet

运行环境与正式契约同 ResNet50。首轮日志为
`processed=460, skipped=0, warmup_runs=1`，完成时间
`2026-09-30T15:09:33+0800`；控制台墙钟约 87 分 24 秒。续跑日志为
`processed=0, skipped=460, warmup_runs=1`，完成时间
`2026-09-30T17:40:10+0800`。

| 指标 | mean | std | n | 方向 |
| --- | ---: | ---: | ---: | --- |
| `efficiency_time_ms` | 10836.084072 | 203.399996 | 460 | 越低越好 |
| `faithfulness_morf_auc_raw`（全部样本，敏感性分析） | 0.054237 | 0.063685 | 460 | 越低越好 |
| `faithfulness_morf_auc_raw`（top-1 正确，主分析） | 0.063524 | 0.066043 | 371 | 越低越好 |
| `faithfulness_morf_auc_raw`（top-1 错误，诊断） | 0.015524 | 0.030592 | 89 | 不进入主排名 |

正确/错误分组口径与 ResNet50 相同。

## VGG16 / ImageNet

运行环境与正式契约同 ResNet50。首轮日志为
`processed=460, skipped=0, warmup_runs=1`，完成时间
`2026-09-30T20:35:46+0800`；控制台墙钟约 2 小时 53 分 46 秒。立即续跑日志为
`processed=0, skipped=460, warmup_runs=1`，完成时间
`2026-09-30T20:36:15+0800`。

| 指标 | mean | std | n | 方向 |
| --- | ---: | ---: | ---: | --- |
| `efficiency_time_ms` | 22435.320230 | 821.290054 | 460 | 越低越好 |
| `faithfulness_morf_auc_raw`（全部样本，敏感性分析） | 0.052469 | 0.063082 | 460 | 越低越好 |
| `faithfulness_morf_auc_raw`（top-1 正确，主分析） | 0.063336 | 0.066025 | 363 | 越低越好 |
| `faithfulness_morf_auc_raw`（top-1 错误，诊断） | 0.011800 | 0.021679 | 97 | 不进入主排名 |

正确/错误分组口径与 ResNet50 相同。

## 完整性与哈希

- 460 个唯一 eval `image_id` 与冻结 metadata 完全一致；逐图指标 920 行，每图恰有
  `efficiency_time_ms` 和 `faithfulness_morf_auc_raw` 两行；预测表 eval 行为 460 条。
- 460 张 float32 NPY 均为有限的 `224×224`、值域 `[0,1]`，合计 92,382,720
  字节。聚合 SHA-256 为
  `81e7719d12f75d7705b770ee572ec405ec06ae9c290e12894eddca1723b2718a`。
- 对应 460 张 eval PNG 合计 4,981,884 字节，聚合 SHA-256 为
  `723d0577e690d5e8b225ac645f013ad04baf5615ad96f5156cea0ddb925d2bd1`。
- 配置快照 SHA-256：
  `e00bc1b564b3d24f236f308f9887d72f76b71564615a2b242362e1a0ed621d16`。
- 续跑状态 SHA-256：
  `fb987c74d95be92429c7a17e0795e8893289f4bd231e7a88c6124017070fc26a`。

DenseNet121/ImageNet 同样通过严格完整性检查：460 个唯一 eval `image_id`、920 行
逐图指标、460 条 eval 预测，以及 460 张有限的 `224×224` float32 NPY 和对应
PNG。NPY 合计 92,382,720 字节，聚合 SHA-256 为
`3e4bf4c52ab6b452f4373f3f8b5f9a435719a23a305a47d784d064a6db5730c0`；PNG 合计
4,661,223 字节，聚合 SHA-256 为
`f52547f28970d3a417d8635eec3ea14c24b1db10a93485c8c43fe9a047c8def9`。配置快照
SHA-256 为 `7714ac8cbd03c22a39a5f6df67501b81b0899e15331e97c55ed69dd833e0f4d7`，
续跑状态 SHA-256 为
`b7e2eb7f38ec8fa4bda8bb8c0de868021cba347379af87257d7e2e9e21a505cc`。

VGG16/ImageNet 也通过相同严格检查。460 张 NPY 合计 92,382,720 字节，聚合
SHA-256 为 `bfbe6047942d735823c8c382352e592aa18c9e1fa4866bd9d0fcaaf4bdcd5dec`；
460 张 PNG 合计 4,631,178 字节，聚合 SHA-256 为
`641a99f9e5cb50a80d8ec56d0a6d90d50de2b5bc9334b0f3281e18ba0cf3864f`。配置快照
SHA-256 为 `f3a7911ef558d14047e797336e0dee104ebd0574adca61cd590fc92b2d3c927e`，
续跑状态 SHA-256 为
`ba66cf786c3f2368ad3775bb54c1990bc4bda8fde77bab94422b1d30f706ed45`。

为使聚合身份可在任意平台精确复算，本文三组 NPY/PNG 聚合哈希统一定义如下：先按冻结
eval ID 过滤并按文件名排序，为每个文件生成
`{"path": 文件名, "bytes": 字节数, "sha256": 文件 SHA-256}`，再对
`json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))` 的
UTF-8 字节计算 SHA-256。此前未明确序列化细节的临时聚合值不再作为正式身份。

共享 `predictions.csv` 和 PNG 目录仍保留 2026-09-16 的历史 debug 记录；本次验收严格
按 `split=eval` 及冻结 eval ID 集合筛选，不把历史 debug 条目计入上述数量或哈希。
大体积结果继续由 Git 忽略并保存在本机，本文只提交可审计摘要和聚合身份。
