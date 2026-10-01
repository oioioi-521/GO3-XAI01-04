# 成员 D：Occlusion 正式评测交接验收

验收日期：2026-09-27；float32 补充验收：2026-09-30；远端状态复核：2026-10-01

实验代码提交：`6d5645f`

交接文档提交：`5672ccf`

## 验收结论

`B_OCCLUSION_EVAL_LIGHT.zip` 可作为六个正式 eval 单元的 faithfulness 和同环境
efficiency 统计输入。后续收到的 `B_OCCLUSION_EVAL_FLOAT32.zip` 已通过详细清单和
数组内容验收，可作为对应 2,760 张原始归因图的正式输入。本次验收仍不覆盖 460 张
stability eval，也不直接构成跨方法最终排名。

## 完整性与可复现性

- 交接包：387,844 bytes；SHA-256
  `996e59e41d0262ef19fd0292a71abc139cbb9bb92c4868451b65938bcd6b5cea`。
- 详细清单：800,037 bytes；SHA-256
  `98541858944b364e2b681aaed7bef549597248d9f9affc23c13aa94310e29845`。
- float32 包：6,260,426 bytes；SHA-256
  `7f677af2c0fe7b4d993acdd4960cb4e693ae0eb506ec913c5ef7ef25a8089fb8`。
- 详细清单中的 2,814 个文件已逐项校验，无缺失、无大小或 SHA-256 不一致；其中
  2,760 张 NPY 均为 `float32`、`224×224`、有限、非恒定且位于 `[0,1]`。
- 六个模型/数据集单元均为 460 张 eval 图，合计 2,760 条预测和 5,520 条逐图指标；
  `failed=0`、`skipped=0`。
- D 的合并门禁重新计算了每单元 `mean`、`std`、`n`，并核对配置、预测上下文、
  checkpoint 哈希和样本集合。
- 全量测试为 79 passed；`pip check` 未发现依赖冲突。

## 预测命中与分析样本

这里的 `top1_correct` 表示冻结目标类别是否为模型最高分输出。VOC 使用的是冻结单目标
口径，不是 20 类多标签 exact-match accuracy。

| 模型 | ImageNet | VOC |
|---|---:|---:|
| ResNet50 | 408/460 (88.70%) | 387/460 (84.13%) |
| DenseNet121 | 371/460 (80.65%) | 388/460 (84.35%) |
| VGG16 | 363/460 (78.91%) | 400/460 (86.96%) |

三模型都命中目标类别的共同样本为：ImageNet 332/460，VOC 353/460。跨模型配对比较
应使用这两个共同样本集。

## 正确/错误分组诊断

`faithfulness_morf_auc_raw` 越低越好。错误组的原始 AUC 在六个单元中都更低，这会让低
目标置信度或误分类样本呈现“解释更好”的假象。

| 模型/数据集 | 正确组 n / raw mean | 错误组 n / raw mean |
|---|---:|---:|
| ResNet50 / ImageNet | 408 / 0.215 | 52 / 0.019 |
| DenseNet121 / ImageNet | 371 / 0.097 | 89 / 0.018 |
| VGG16 / ImageNet | 363 / 0.048 | 97 / 0.009 |
| ResNet50 / VOC | 387 / 0.649 | 73 / 0.409 |
| DenseNet121 / VOC | 388 / 0.485 | 72 / 0.284 |
| VGG16 / VOC | 400 / 0.280 | 60 / 0.106 |

因此冻结以下统计口径：

1. 单模型 faithfulness 主分析仅使用该模型的 `top1_correct` 样本。
2. 全部 460 张结果作为敏感性分析单独报告，不能替代主分析。
3. 跨模型配对比较使用 `common_correct_cohort.csv` 中三模型共同命中的样本。
4. ImageNet 与 VOC 分开分析；VOC 口径不得改写成多标签预测正确率。
5. efficiency 只在同一硬件、同一计时协议内比较。

## 未验收项

- PR #10 已于 2026-09-30 合并，merge commit 为 `00d73b4`；其中 `9891eec` 已回应
  引用图身份、debug/eval 隔离和 resume/force 三项阻塞。D 已完成统计复核，但
  Issue #8 仍为 open，C 的 runner/schema 最终确认尚未公开记录，因此全组协议状态
  仍不能由 D 单方面改写为 frozen。
- 当前收到的是单图和 40 张 Occlusion stability 工程门禁，不是六个 460 张正式
  stability eval；不得将门禁均值写成正式方法排名。
- 本结论只接受 Occlusion 的现有正式 eval 结果，不代表 KernelSHAP 或其他方法已经完成。
