# 成员 D：稳定性统计复核

初审日期：2026-09-23；修复与门禁复核：2026-09-30；远端状态复核：2026-10-01。

复核对象：PR #10 的 `9891eec`，以及 PR #14 的 Occlusion 门禁交接 `618e4fb`。

## 结论

成员 D 通过统计聚合、退化计分、修复源码静态复核和收到的门禁包重算，同意冻结以下
统计定义。`9891eec` 已回应 D/B 先前提出的引用图身份、debug/eval 隔离和基础 runner
resume/force 身份保持问题；本结论仍不代替 C 对 runner/schema 和集成顺序的最终确认。

- 每张图固定 5 次重复；每次计算原始 `A(x)` 与扰动 `A(x+δ)` 的全像素 Spearman。
- 有限且非退化的相关系数直接作为该次 `aggregate_score`；非有限、常量或相关系数未定义时，
  trace 保留明确状态，主分数按 0 计入，不把退化重复从均值分母中删除。
- 每图 `stability_spearman` 是 5 个 `aggregate_score` 的算术均值；
  `stability_valid_rate = valid_repeats / 5`，两者分开报告。
- 单元均值、标准差和样本数基于逐图值汇总，避免重复级样本伪增；跨方法和模型复用
  `dataset/image_id/repeat_idx` 派生的相同扰动，以支持配对比较。
- top-10% Jaccard 只作为诊断，不进入主排名；稳定性补跑不重复计算 MoRF，
  也不计入 `efficiency_time_ms`。

## 复核证据

- 候选协议 40 张门禁：1,200 个唯一输入扰动的预测保持率为 98.17%。
- IG 为 1,200/1,200 有效，Spearman 均值 0.701；Grad-CAM 为 1,185/1,200 有效，
  Spearman 均值 0.974。
- 15 个退化重复来自 3 张原始常量 Grad-CAM 图，每张 5 次；按 0 计分且另报有效率，
  能避免删除困难样本后人为抬高稳定性。
- 修复提交 `9891eec` 为 reference map 增加基础配置、预测上下文、checkpoint 和 NPY
  哈希绑定；旧图必须用当前模型/配置重算并在容差内一致后才能写
  `verified_recomputation` sidecar。状态和 trace 同时纳入 split、输出路径及配置哈希，
  基础 runner 在已有稳定性行时对失效重跑采取 fail-closed。
- 接收的轻量门禁包为 373,591 bytes，SHA-256
  `d35a6d47532dae15c063d94a6bb5e22708652076bd2e14b2b9f2eca2d7994174`；
  包内 manifest 的 407 个文件全部通过大小和 SHA-256 校验。
- D 重算了 6 个单图和 6 个 40 张 debug 单元：合计 246 个单元内图片、1,230 条
  trace、492 条稳定性逐图指标；所有 trace seed 均符合
  `dataset/image_id/repeat` 规则，同图扰动跨三模型一致，全部状态为 `valid`，各单元
  立即续跑证据均为 0 processed。

## Occlusion 40 张门禁结果

这些数值只证明候选协议在 Occlusion 上可执行，不是 460 张正式结果或跨方法排名。

| 单元 | stability_spearman | valid_rate | 扰动归因 ms |
|---|---:|---:|---:|
| ResNet50 / ImageNet | 0.918921 | 1.000 | 126.369 |
| ResNet50 / VOC | 0.906132 | 1.000 | 126.690 |
| DenseNet121 / ImageNet | 0.986859 | 1.000 | 155.336 |
| DenseNet121 / VOC | 0.988125 | 1.000 | 151.745 |
| VGG16 / ImageNet | 0.989010 | 1.000 | 260.569 |
| VGG16 / VOC | 0.991460 | 1.000 | 260.065 |

轻量包不含参考 NPY；D 已核对 sidecar 的 map SHA-256 与输入清单、基础配置、checkpoint
和 `verified_recomputation` 绑定，但这不等于独立读取了门禁副本中的 NPY 内容。

## 状态边界

该复核确认 D 负责的统计定义、聚合和退化计分，并接受 Occlusion 单图/40 张候选门禁。
PR #10 已于 2026-09-30 合并（merge commit `00d73b4`）；PR #14 仍为 draft，Issue #8
仍为 open。C 的 runner/schema 最终确认和六个 460 张 stability eval 完成前，全组协议
不能由 D 单方面标记为 frozen，门禁数值也不能作为正式课程实验排名。
