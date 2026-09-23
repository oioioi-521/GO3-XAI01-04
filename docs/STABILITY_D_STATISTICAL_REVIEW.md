# 成员 D：稳定性统计复核

复核日期：2026-09-23。复核对象：PR #10，提交 `ab60d54`。

## 结论

成员 D 通过统计聚合与退化计分复核，同意冻结以下定义；本结论不代替 C 对 runner、
CSV upsert、resume 和扰动位置的复核。

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
- D 在 PR 头提交 `ab60d54` 上独立运行完整测试，结果为 64 passed；仅有既有
  `torch.load` FutureWarning，无测试失败。

## 状态边界

该复核只确认统计定义与实现语义。PR #10 尚未合并，协议在 C 的 runner/schema 复核完成前
仍不能标记为 frozen，pilot 数值也不能作为正式课程实验结果。
