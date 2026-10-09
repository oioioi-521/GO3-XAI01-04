# 成员 C：稳定性 runner/schema 最终复核

复核日期：2026-09-30。复核对象：PR #10 头提交 `9891eec`。

## 结论

runner/schema 复核通过，未发现阻塞合并的问题。该结论覆盖候选稳定性协议的运行入口、
结果身份、断点续跑和 RISE 接口兼容性；不把 1 张或 40 张工程门禁误写为 460 张正式
stability 结果，也不替代仓库维护者的最终合并操作。

## 验证范围

- 在 PR #10 原始提交上执行完整测试：`68 passed`。
- 将成员 C 的 RISE 正式分支临时合入独立 review worktree，无冲突；执行合并后完整测试：
  `72 passed`。
- 用 RISE float32 `224×224` 参考图直接调用稳定性接口完成 smoke，得到有限分数
  `0.999997`，验证方法适配层、数组形状和输出结构可连通。
- 复核 split 与输出路径进入状态身份，debug 门禁不会占用后续 eval 的 resume 状态。
- 复核 reference map 的基础配置哈希、预测上下文、checkpoint 哈希和 NPY 哈希绑定；
  缺少或不匹配的 provenance 不能静默复用。
- 复核固定扰动 seed 由 `dataset/image_id/repeat` 派生，可在同数据集、同图片和同重复号下
  跨模型配对；trace 保留重复级 seed、状态和计分证据。
- 复核基础 runner 在稳定性输出已经存在但身份失效时 fail-closed，避免 resume/force
  静默覆盖或混用结果。
- 项目环境补装 SciPy 后执行 `pip check`，未发现依赖冲突。

## 状态边界

PR #10 尚未合并，因此协议在仓库层面仍应保持“候选、已通过 C/D 复核”，不能由本分支
单方面标记为 frozen。合并后还需对每种方法执行六个 460 张 eval 稳定性单元；在这些
正式结果齐全前，不生成三维 Pareto 或最终跨方法排名。

