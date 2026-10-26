# B / M5 交付入口

2026-10-26 · YaoYang-123456 · Refs #19。基于 `feat/kernelshap-analysis` 的 `dafb65591349d426637855a6d4b6c9b52b979b64`；分支 `docs/b-m5-review-20261026`。报告/PPT/统计终稿仍由对应成员合稿，本文不标记全组已验收。

| 交付 | 入口 |
| --- | --- |
| §2.1/§2.2/§2.4及逐方法响应 | [模型与评价协议](B_M5_MODEL_AND_METRICS.md) |
| 先交E的规则和补件清单 | [规则v1](B_M5_CASE_SELECTION.md)、[387图得分](../analysis/b_m5_20261026/selection_scores.csv) |
| 四例身份、六方法真值与来源 | [24行候选CSV](../analysis/b_m5_20261026/case_candidates.csv)、[§8初稿](B_M5_CASE_ANALYSIS.md) |
| Occlusion说明与答辩 | [方法材料](B_M5_OCCLUSION_AND_DEFENSE.md) |
| 给D的统计例外核对 | [三口径数值和问题](B_M5_D_STATISTICAL_CHECK.md) |
| 对C的环形复核 | [PR24复核](B_M5_C_RING_REVIEW.md)、[逐文件/测试回执](../analysis/b_m5_20261026/c_review.json) |
| 轻量机器可读证据 | [evidence.json](../analysis/b_m5_20261026/evidence.json)、[冻结源码hash](../analysis/b_m5_20261026/frozen_source_inventory.csv) |
| 本轮验证及命令 | [验证记录](B_M5_VALIDATION.md) |
| 未发送的简短交接文字 | [交接草稿](B_M5_HANDOFF_TEXTS.md) |

## 已完成与需协调的最少事项

本轮已读取#18–#22正文/全部评论、A#23责任表及正文、C#24最新候选与E既有案例规则；保护原工作树，在独立B worktree工作。新旧轻量包身份、内部hash、CSV键/值/样本集、预测/权重绑定、源码响应量、模型隔离、候选筛选均已复核；3例原图和18张PNG/sidecar字节及视觉已检查。补充第四例常量Grad-CAM仅有trace/指标证据，map未完成视觉核验。没有新增正式实验。

- E：确认规则细化与最终3例，是否追加298；E负责最终面板、PPT和Demo。若要B核验NPY或展示298，按规则页精确补件。
- D：回复IG/KernelSHAP均值CI与秩结论的具体核对项；目前只能写“B已核对、待D确认”。
- C：处理环形复核的hash口径/四图门禁及文档状态意见；保留其历史材料，不要求重复合入旧工具。

不代发评论或approval、不merge/关闭Issue/发布main。草稿PR仅提交本轮可审查材料。原ZIP、图片、权重、NPY、预览拼图和大结果不入Git。
