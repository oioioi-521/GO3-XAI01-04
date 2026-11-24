# B / M5 交付入口

2026-10-26 · YaoYang-123456 · Refs #19。基于 `feat/kernelshap-analysis` 的 `dafb65591349d426637855a6d4b6c9b52b979b64`；分支 `docs/b-m5-review-20261026`。报告/PPT/统计终稿仍由对应成员合稿，本文不标记全组已验收。

## 11/4 最终共同口径核对状态

当前记录：[B最终报告/PPT核对](B_M5_FINAL_REPORT_PPT_REVIEW_20261104.md)、[轻量回执](../analysis/b_m5_20261104/final_review_receipt.json)。实际对象为A报告v4 `3cfd4a8`、E十二页v7 `d00553f`，B原材料采用已获A批准的 `f224484`。报告B责任口径、PPT可见内容与所查数值相容；**第1/6/11/12页备注仍需E按记录修正，暂不宣布整套展示最终通过**。具体意见在[Issue #19](https://github.com/oioioi-521/GO3-XAI01-04/issues/19)发布一次；Issue保持开放，PR #25保持Draft。

A的24组NPY补验、D的IG/KS及v7统计确认、E的三例和并列规则确认、A对C两处门禁的修复均已有记录，**不再列为首次待办**。本轮没有重跑训练/归因/MoRF/完整统计/选图，没有把他人核验改记为B执行。当前必要文字修改由E负责；C新增统计独立复现、独立设备Demo、E→A与全组计时继续由各责任人留回执。

下方是10/26交付与发布历史入口；其中当时的“待E/D/C确认”“NPY未收到”等保留原日期和执行范围，以本段及11/4记录区分当前状态。

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
| 已发布交接与Comment review | [发布记录及正文](B_M5_HANDOFF_TEXTS.md)、[机器可读回执](../analysis/b_m5_20261026/publication_receipts.json) |

## 10/26 已完成范围与当时协作项（历史）

本轮已读取#18–#22正文/全部评论、A#23责任表及正文、C#24最新候选与E既有案例规则；保护原工作树，在独立B worktree工作。新旧轻量包身份、内部hash、CSV键/值/样本集、预测/权重绑定、源码响应量、模型隔离、候选筛选均已复核；3例原图和18张PNG/sidecar字节及视觉已检查。补充第四例常量Grad-CAM仅有trace/指标证据，map未完成视觉核验。没有新增正式实验。

- E：确认规则细化与最终3例，是否追加298；E负责最终面板、PPT和Demo。若要B核验NPY或展示298，按规则页精确补件。
- D：回复IG/KernelSHAP均值CI与秩结论的具体核对项；目前只能写“B已核对、待D确认”。
- C：处理环形复核的hash口径/四图门禁及文档状态意见；保留其历史材料，不要求重复合入旧工具。

2026-10-26，经用户明确授权，已在#19/#22/#21发布交接评论，并对#24提交一次Comment review；链接及实际发布时间见发布记录，E/D/C确认仍待回复。未提交Approve或Request changes，未merge/关闭Issue/发布main；#25保持Draft。原ZIP、图片、权重、NPY、预览拼图和大结果不入Git。
