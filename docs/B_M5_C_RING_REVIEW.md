# B→C 环形复核：PR #24

2026-10-26 · B / #19 → C / #20。审查固定 head `507981464a770ee92dc43b00567f9b42f81071f4`，base 为 main `5bf6a014455d44f0c8ae7a6de148caf1b9b1acbf`。PR 是 open/draft、未合并；不能当作已发布入口。B 在独立 detached worktree 中查看并运行轻量检查，未改写 C 文件，未发 review、approval 或评论。

## 本轮亲自执行与只查阅的证据

| 类型 | 实际范围与结果 |
| --- | --- |
| 代码/文件审查 | 阅读PR最新描述与diff，重点比较2989537→候选的66个变更路径；读C_M5_DELIVERY/INTEGRATION/REPRODUCTION/REPORT_CONTRIBUTION、README、协议、审计/复现脚本及B工具测试 |
| B实际运行 | `python scripts/audit_m5_integration.py --output results/b_review_integration.json`：passed；191受保护基线文件无变化，三个旧head均为祖先；#14路径6项基线已有、30项按来源blob保留 |
| B实际运行 | CPU、`CUDA_VISIBLE_DEVICES=''`：test_occlusion、test_occlusion_eval_configs、test_occlusion_gate、test_occlusion_stability_gate_tools、test_reproduce_analysis，共35 passed；另test_data_version 3 passed，总38项，无失败/跳过 |
| B实际运行但非C全复现 | 在B工作树运行正式轻量包validator、三口径/配置/权重绑定复核及候选复现；见B evidence.json |
| 仅查阅C证据 | C回执记录七表1872条派生记录、四图、报告全文匹配，独立CPU16项及归因环境全仓262项通过；本轮**没有重新执行**其10000 bootstrap/4999置换、四图生成、全仓262项、GPU推理或服务器resume |

本轮审计回执和B文件逐项blob见 [c_review.json](../analysis/b_m5_20261026/c_review.json)。软件测试中的小型CPU模型不属于正式模型实验。没有把C回执中的“passed”写成B亲自完整重跑通过。

## B材料与旧新边界

Occlusion核心实现、六份eval配置、run_stability、D统计、冻结DCU配置及LIME门禁在受保护清单内，均保留2989537的blob；本轮不要求重新合入它们。C候选另保留B的准备/运行/验收/打包历史工具、门禁配置、测试和历史交接文档。#14原head的36个差异路径已逐文件记录；其中30项来源内容直接保留，6项基线已有。保留贡献与历史结果是可追溯证据，不使旧RTX/Windows数据变成统一DCU输入。

最新#20正文（updated 2026-10-26T06:03:27Z）说旧PR已关闭，不再要求全部并回。PR#24创建于06:45:14Z，仍写“旧PR待处理”和“已整合旧工具”；时间与描述不能简单用“正文较新”消解。实际提交确实含三次merge、旧工具已入候选；本轮如实记录差异，不删除C保留材料、不重新打开#14，也不把“保留”称为本轮必要重跑入口。

README仍将统一DCU包作为正式输入、旧GPU材料另列；C模型/指标正文正确保留VOC单目标、三cohort和共享计时边界。其“ImageNet/VOC softmax/sigmoid”段宜链接B的逐方法响应表，防止读者理解成所有attributor均解释概率。CPU复现命令只运行本仓库分析模块，不执行ZIP源码；输出新建并限制于checkout的results，有保留旧证据的约束。

## 待C修订或明确回复的具体项

1. **严格字节hash与文本规范化冲突。** `preprocessing/verify_data_version.py` 的diff把流式原字节hash改成对所有CSV/JSON先CRLF→LF；`tests/test_data_version.py`新增允许两种字节得到同hash。实测3项测试通过，证明这确实是候选行为。最新#20要求维持严格字节hash及LF约定、不在清理中引入旧兼容改动；C_INTEGRATION却称其为等效LF校验。请C按当前要求恢复严格字节比较，或明确把此函数命名/文档标为规范化文本身份且取得相应口径确认。原ZIP验证仍严格逐字节，未发现其被修改；不能把两种hash概念混称。本项是具体待处理意见，B未擅改。
2. **复现总状态未纳入图字节比较。** `scripts/reproduce_analysis.py` 中 `passed` 只要求图数=4；`figure_byte_matches_published`在后面回执中计算，没加入`passed`。C已有回执四项均true，当前结果无已知差异；但将来即使四图字节不同，总状态仍可能passed。建议先算该字典并要求all，新增一个“4图存在但1图hash不同须失败”的针对性回归。B未重跑四图，属于代码审查发现，不把假设失败写成实际失败。
3. **文档状态更新。** C_INTEGRATION/PR描述仍称旧PR开放；协议开头虽加当前状态，后文仍有未加日期的“Issue #8开放/需补跑”。请将历史段加明确日期并修正现行状态；保留原始证据正文，避免读者从当前入口误启动历史队列。
4. **基线后续差异与A材料。** C固定head只含2989537，尚不含dafb655的RQ修正/案例选择性打包；也未整合PR#23正文。这些仍是待合稿工作，不能称最终报告已整合。E分支现已有真实PNG及Demo草稿，C交付的“需要大归档”可补充小包/全量目录入口，保留数据许可。

建议C修订后复查上述两处逻辑/哈希门禁和文档差异；本轮没有给PR approval、没有代替D的统计审查，没有变更合并或发布状态。
