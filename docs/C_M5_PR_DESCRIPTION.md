## 结果

将固定分析基线 `29895378e930138e042fd29ad029d0f14e8b9186` 与 #14/#15/#16 的独有工具、测试及历史材料合入交付候选，保留三个 head 的祖先关系。README、当前分工与 C 交付索引已更新；独立 CPU 环境从冻结轻量 ZIP 重建七张统计表、四图和报告。

Refs #20。PR 保持 Draft；main 发布、旧 PR 状态处理和成员签收待评审。

## 整合范围

- #14：Occlusion 门禁、稳定性执行/验收/打包及测试；#15：RISE 预算/批处理、测试和历史交接；#16：A 稳定性 runner、测试、结果及 checkpoint 说明。
- 保留 DCU 配置、RISE 激活语义、D 统计与 LIME 门禁修复。191 个受保护基线文件按 Git blob 核验一致。README 与重复 LF 校验冲突逐项记录。
- 新增 `scripts/reproduce_analysis.py` 和 `scripts/audit_m5_integration.py`，提供保留前次证据的复现入口与逐文件清单；Python 源码固定 LF。

## 验证

- 全仓 262 passed，0 failed，0 skipped；原 D 盘 runner 环境使用独立补充依赖，未升级旧环境。
- 独立 Python 3.12 CPU 环境 16 passed，无 torch，依赖兼容性检查通过。
- 输入 SHA-256：`14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`；36 单元、66,240 行指标、82,800 条 trace、90 配置通过验收。
- 七张表共 1,872 条派生记录逐值一致；报告与附表规范化换行后全文一致；四图逐字节一致并完成目视检查。
- LF 统计 JSON SHA-256：`6c098e27cc36e98c46172845ca5d5c5f7477cb0df29b23325dbe27d7ecbbe473`，与 D 发布一致。
- 原始 ZIP、照片、权重、NPY 未入库；未重跑 GPU 归因或未归档噪声图。

## 审查入口与剩余事项

从 `docs/C_M5_DELIVERY.md` 开始；文件去向见 `docs/C_M5_INTEGRATION.md` 与 `analysis/m5_integration_20261026.json`；复现命令、环境和回执见 `docs/C_M5_REPRODUCTION.md`；C 的章节与答辩要点见 `docs/C_M5_REPORT_CONTRIBUTION.md`。

请 A/B/D 核对本人工具与历史材料的保留范围，E 核对 Demo 数据路径。最终报告、PPT、Demo 与 main 发布尚未完成。D 的统计收尾由 #21 继续，报告与展示由 #18/#22 继续；旧 PR 等待本候选评审确认后处理。
