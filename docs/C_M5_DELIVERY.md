# C 的交付与复现入口

2026-11-22，M5 集成候选。A/B/D/E 固定版本已纳入 `integrate/m5-delivery-20261026`；C 已从冻结包独立重算 10/15、10/28、10/29 统计。当前面向 main 的 [PR #24](https://github.com/oioioi-521/GO3-XAI01-04/pull/24) 保持 Draft，A 已补齐异机 Demo 验收，待 B/E 复核及全组计时。

| 交付 | 当前入口 |
| --- | --- |
| 安装与统一索引 | [README](../README.md) |
| A 报告 v4 | [正文](FINAL_REPORT_DRAFT.md)、[合稿交接](FINAL_REPORT_HANDOFF.md) |
| B 案例与共同口径 | [交付](B_M5_DELIVERY.md)、[案例](B_M5_CASE_ANALYSIS.md)、[四页备注复核](B_M5_FINAL_REPORT_PPT_REVIEW_20261104.md) |
| D 最终统计 | [索引](D_DELIVERY_INDEX.md)、[重复测量](D_M5_REPEATED_MEASURES_20261028.md)、[混合设计](D_M5_MIXED_FACTORIAL_20261029.md)、[11/20 确认](D_M5_NOTES_CLOSEOUT_20261120.md) |
| E 展示 v7 | [PPTX](../deliverables/e_m5/GO3-XAI01-04_E_M5_展示初稿_20261101_v7.pptx)、[PDF](../deliverables/e_m5/GO3-XAI01-04_E_M5_展示初稿_20261101_v7.pdf)、[修订说明](../deliverables/e_m5/E_M5_DELIVERY_NOTE.md) |
| E 三例 Demo | [启动/来源](../deliverables/e_m5/demo/README.md)、[复现步骤](../deliverables/e_m5/REPRODUCE_DEMO.md)、[页面](../deliverables/e_m5/demo/index.html) |
| C 来源整合 | [说明](C_M5_INTEGRATION.md)、[逐文件审计](../analysis/c_m5_20261121/integration_inventory.json) |
| C 独立复现 | [本轮范围与差异](C_M5_REPRODUCTION_20261121.md)、[原统计回执](../analysis/c_m5_20261121/original_receipt.json)、[新增统计回执](../analysis/c_m5_20261121/extensions_receipt.json) |
| C 软件和资源检查 | [测试](../analysis/c_m5_20261121/test_receipt.json)、[Demo 本机回执](../analysis/c_m5_20261121/demo_receipt.json) |
| 数据与权重 | [DCU 归档](DCU_TEAM_HANDOFF_20261014.md)、[checkpoint manifest](VOC20_CHECKPOINT_MANIFEST.json) |
| 历史交付 | [10/26 独立复现](C_M5_REPRODUCTION.md)、[当时逐文件审计](../analysis/m5_integration_20261026.json)、[C 方法章节](C_M5_REPORT_CONTRIBUTION.md) |

11/21 复现时固定来源为 A `138b865`、B `73310e8`、D `468a686`（含 `7f9db4f`）、E `e2003bf`；完整提交号见[整合说明](C_M5_INTEGRATION.md)。原 #14/#15/#16 已关闭且其提交祖先仍保留。旧文档中“C 尚未整合/新增统计未复现/D 未交最终统计”的段落只反映原日期，以本页和本轮回执为当前状态。

## 本轮验收

- 原七张统计表 1,872 条派生记录逐值一致；四图原字节相同。
- 新增两批 894 条派生记录使用原发布完整抽样预算重建；三图原字节相同，报告及附表全文一致；JSON 浮点末位差异按明确数值容差通过并完整披露。
- 全仓 299 项测试、新增门禁 10 项测试分别通过；独立 CPU 子集53项通过，不重复相加。
- 原191个受保护文件中188个不变，3个报告文件更新与 D 最新来源一致；冻结配置、输入和方法实现未变。
- 三例原图、18 PNG、18来源侧车原字节校验通过，72个指标与冻结表一致，41项本机 HTTP 检查通过。侧车 Windows 换行问题已通过 LF 属性处理。

## 异机 Demo 与最终发布

11/22，A 在独立 Arch Linux 设备上恢复三张原图，实测案例切换、各六图四指标、逐卡来源及 PDF 备用页，另用 Impress 查看 PPTX 案例页。浏览器已补全稳定性有效率并修复热图裁切和哈希溢出。见[A 的验收回执](../analysis/a_m5_20261122/demo_review.json)。以下步骤供演练复用。

```powershell
python -m http.server 8000 --directory deliverables/e_m5/demo
```

浏览器打开 `http://localhost:8000`，逐项记录实际结果：

| 必验项目 | 完成条件 |
| --- | --- |
| 设备和版本 | 设备标识、系统、浏览器及版本、日期、候选完整 commit、PPTX SHA |
| 215 / 3803 / 7612 | 每例原图显示、六方法地图 6/6、四项指标显示 |
| 页面交互 | 三例连续切换，逐例展开六份来源记录，检查控制台/加载异常 |
| 静态备用 | 打开 v7 PDF/PPTX，定位案例页和来源说明；记录备用切换是否可用 |
| 实测回执 | 由实际操作者填写成功/失败、异常和处理；不以本机 HTTP 或文件哈希代替 |

PPTX 当前 SHA-256 为 `69081383bcace59d4301c3eb4e665118a3b01e7abfe4838f6a13456d904b8158`。C 的 11/21 检查覆盖 Windows 本机文件与 HTTP。A 的 11/22 回执补齐另一设备上的交互。

发布前仍需 B/E 按新 PPTX 确认四处备注、E→A 报告/LIME 问答复核、全组实际计时及 A 组织最终冻结/评审。D 的最终统计与 11/20 备注确认已交，不再列为未交事项。C 本轮候选可用于这些剩余检查；没有自动合并 main、关闭 Issue 或代签组员结论。