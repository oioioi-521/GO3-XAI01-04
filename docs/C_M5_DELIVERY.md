# C 的交付与复现入口

2026-10-26，M5 候选。36 个 DCU 正式单元及 D 首版统计齐全；本阶段复用冻结结果，完成分支整合和独立表格分析复现。最终报告、PPT、固定 Demo 与 main 发布仍需全组评审。

| 工作 | 入口 |
| --- | --- |
| 三个 PR 的文件去向 | [整合说明](C_M5_INTEGRATION.md)、[逐文件 JSON](../analysis/m5_integration_20261026.json) |
| CPU 环境、命令、结果与差异 | [复现记录](C_M5_REPRODUCTION.md)、[验收回执](../analysis/c_m5_reproduction_receipt.json) |
| 安装使用 | [README](../README.md) |
| 输入与归档 / 权重 | [DCU 交接](DCU_TEAM_HANDOFF_20261014.md)、[checkpoint manifest](VOC20_CHECKPOINT_MANIFEST.json) |
| 正式统计与四图 | [D 交付](D_DELIVERY_INDEX.md)、[统计报告](DCU_STATISTICAL_ANALYSIS_20261015.md) |
| C 的章节与答辩材料 | [报告贡献](C_M5_REPORT_CONTRIBUTION.md) |
| 当前分工 | [TEAM_AND_ROADMAP](TEAM_AND_ROADMAP.md) |

## 固定 Demo 对接

| 模式 | 数据与路径 | 验收范围 |
| --- | --- | --- |
| CPU 统计展示 | 轻量 ZIP，经 `scripts/reproduce_analysis.py` 输出 `results/<新目录>/figures/` | 重建统计表、四图和报告，无需 GPU |
| 现成真实热图 | 前五方法大归档、LIME 归档及组内冻结原图 | 核对同图身份后展示，不声称重新归因 |
| GPU 重新归因 | 配套原图、checkpoint、配置和目标设备，映射本机路径 | 单独记录运行环境与输出，不与 CPU 复现混称 |

真实热图使用相同 dataset / model / image_id / metadata target，核对 checkpoint、配置与 map 来源。前五方法图在 `run/formal/<method>-<model>-<dataset>/maps_float/`；LIME 从 `run/merged/maps_index.json` 定位分片。两份大归档不含原始照片和模型权重，完整名称、大小与 SHA 见 [DCU 交接](DCU_TEAM_HANDOFF_20261014.md#原始归因图按需另发)。

E 在 #22 确定固定样例和演示入口，B 在 #19 复核典型/失败案例筛选。本候选提供路径契约，尚未声称 E 已确认 Demo 可运行。无原图 / NPY 时可展示真实统计复现，不把 synthetic 样例标成真实 eval。

## 发布前待办

- A/B/D 复核整合范围和本人材料；C 据评审处理原 PR。
- D 完成 #21 统计收尾，A 汇总报告，E 提供 PPT / Demo 最终入口。
- 组内确认受控数据获取位置和访问人；本索引不提供虚构下载地址。
- 候选 PR 经评审后合入 main，记录最终版本与交付入口。
