# C 的最终分支整合清单

## 11/22 评审更新

在下方 11/21 复现候选上接入 A 报告验收版 `f8c49a04ac69a5086415c671fed616654eff38d2` 和 E 的 Demo 页面修订 `fb03a22716d14d3d7df4e10dc1d40303b4f4b3f7`。逐文件审计使用这两个新来源，复现回执仍对应 11/21 的实际运行。详见[A 异机验收](../analysis/a_m5_20261122/demo_review.json)。

## 11/21 复现候选

本轮从 PR #24 的 A 门禁修复 `eb275ed535a1ba660608a9606fd3847f83460405` 接续，以 merge 保留以下固定交付及其祖先关系：

| 成员 | 固定来源 | 内容 |
| --- | --- | --- |
| A / #23 | `138b865e161d4808ab696864ce68b8daf5edafd5` | 报告 v4、三轴图、11/5 状态接收 |
| B / #25 | `73310e85614193e2e40d456a007ae16fc725fd8b` | 案例、方法指标、答辩及四页备注复核 |
| D | `468a686de1700e89456a9dcddeb4b2cf175afcf9` | 包含 `7f9db4f` 最终统计确认及 11/20 备注收尾 |
| E / #26 | `e2003bf78b8f15cbf65294580ad453d67d7da2e1` | 十二页 v7 PPT/PDF、三例 Demo、四页备注修订 |

只有 `.gitattributes` 出现文本冲突，合并保留 C 的全部 Python LF 规则和 D 的审计文件规则。另增加 Demo 侧车 JSON 的 LF 规则，解决 Windows `core.autocrlf=true` 导致原字节 SHA 改变的问题；侧车 Git blob 与 E 来源相同，没有改写清单哈希来适应换行。

[本轮逐文件审计](../analysis/c_m5_20261121/integration_inventory.json)核对五个固定来源和 #14/#15/#16 历史祖先。原 191 个受保护文件中 188 个保持基线 blob；三个明确更新项为 `analysis/report_dcu_statistics.py`、`docs/DCU_STATISTICAL_ANALYSIS_20261015.md`、`docs/D_DELIVERY_INDEX.md`，均与 D 最新来源 blob 相同。冻结配置、数据、原统计 JSON/四图及方法实现保持原身份。

```powershell
python scripts/audit_m5_candidate.py --output results/candidate_audit.json
```

10/26 的 `audit_m5_integration.py` 和其回执保留为历史验收；该脚本的“191 项全部不能变”针对当时版本，不能用于否定本轮有来源证据的三个报告更新。当前候选使用上方新审计。C 的 README、交付索引与新增工具单列，不冒充队友原文或签收。

原 #14/#15/#16 已核实为关闭且未直接合并，其源码祖先已在候选中保留，无需重开。当前候选继续在 #24 评审；没有合入 main，也没有代签 B/E 修订确认、独立设备 Demo 或实际计时。当前剩余事项见[交付索引](C_M5_DELIVERY.md)。

## 10/26 历史整合

2026-10-26，对应 [#20](https://github.com/oioioi-521/GO3-XAI01-04/issues/20)。基线固定为 `29895378e930138e042fd29ad029d0f14e8b9186`。候选通过三次 merge 保留方法负责人的原始提交祖先关系。

## 三个 PR 的处理

| PR 与固定 head | 集成内容 | 保留当前实现 |
| --- | --- | --- |
| [#14](https://github.com/oioioi-521/GO3-XAI01-04/pull/14) `a2246ea66efb58815ef324cb3938162fba941c7f` | Occlusion debug 门禁配置、runner、稳定性准备/执行/验收与打包工具、对应测试和 B 历史材料 | 六份 eval YAML 已在基线，不回写 DCU 配置、冻结结果和 LIME 门禁 |
| [#15](https://github.com/oioioi-521/GO3-XAI01-04/pull/15) `cb1f1abb98cec0e69dc6231897f81a6ff26cda44` | RISE 预算采样、基础批处理、预算 CSV、测试和 checkpoint / 结果交接 | RISE 实现与六份配置保留基线；README 用最新交付说明；重复 LF 校验采用 #14 的等效实现与内容变更测试 |
| [#16](https://github.com/oioioi-521/GO3-XAI01-04/pull/16) `d7a4d6e8bb7862bad32ae6323a975ce72d196b51` | A 稳定性 runner、测试、12 单元汇总、checkpoint 交接与协议冻结说明 | README 保留当前入口，旧 GPU 结果与统一 DCU 批次分别说明 |

#14 无文本冲突。#15 冲突在 README、`preprocessing/verify_data_version.py` 及对应测试；当时保留 README 与先前集成的 LF 等效校验，后者已由 10/31 A 修复为严格原字节哈希。#16 仅 README 冲突。旧 PR 后续已关闭，见上方当前状态。

## 逐文件证据

[机器可读清单](../analysis/m5_integration_20261026.json)覆盖每个 head 自 merge base 以来的全部差异路径，记录 baseline / source / candidate blob 与去向：`integrated` 为来源内容已集成，`already_in_baseline` 为基线已有，`retained_current_equivalent_or_newer` 为保留基线，`reconciled_in_candidate` 为候选综合处理。

```powershell
python scripts/audit_m5_integration.py --output results/m5_integration_review.json
```

该命令检查三个 head 均为候选祖先，并以 Git blob 核对基线既有 `configs/`、`data/`、`experiments/`、`models/`、`analysis/`、`scripts/` 及 D/DCU/LIME 正式说明。受保护文件有差异即失败；Python LF 属性只规范 checkout，不改动 Git 中的原代码。

## 历史与正式结果

B 的旧 GPU 门禁/稳定性、C 的 RTX 4060 RISE 基础 eval、A 的 RTX 2080 Ti 稳定性材料保留各自运行身份，不取代统一 DCU 输入，也不拼接不同硬件的计时。

原 `xai01-04` 中未提交的两份 RISE 说明和 `analysis/package_c_rise.py` 保持原样。本候选吸收 #15 固定提交，没有混入未审查的本地修改。历史文档中的“待补跑/Issue #8 开放”按当时日期阅读；当前状态以 [DCU 交接](DCU_TEAM_HANDOFF_20261014.md)和 [D 交付](D_DELIVERY_INDEX.md)为准。

发布前请 B 核对新增 Occlusion 工具与路径，A 核对稳定性 runner / checkpoint，D 核对统计表与 LIME 修复保留，E 核对 [Demo 数据契约](C_M5_DELIVERY.md)。候选先提交审查 PR，最终 main 合并等待全组评审。
