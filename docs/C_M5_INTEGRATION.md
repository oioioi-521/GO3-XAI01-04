# C 的最终分支整合清单

2026-10-26，对应 [#20](https://github.com/oioioi-521/GO3-XAI01-04/issues/20)。基线固定为 `29895378e930138e042fd29ad029d0f14e8b9186`。候选通过三次 merge 保留方法负责人的原始提交祖先关系。

## 三个 PR 的处理

| PR 与固定 head | 集成内容 | 保留当前实现 |
| --- | --- | --- |
| [#14](https://github.com/oioioi-521/GO3-XAI01-04/pull/14) `a2246ea66efb58815ef324cb3938162fba941c7f` | Occlusion debug 门禁配置、runner、稳定性准备/执行/验收与打包工具、对应测试和 B 历史材料 | 六份 eval YAML 已在基线，不回写 DCU 配置、冻结结果和 LIME 门禁 |
| [#15](https://github.com/oioioi-521/GO3-XAI01-04/pull/15) `cb1f1abb98cec0e69dc6231897f81a6ff26cda44` | RISE 预算采样、基础批处理、预算 CSV、测试和 checkpoint / 结果交接 | RISE 实现与六份配置保留基线；README 用最新交付说明；重复 LF 校验采用 #14 的等效实现与内容变更测试 |
| [#16](https://github.com/oioioi-521/GO3-XAI01-04/pull/16) `d7a4d6e8bb7862bad32ae6323a975ce72d196b51` | A 稳定性 runner、测试、12 单元汇总、checkpoint 交接与协议冻结说明 | README 保留当前入口，旧 GPU 结果与统一 DCU 批次分别说明 |

#14 无文本冲突。#15 冲突在 README、`preprocessing/verify_data_version.py` 及对应测试；保留当前 README 与先前集成的 LF 等效校验。#16 仅 README 冲突。原 PR 保持开放，待 A/B/D 核对及发布评审后处理，本记录不代替签收。

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
