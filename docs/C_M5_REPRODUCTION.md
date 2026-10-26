# C 的独立分析复现记录

2026-10-26，使用冻结轻量交接包，在独立 Python 3.12 CPU 环境完成“ZIP → 校验 → 七张派生统计表 → 四图 → 报告”复现。自动回执状态为 `passed`，见 [完整回执](../analysis/c_m5_reproduction_receipt.json)。原始逐图数据与完整运行输出按约定留在外部交接和本地 `results/`，不入 Git。

## 代码与环境

本次运行提交：`6574ecd56bee10e0fa1212343ffd745ca93a46a8`，运行开始时工作树干净。该提交以 `29895378e930138e042fd29ad029d0f14e8b9186` 为基线，保留三个旧 PR 的合并历史。后续文档和验收回执提交不改变本次统计模块。

| 依赖 | 版本 |
| --- | --- |
| Python | 3.12.14 |
| NumPy | 2.3.5 |
| pandas | 3.0.1 |
| SciPy | 1.15.3 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.10.7 |
| pytest | 8.4.2 |

直接依赖与 D 发布版本一致，环境没有安装 torch；依赖兼容性检查通过。归因 runner 的旧环境未升级。安装与通用命令见 [README](../README.md)，自动步骤由 `scripts/reproduce_analysis.py` 调用本仓库模块完成。

## 输入与实际命令

本地输入迁至 D 盘 `new/inputs/dcu_team_handoff_20261014.zip`，13,707,240 字节，SHA-256 为 `14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`。C 盘微信原件在副本核验后删除。分析环境、缓存、临时文件和输出均位于 D 盘项目目录。

本机从集成 checkout 根目录执行：

```powershell
. ../new/Use-DDrive.ps1
& ./.venv/Scripts/python.exe scripts/reproduce_analysis.py --handoff ../new/inputs/dcu_team_handoff_20261014.zip --output-dir results/c_reproduction_final_20261026
```

`.venv` 是本机已创建的独立 CPU 环境；其他成员按 README 新建 `.venv-analysis`。`new/` 为本机存储配置，不是仓库安装前置条件。标准命令可使用任意经 hash 校验的交接包路径。

内部依次运行 `analysis.dcu_statistics`（10000 次 bootstrap、4999 次置换）、`analysis.plot_dcu_statistics`、`analysis.report_dcu_statistics`。精确参数保存在回执 `commands`，本地 `step_1.log` 至 `step_3.log` 保留完整输出；所有步骤退出码为 0。

## 结果

| 验收项 | 实际结果 |
| --- | --- |
| ZIP CRC / manifest / checksum | 720 文件、721 checksum 项通过 |
| 矩阵 / 指标 / 稳定性 | 36 单元、66,240 行指标、82,800 条 trace |
| 执行配置 | 90 份快照通过 |
| 退化披露 | 28 条 Grad-CAM trace 按协议计零，未删除 |
| 描述汇总 | 432 行逐值一致 |
| 方法总体 / 配对 | 54 / 810 行逐值一致 |
| Pareto / 相关 | 108 / 324 行逐值一致 |
| 跨模型总体 / 配对 | 36 / 108 行逐值一致 |
| 样本验收与统计协议 | 与 D 已发布 JSON 完全一致 |
| 统计 / 校验源码身份 | 两份 SHA-256 均与 D 已发布身份一致 |
| 报告 / 配对附表 | 换行规范化后全文一致 |
| 四张 PNG | 与已发布文件逐字节一致；已逐张目视检查标题、图例、坐标、样本量与限制说明 |
| 独立 CPU 软件测试 | 16 passed，覆盖 D 的 15 项测试及新增表比较回归测试 |

七张表共 1872 条派生记录，不把这些记录数当作独立样本量。原图配对和不同 cohort 的样本数由统计协议指定。四张图沿用 D 的正式结果，未重新定义指标、预算、检验族或前沿。

## 首次差异与处理

初次重建的七张表与四图已经一致，但报告中两处源码 SHA 不同。对比确认 Windows checkout 的 CRLF 与仓库 LF 产生字节差异；规范化为 LF 后两份源码分别匹配 D 已发布哈希。新增 `*.py text eol=lf`，重新从干净的 `6574ecd` 运行，自动验收全部通过。前次输出保留在本地，不覆盖失败证据。

JSON 与 Markdown 的原始文件 hash 会受系统换行影响，因此表格按解析后逐值比较，报告按文本换行规范化比较；回执仍保留本次原始产物 SHA。PNG 没有文本换行差异，本次四图均逐字节匹配。

本次统计 JSON 规范化为 LF 后 SHA-256 为 `6c098e27cc36e98c46172845ca5d5c5f7477cb0df29b23325dbe27d7ecbbe473`，与 D 的发布身份一致。

## 集成回归验证

全仓回归 **262 passed，0 failed，0 skipped**，见 [测试回执](../analysis/c_m5_test_receipt.json)。这轮测试使用原 D 盘 runner 环境，并通过 `PYTHONPATH` 加载新目录 `.cache/runner-test-extra` 中的补充依赖；没有升级旧环境。其版本与独立 CPU 复现环境不同，回执分别记录。

首次全仓测试暴露的是缺失 statsmodels、scikit-learn、LIME 及其传递依赖；补齐后全量通过。最终命令为 `python -m pytest -q --basetemp=results/pytest_m5_verified --junitxml=results/m5_verified_tests.xml`。测试临时目录、日志和补充依赖均位于 D 盘；CPU 线程环境设为 1。软件回归不作为新的 GPU 实验或计时证据。

## 验收边界

没有执行 ZIP 内的代码，没有重新运行模型推理、GPU 归因或服务器 resume。噪声图未归档，不能独立从图重算所有 Spearman；本次核对的是 trace、配置与聚合的一致性。共享 DCU 两轮计时的探索性限制保留。真实热图、固定 Demo 和最终报告 / PPT 按 [交付索引](C_M5_DELIVERY.md)由全组继续复核。
