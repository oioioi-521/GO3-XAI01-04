# XAI01-04 归因方法模式发现

《机器学习综合实践》五人项目：在固定预算下比较六种归因方法、三种模型和两种数据集，研究优势模式、三指标权衡和指标冲突。2026-11-21：C 已整合 A 报告 v4、B 案例复核、D 最终统计及 11/20 收尾、E 展示 v7，并独立复现 10/15、10/28、10/29 三批统计；当前为待评审候选。

## 交付入口

| 内容 | 入口 | 当前状态 |
| --- | --- | --- |
| C 的集成与复现 | [交付索引](docs/C_M5_DELIVERY.md)、[整合清单](docs/C_M5_INTEGRATION.md)、[11/21 独立复现](docs/C_M5_REPRODUCTION_20261121.md) | 新增统计已复现，浮点末位差异完整披露 |
| 统一数据与大归档 | [36 单元交接与获取要求](docs/DCU_TEAM_HANDOFF_20261014.md) | 组内受控交接，原始数据不进 Git |
| 正式统计 | [原配对统计](docs/DCU_STATISTICAL_ANALYSIS_20261015.md)、[10/28 重复测量](docs/D_M5_REPEATED_MEASURES_20261028.md)、[10/29 混合设计](docs/D_M5_MIXED_FACTORIAL_20261029.md) | 冻结 DCU 输入；同任务比较与跨任务原始评分差分开解释 |
| 报告 v4 | [报告正文](docs/FINAL_REPORT_DRAFT.md)、[A 交接](docs/FINAL_REPORT_HANDOFF.md)、[B 案例](docs/B_M5_CASE_ANALYSIS.md) | 已接入；历史待办以 C 当前索引校正 |
| 展示 v7 | [PPTX](deliverables/e_m5/GO3-XAI01-04_E_M5_展示初稿_20261101_v7.pptx)、[PDF](deliverables/e_m5/GO3-XAI01-04_E_M5_展示初稿_20261101_v7.pdf) | 已含四页备注修订及 D 11/20 确认 |
| 三例 Demo | [启动与来源](deliverables/e_m5/demo/README.md)、[独立演示步骤](deliverables/e_m5/REPRODUCE_DEMO.md)、[C 本机检查](analysis/c_m5_20261121/demo_receipt.json) | 本机资源/HTTP 已核对；异机浏览器演练待实测 |
| 当前分工 | [TEAM_AND_ROADMAP](docs/TEAM_AND_ROADMAP.md) | #18–#22，另保留历史实验分工 |

候选分支为 `integrate/m5-delivery-20261026`，面向 main 的 [PR #24](https://github.com/oioioi-521/GO3-XAI01-04/pull/24) 保持 Draft。原整合起点 `2989537` 及 #14/#15/#16 的祖先关系保留；三个旧 PR 已关闭，当前来源固定版本见[整合清单](docs/C_M5_INTEGRATION.md)。异机 Demo、B/E 修订后确认、E→A 与全组实际计时仍需完成后评审发布。

## CPU 独立分析复现

统计分析不需要 torch、GPU、原图或权重。使用独立 Python 3.12 环境，避免与归因环境的 pandas 版本要求混装。以下为 PowerShell 命令，在本仓库根目录执行：

```powershell
python -m venv .venv-analysis
& ./.venv-analysis/Scripts/python.exe -m pip install -r requirements-analysis.txt
& ./.venv-analysis/Scripts/python.exe scripts/reproduce_analysis.py --handoff 'D:/path/dcu_team_handoff_20261014.zip' --output-dir results/reproduction
& ./.venv-analysis/Scripts/python.exe scripts/reproduce_m5_extensions.py --handoff 'D:/path/dcu_team_handoff_20261014.zip' --output-dir results/reproduction_extensions
& ./.venv-analysis/Scripts/python.exe -m pytest -q tests/test_dcu_statistics.py tests/test_reproduce_analysis.py
```

`--handoff` 指向组内取得的原始轻量 ZIP，允许文件名有差异，内容必须匹配：

```text
13,707,240 bytes
SHA-256 14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc
```

脚本先核对整包 SHA，再调用本仓库的校验/分析模块，重建七张统计表、四张 PNG、统计报告和配对表；不执行 ZIP 内源码。输出包含 `receipt.json`、三份步骤日志和产物哈希。输出目录必须位于本 checkout 的 `results/` 下且尚不存在；重复验收请换目录，保留前次证据。

Python 源文件固定为 LF，确保 Windows 源码哈希与 D 已发布记录一致。结果与差异解释见[独立复现记录](docs/C_M5_REPRODUCTION.md)。CPU 统计复现、现成归因图展示和 GPU 重新归因分别验收。

第二条复现命令以原发布预算重建 10/28、10/29 的 894 条派生记录、三张图、两篇报告及附表，并另做 42 项混合设计公式核验。图片须原字节一致，输入/源码/协议/离散字段严格相同；数值容差及首次失败证据见 [11/21 记录](docs/C_M5_REPRODUCTION_20261121.md)，通过状态明确为 `passed_with_float_differences`，不声称 JSON 逐值完全相同。

## 归因开发与测试

使用另一环境安装 `requirements.txt`；包括历史 ANOVA 工具的全仓测试还需 `analysis/requirements-analysis.txt`。不要把根目录锁定的 CPU 复现依赖混装进归因环境。

```powershell
python -m venv .venv-runner
& ./.venv-runner/Scripts/python.exe -m pip install -r requirements.txt -r analysis/requirements-analysis.txt
& ./.venv-runner/Scripts/python.exe -m pytest -q
```

原图、权重和正式路径按受控交接配置。服务器快照内的绝对路径保留为证据，重新部署时需映射。旧 GPU 配置保留原身份，六方法最终分析统一读取 DCU 包，不混入历史 RTX 2080 Ti / RTX 4060 结果。

RISE 历史预算与批处理入口见 [RISE_FORMAL_HANDOFF](docs/RISE_FORMAL_HANDOFF.md)，正式协议见 [STABILITY_PROTOCOL](docs/STABILITY_PROTOCOL.md)。`analysis/rise_formal_batch.py` 用于历史基础指标流程，不接管 DCU/LIME 队列或验收已追加稳定性的统一表。

## 样本与结论范围

- 方法为 IG、Grad-CAM、Occlusion、RISE、KernelSHAP、LIME；模型为 ResNet50、DenseNet121、VGG16；ImageNet / VOC 各 460 张 eval。
- 单模型主分析用模型命中组；跨模型用三模型共同命中组。VOC 命中表示冻结单目标为 sigmoid argmax，不是多标签 exact match 或 mAP。
- raw MoRF AUC 越低越好，RGB 噪声稳定性越高越好，wrapper 耗时越低越好；五次扰动先按原图汇总。
- RISE 六组平均噪声稳定性最高，IG 五组平均 AUC 最低；不存在据此可宣布的通用全指标冠军。共享 DCU 两轮计时不是受控硬件排名。

## 数据与文件约定

冻结 metadata / 标签和小型派生统计入 Git；原始图片、checkpoint、逐图大表、NPY、ZIP 和完整输出通过组内受控渠道获取。归档名、大小和 SHA 见[交接说明](docs/DCU_TEAM_HANDOFF_20261014.md)，VOC 权重见 [manifest](docs/VOC20_CHECKPOINT_MANIFEST.json)。Demo 原图留在忽略目录；其来源侧车固定 LF，并按原字节 SHA 验收。

课程要求与早期技术方案见[小组工作指导](XAI01-04_小组工作指导.md)；课程 `project-pool.pdf` 另行提供。
