# XAI01-04 归因方法模式发现

《机器学习综合实践》五人项目：在固定预算下比较六种归因方法、三种模型和两种数据集，研究优势模式、三指标权衡和指标冲突。36/36 个 DCU 正式单元及 D 的首版配对统计已完成；当前处于 M4/M5 交付整合阶段。

## 交付入口

| 内容 | 入口 | 当前状态 |
| --- | --- | --- |
| C 的集成与复现 | [交付索引](docs/C_M5_DELIVERY.md)、[PR 整合清单](docs/C_M5_INTEGRATION.md)、[独立复现记录](docs/C_M5_REPRODUCTION.md) | 本分支交付候选，待组内评审 |
| 统一数据与大归档 | [36 单元交接与获取要求](docs/DCU_TEAM_HANDOFF_20261014.md) | 组内受控交接，原始数据不进 Git |
| 正式统计与四张图 | [D 交付索引](docs/D_DELIVERY_INDEX.md)、[统计报告](docs/DCU_STATISTICAL_ANALYSIS_20261015.md) | 固定 DCU 批次，保留共享机器计时边界 |
| 报告与展示 | [给 A/E 的图表清单](docs/D_TO_E_HANDOFF_20261016.md)、[C 的章节与答辩要点](docs/C_M5_REPORT_CONTRIBUTION.md) | A 汇总报告，E 汇总 PPT / Demo，尚未冻结发布 |
| 当前分工 | [TEAM_AND_ROADMAP](docs/TEAM_AND_ROADMAP.md) | #18–#22，另保留历史实验分工 |

候选基于 `29895378e930138e042fd29ad029d0f14e8b9186`，吸收 #14/#15/#16 的固定 head，保留原提交祖先关系。分支为 `integrate/m5-delivery-20261026`；评审前不将其视为 main 发布版。

## CPU 独立分析复现

统计分析不需要 torch、GPU、原图或权重。使用独立 Python 3.12 环境，避免与归因环境的 pandas 版本要求混装。以下为 PowerShell 命令，在本仓库根目录执行：

```powershell
python -m venv .venv-analysis
& ./.venv-analysis/Scripts/python.exe -m pip install -r requirements-analysis.txt
& ./.venv-analysis/Scripts/python.exe scripts/reproduce_analysis.py --handoff 'D:/path/dcu_team_handoff_20261014.zip' --output-dir results/reproduction
& ./.venv-analysis/Scripts/python.exe -m pytest -q tests/test_dcu_statistics.py tests/test_reproduce_analysis.py
```

`--handoff` 指向组内取得的原始轻量 ZIP，允许文件名有差异，内容必须匹配：

```text
13,707,240 bytes
SHA-256 14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc
```

脚本先核对整包 SHA，再调用本仓库的校验/分析模块，重建七张统计表、四张 PNG、统计报告和配对表；不执行 ZIP 内源码。输出包含 `receipt.json`、三份步骤日志和产物哈希。输出目录必须位于本 checkout 的 `results/` 下且尚不存在；重复验收请换目录，保留前次证据。

Python 源文件固定为 LF，确保 Windows 源码哈希与 D 已发布记录一致。结果与差异解释见[独立复现记录](docs/C_M5_REPRODUCTION.md)。CPU 统计复现、现成归因图展示和 GPU 重新归因分别验收。

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

冻结 metadata / 标签和小型派生统计入 Git；原始图片、checkpoint、逐图大表、NPY、ZIP 和完整输出通过组内受控渠道获取。归档名、大小和 SHA 见[交接说明](docs/DCU_TEAM_HANDOFF_20261014.md)，VOC 权重见 [manifest](docs/VOC20_CHECKPOINT_MANIFEST.json)。报告、PPT、Demo 与受控数据访问入口待评审后在交付索引更新。

课程要求与早期技术方案见[小组工作指导](XAI01-04_小组工作指导.md)；课程 `project-pool.pdf` 另行提供。
