# B / M5 验证记录

2026-10-26。只读取既有实验观察值、源码与归档预览，不运行正式模型或生成归因。完整回执见 [evidence.json](../analysis/b_m5_20261026/evidence.json)和 [C审查回执](../analysis/b_m5_20261026/c_review.json)。

| 对象 | 本轮实际结果 |
| --- | --- |
| 10/14正式ZIP | SHA匹配；CRC；720文件manifest、721 checksum通过；36单元/66240指标/82800 trace/90稳定性快照；metadata、预测权重、配对seed、三cohort及均值/SD/n通过 |
| 10/26补充ZIP | SHA=`a1ae2e…c2894a`；CRC；1001文件manifest和1002 checksum通过；原包722项逐字节一致，内嵌ZIP SHA匹配 |
| 全量资产目录 | 16560唯一组合；四指标、target、预测/权重、cohort与正式表连接通过；基础配置snapshot hash通过。目录中NPY/image hash不冒称本轮验过未收到的二进制 |
| 模型与预算 | VOC train4037/validation474/frozen500互斥；旧清单CRLF文本身份另记；90份正式基础配置含预算、输出维数和预热检查通过；15份冻结执行文件hash入索引 |
| 候选 | 387图得分重算；215/3803/7612与E一致；298为额外Grad-CAM常量候选；24行六方法标量及精确来源保存 |
| 可见资产 | 三例18张PNG与18份sidecar字节、身份和尺寸通过，原图hash通过且已实际查看；298原图hash通过，map未收到 |
| 统计例外 | 三cohort的n、KS−IG均值差、胜图数从CSV复核；CI/秩效应/p对照既有JSON与D报告；未重跑检验 |
| 新程序与相关工具 | 15 passed：新增筛选3项及现有选择性打包12项；人工fixture不当正式数据 |
| C候选 | 191受保护blob、旧head祖先通过；35工具/复现测试+3数据版本测试，共38 passed，0失败/跳过；C的262/16及全统计复现仅查阅 |
| 成文与重复生成 | 38处本地链接、24行案例与6行Occlusion汇总数值通过；候选表/387图得分/源码索引/例外摘录重复生成逐字节相同；密钥扫描通过，见[成文检查回执](../analysis/b_m5_20261026/written_materials_check.json) |

## 可重复命令

在B分支的分析环境执行（不依赖权重/图片可完成标量与E预览检查）：

```powershell
python -m analysis.b_m5_review --handoff 'C:\Users\hp\Downloads\dcu_team_handoff_20261014.zip' --analysis-handoff 'C:\Users\hp\Downloads\dcu_analysis_handoff_20261026.zip' --output-dir results/b_m5_fresh_check
python -m pytest -q tests/test_b_m5_review.py tests/test_package_dcu_cases.py
git diff --check
```

本轮使用原仓库`.venv`的Python3.12.14/pandas2.3.3/NumPy2.5.3，未升级依赖；可选`--image-root`指向既有授权data目录。本轮每次使用新输出目录，先前失败证据仍在忽略的results/b_m5内。

C的独立worktree固定在5079814，命令：

```powershell
python scripts/audit_m5_integration.py --output results/b_review_integration.json
python -m pytest -q tests/test_occlusion.py tests/test_occlusion_eval_configs.py tests/test_occlusion_gate.py tests/test_occlusion_stability_gate_tools.py tests/test_reproduce_analysis.py
python -m pytest -q tests/test_data_version.py
```

CPU检查设置`CUDA_VISIBLE_DEVICES=''`。没有执行历史runner、正式resume或服务器命令。

## 初次差异、失败及范围

- 训练CSV/版本JSON原字节hash与旧manifest不符；核对CRLF表示后匹配。旧manifest按原样保留，回执明确raw_match，不把规范化结果冒充原字节验收。
- 扩展到未改动的`tests/test_dcu_statistics.py`时，旧venv缺SciPy，收集失败；没有写成测试通过。收回到本轮新增筛选与相关打包逻辑后15项通过；实际正式ZIP validator及CSV复核已经执行，不以扩展测试失败代替其结果。没有安装/升级旧环境。全统计重建仍由C的已交证据负责。
- 新预算核对的首版错误计入稳定性snapshot，90项断言失败；改为只选基础`*_unit.yaml`后重新核对。该错误属于本轮工具筛选，不是正式数据缺单元；失败目录未覆盖。
- DeepSeek模型列表鉴权成功；仅一次max_tokens=1200的问答整理生成请求，50秒读超时、没有答案/usage，无自动重试，未用于材料或审计结论。密钥未存文件或Git。

本轮未重算正式bootstrap/置换、未保存或重建噪声map、未从PNG重算指标、未宣称实测硬件性能或完整科学独立复现。
