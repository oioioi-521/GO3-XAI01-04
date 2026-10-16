# 六方法 DCU 正式数据交接（2026-10-14）

2026-10-16 补充：D 已于 10 月 15 日完成该 ZIP 的受限独立表格验收与配对统计。
见 [D 统计报告](DCU_STATISTICAL_ANALYSIS_20261015.md)和
[给 E/A 的可直接整合清单](D_TO_E_HANDOFF_20261016.md)。
下文“尚待统计”为原始交接时状态；新门禁修复不回写本批次冻结结果。

六方法 × 三模型 × 两数据集共 **36 组正式结果**已完成并通过验收。每组 460 张 eval、每图五次 RGB 噪声重复。组内现在可以开始统一统计、图表和报告整合，无需为了补齐矩阵再跑一轮。

交接包：`results/dcu_team_handoff_20261014.zip`；解压目录：`results/team-handoff-20261014/`。结果包按仓库约定通过网盘或群文件交接，不进入 Git。以下文件路径均相对于解压目录。

## 建议分享什么

**全组先共享这个轻量 ZIP。** 它包含完整分析输入、便于阅读的 Excel、运行配置、源码、数据与权重清单、验收记录和结果说明。统计分析不需要模型权重、原始图片或 DCU。

| 内容 | 文件 | 用途与数量 |
| --- | --- | --- |
| 直观概览 | `outputs/20261014/dcu_summary.xlsx` | 36 组全 eval 均值、指标方向、统一样本口径；便于组会查看 |
| 全部逐图指标 | `data/per_image.csv` | 66,240 行，16,560 个方法/模型/数据集/原图组合 × 4 指标 |
| 全部单元汇总 | `data/units.csv` | 144 行，均值、样本标准差、n、原配置 hash |
| 稳定性明细 | `data/stability_trace.csv` | 82,800 行，每原图组合 5 次重复，保留 seed/target/status/score |
| 单模型正确组 | `data/per_image_model_correct.csv`、`data/units_model_correct.csv` | 55,608 行逐图指标、144 行重算汇总；单模型忠实性主分析入口 |
| 三模型共同正确组 | `data/per_image_common_correct.csv`、`data/units_common_correct.csv` | 49,320 行逐图指标、144 行重算汇总；跨模型配对入口 |
| 样本筛选依据 | `data/cohorts.csv`、`data/cohort_counts.csv` | 2,760 条模型/数据集/图像标记与 6 组计数 |
| 预测依据 | `data/predictions.csv`、`data/method_predictions.csv` | 2,760 条共同预测上下文；另存 16,560 条逐方法原始预测与配置 hash |
| 平铺均值表 | `data/unit_overview.csv` | 36 行，供图表工具导入；标准差保留在 `units.csv` |
| 配置及版本 | `configs/`、`source/` | 前五方法 30 单元、LIME 60 分片的真实配置；冻结发布代码、LIME 独立执行层及仓库源码快照 |
| 复核与说明 | `evidence/`、`docs/`、`manifest.json`、`SHA256SUMS` | 原验收证据、本次合并审计、预算局限、逐文件 SHA-256 |

## 分给谁，接下来做什么

沿用已有分工，下面是建议的接收内容与下一步，不新增实验任务：

| 成员 | 优先使用 | 建议交付 |
| --- | --- | --- |
| D：统计分析 | 正确组/共同正确组逐图表、trace、全部汇总 | 按图配对的检验、效应量、置信区间、多重比较处理、三指标 Pareto 与方法选择结论；保留全样本敏感性分析 |
| E：图表与 Demo | Excel/均值表、逐图表、LIME 优化和预算说明 | 统一六方法图表；结合原图归因 NPY 制作同图热力图和 Demo；典型/失败案例注明筛选规则 |
| B：模型与指标复核 | 预测表、cohorts、metadata、权重 hash | 核对预测正确组、真值目标与 VOC 解释口径 |
| C：数据与 pipeline | 冻结 metadata、配置、源码、合并审计 | 复核六方法图像集合与数据版本；后续分析共同使用这份交接输入 |
| A：报告统筹 | 两轮正式报告、时间口径、预算局限 | 整合实验设置、结果与限制；各方法负责人复核自己方法的结果说明 |

全组统计结论尚未在本次交接中产生。均值排序不是显著性结论，稳定性高也不自动代表解释更忠实。

## 样本口径

| 数据集 | 模型 | 全 eval | 本模型预测正确 | 三模型共同预测正确 |
| --- | --- | ---: | ---: | ---: |
| ImageNet | ResNet50 | 460 | 408 | 332 |
| ImageNet | DenseNet121 | 460 | 371 | 332 |
| ImageNet | VGG16 | 460 | 363 | 332 |
| VOC | ResNet50 | 460 | 387 | 353 |
| VOC | DenseNet121 | 460 | 388 | 353 |
| VOC | VGG16 | 460 | 400 | 353 |

单模型主分析使用该模型正确组；跨模型比较使用同数据集三模型共同正确组。全样本表完整保留，用于敏感性分析。筛选只依据预测，不根据解释分数删样本；每种方法在相同模型/数据集下的入选 ID 完全相同。

VOC 的 `correct=1` 表示**冻结单目标类别为 sigmoid 输出的 argmax**，不是多标签 exact match，也不是 mAP。ImageNet 与 VOC 的输出激活不同，raw MoRF 按数据集分别解释。

`per_image.csv` 唯一键为 `image_id,dataset,model,method,metric`；`cohorts.csv` 与共同预测表的连接键为 `image_id,dataset,model`。跨模型配对的原图键为 `image_id,dataset`。

## 指标及统计注意事项

| 指标 | 单位 / 方向 | 当前定义 |
| --- | --- | --- |
| `efficiency_time_ms` | 毫秒，越低越好 | 单次原图归因 wrapper 的实测时间，不含预测、MoRF 和文件 I/O |
| `faithfulness_morf_auc_raw` | 无量纲，越低越好 | 真值 target、21 点删除曲线的 raw AUC；不能当作相对准确率下降比例 |
| `stability_spearman` | 越高越好 | 固定方法 seed 下，5 次 sigma=0.005 RGB 高斯噪声的 score 均值 |
| `stability_valid_rate` | 0–1，越高越好 | 可定义相关系数的重复比例，不代表解释正确率 |

- 每张原图的五次噪声重复不是五个独立样本。检验、标准差和区间估计以原图为单位，保留同图六方法配对关系。单元标准差使用 `ddof=1`，没有平均 LIME 分片标准差。
- Grad-CAM 有 28 条无效重复（常量归因图）：原始 `spearman` 留空，按协议 `score=0`，计入有效率。没有删图、修图或填造相关系数。其余方法重复均有效。
- 正式稳定性是输入噪声测试；LIME 预算阶段的跨采样 seed 一致性是另一种检查，不能混写。
- LIME 为优化版 **8000 samples、batch32**。720 张 debug 回归 map 及 R² 与原版逐值一致；仍存在采样敏感性和局部线性拟合较弱的样本，8000 不是“已收敛”的证明。
- 方法的预算和参数必须一起报告，不能仅说“使用默认 LIME/RISE/SHAP”。实际配置在 `configs/`；不要用历史中期参数替代。
- 前五方法正式墙钟 4:21:06.2，LIME 正式墙钟 3:52:06.7，两轮相加 8:13:12.9。它们不是六方法同队列的一次重跑。服务器负载、并发任务组合随时间变化，耗时比较应注明这些条件。
- 历史 2080 Ti 数据仅作已有匹配单元的历史对照，不能与本包拼接成同机正式矩阵；没有匹配的 2080 Ti LIME8000 全量加速比。

## 接收与使用

解压 ZIP 后先在解压目录执行：

```bash
python verify_handoff.py
```

该命令只依赖 Python 标准库，核对清单、文件大小、SHA-256 和分析表行数，不运行 GPU。ZIP 本身的 SHA-256 在外部同名 `.sha256` 文件中。

若要使用仓库现有分析程序，解压 `source/repository.zip` 到一个独立目录并安装项目分析依赖；在仓库根目录运行，例如：

```bash
python -m analysis.validate_results \
  --per-image /path/to/team-handoff-20261014/data/per_image.csv \
  --units /path/to/team-handoff-20261014/data/units.csv
```

读取 CSV 时保留 `image_id`、`seed`、`config_hash`、SHA 等文本字段。尤其不要让 Excel 把 64 位 seed 转成浮点数。推荐程序读入：

```python
import pandas as pd

df = pd.read_csv("data/per_image_model_correct.csv", dtype=str, keep_default_na=False)
df["value"] = pd.to_numeric(df["value"], errors="raise")
# 每个数据集/模型内部，按 image_id 对齐六方法。
paired = df[df["metric"] == "faithfulness_morf_auc_raw"].pivot(
    index=["dataset", "model", "image_id"], columns="method", values="value"
)
assert not paired.isna().any().any()
```

本次重新核对六方法预测身份完全一致。LIME 合并 CSV 的置信度小数末位与前五方法最大差 `1.1102230246251565e-16`，float32 相同。共同预测表选用 IG 原始记录，逐方法表保留所有原始字符串及配置 hash。

`units_model_correct.csv` 和 `units_common_correct.csv` 重新计算各自样本的均值、样本标准差与 n，`config_hash` 仍标识原实验配置；**筛选口径由文件名和 `cohorts.csv` 指定**，不能用这个 hash 误判为全样本汇总。

## 原始归因图按需另发

做统计只需轻量包。需要同图热力图、float32 NPY 或完整运行日志时，再交接两个完整归档：

| 归档 | 字节数 | 内容 |
| --- | ---: | --- |
| `dcu_full_five_20261013_complete.tar.gz` | 1,428,278,650 | 前五方法 13,800 张基础归因 NPY/PNG、完整结果与日志 |
| `lime_vectorized_formal_complete_20261014_v2.tar.gz` | 43,356,655 | LIME 2,760 张基础归因 NPY/PNG、60 分片、源码与验收记录 |

SHA-256 分别为：

```text
852eb7c8c4eb74124b6c599d2fd54bf1f067f9c0579ce9a0542dfada040dbede  dcu_full_five_20261013_complete.tar.gz
38510f8eeda54ae38ee77acaf04580666779954ff7565b59d9fbd58438fe1cd3  lime_vectorized_formal_complete_20261014_v2.tar.gz
```

前五方法图在 `run/formal/<method>-<model>-<dataset>/maps_float/`，LIME 从 `run/merged/maps_index.json` 定位对应分片。归档不重复包含巨大模型权重和原始照片；做热力图叠加仍需组内冻结数据。五次噪声 map 没有按协议保存，不能宣称可从这些归档独立重算所有噪声 map 的相关系数。

## 版本与验收范围

仓库源码快照为提交 `f5311404b254e69ee9adf39c31bcbbfc9f3c5315`。`source/five/release/`、`source/lime/release/` 与 `source/lime/overlay/` 另外保留实际运行的冻结代码。配置内的服务器路径是运行证据，重新部署时需映射数据/权重路径，不能在成员电脑上直接启动原队列。

本次交接核对原 CSV hash、36 组 ID 与 metadata target、六方法预测/权重身份、五次 trace 及配对 seed、逐图聚合、均值和标准差、两个筛选集完整性。既有 NPY 全量验收记录一并保留，本次没有重新进行 GPU 计算。

原始预算审阅 JSON 保留较早的六整单元 worker 规划文字；最终优化版运行采用 **60 个分片、最多 8 并发**，以实际 `plan.json`、分片配置和正式报告为准，不修改冻结历史证据。

结果解读详见 [前五方法正式报告](DCU_FULL_RESULTS_20261013.md)、[LIME 正式报告](LIME_DCU_RESULTS_20261013.md) 和 [LIME 预算审阅](LIME_DCU_BUDGET_REVIEW_20261013.md)。
