# E: 六种归因方法的指标空间聚类

日期：2026-11-22  
任务：[#27](https://github.com/oioioi-521/GO3-XAI01-04/issues/27)，关联总任务 [#18](https://github.com/oioioi-521/GO3-XAI01-04/issues/18)  
分析基线：`integrate/m5-delivery-20261026`，`990929c40c0e1ee4202d4f8065d977f873cafbc7`

## 结果

在六个 dataset×model 条件内分别对齐指标方向并做 z-score，再合并为每种方法 18 维特征，Ward 层次聚类选择 **2 组**：

| 聚类输入 | 聚类数 | 轮廓系数 | 分组 |
| --- | ---: | ---: | --- |
| raw MoRF AUC + 噪声稳定性 + wrapper 时间 | 2 | 0.2805 | Grad-CAM、KernelSHAP、Occlusion、RISE；IG、LIME |
| raw MoRF AUC + 噪声稳定性（去掉时间） | 2 | 0.4511 | Grad-CAM、KernelSHAP、Occlusion、RISE；IG、LIME |

去掉时间后两组成员不变，分离更清楚。按各条件内方向对齐的标准分计算，两组在噪声稳定性的平均差为 2.063 个标准差（前一组相对更高）；忠实性差为 1.142 个标准差（IG/LIME 组的 raw MoRF AUC 相对更低）。这反映固定批次中的指标权衡，不代表单个方法在每个条件都胜出。

在完整 18 维特征中，IG/LIME 组的相对 wrapper 时间画像较弱，但该时间来自共享 DCU 批次，因此不把它解释为受控硬件速度差。删去时间后，k=2 的成员不变；k=3 的细分会变化，说明更细分组对共享计时特征敏感。候选 k=2、3、4 的轮廓系数均保留在结果 JSON 中。

**可用于报告的一句话：** 在每个模型和数据集内将三项指标按方向标准化后，六种方法呈现两种较粗的画像：Grad-CAM、KernelSHAP、Occlusion 与 RISE 相对偏向较高噪声稳定性，IG 与 LIME 的组均值相对偏向较低 raw MoRF AUC；这一粗分组在去除共享批次耗时后保持不变。

## 输入与范围

主分析直接读取冻结交接包内 `team-handoff-20261014/data/units_model_correct.csv` 的 36×4 个单元均值。六个条件的样本数为：ImageNet ResNet50/DenseNet121/VGG16 分别 408/371/363，VOC 分别 387/388/400。

ZIP 大小为 13,707,240 字节，SHA-256 `14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc`；722 个成员的 CRC 检查通过。目标 CSV 为 13,294 字节，SHA-256 `8115114fa94d815d5dde419216376d0aa4a799457b5dfeda862e32d090b304e8`，与包内 `manifest.json` 和 `SHA256SUMS` 一致；manifest 有 720 个文件条目，目标表含 144 行。没有解压或执行包内文件。

与仓库中 D 已验收的 `analysis/dcu_20261015/statistics.json` 对照：108 个所用 metric keys 全部对应，55 个均值浮点值完全相同，其余 53 个只存在末位差异，最大绝对差 `9.992007221626409e-16`，小于 `1e-15` 的显式比较容差；聚类成员和报告精度下的结论未变。分析运行环境：NumPy 2.3.5、SciPy 1.16.3、Matplotlib 3.10.6。没有运行 GPU 或生成新归因图。

## 方法

1. 每行对应一个方法；三项特征分别来自 `faithfulness_morf_auc_raw`、`stability_spearman` 和 `efficiency_time_ms` 的单元均值。
2. 每个 dataset×model 条件单独在六种方法间做人口标准差 z-score。raw MoRF AUC 和耗时乘以 `-1`，稳定性乘以 `+1`，因此所有变换特征均为“越高越好”。不跨数据集或模型直接比较原始数值。
3. 主分析拼接六个条件×三项指标，形成 6 行×18 列特征。时间敏感性分析移除耗时列，形成 6 行×12 列特征。三个指标各自占 6 个特征，条件等权。
4. 使用欧氏距离和 Ward linkage。对 k=2、3、4 计算平均轮廓系数，选择轮廓系数最高的 k；并列时取较小 k。两种输入下 k=2 轮廓系数最高，树状图末端合并高度也给出明显的最后一次合并间隔。六个方法只是六个待聚类对象，轮廓系数用于小样本描述，不是显著性检验。

## 复现

在仓库根目录使用 CPU 环境：

```bash
python -m pip install -r analysis/requirements-analysis.txt
python analysis/e_method_clustering_20261122.py \
  --handoff-zip /path/to/dcu_team_handoff_20261014.zip
```

脚本会先核验 ZIP 字节 SHA、CRC、manifest、`SHA256SUMS` 与目标 CSV SHA，再从 CSV 读取聚类均值；不会提取或执行包内源码。若暂时没有 ZIP，也可不传 `--handoff-zip`，从仓库中 D 已验收的汇总快照复现。可指定输出目录：

```bash
python analysis/e_method_clustering_20261122.py \
  --handoff-zip /path/to/dcu_team_handoff_20261014.zip \
  --output-dir analysis/e_method_clustering_20261122
```

产物：

- [`e_method_clustering_20261122.py`](../analysis/e_method_clustering_20261122.py)：可复现脚本
- [`unit_means.csv`](../analysis/e_method_clustering_20261122/unit_means.csv)：36 个方法×条件的原始汇总均值
- [`method_features_18d.csv`](../analysis/e_method_clustering_20261122/method_features_18d.csv) 和 [`method_features_quality_12d.csv`](../analysis/e_method_clustering_20261122/method_features_quality_12d.csv)：方向对齐、按条件标准化后的聚类特征
- [`clustering_results.json`](../analysis/e_method_clustering_20261122/clustering_results.json)：输入身份、特征定义、候选 k、距离、聚类成员和指标画像
- [`six_method_dendrograms.png`](../analysis/e_method_clustering_20261122/six_method_dendrograms.png)：含耗时与去掉耗时的对照树状图

## 限制

- 每种方法只有一个聚类对象（共六个），结论是对这六个固定方法在本批六个条件中均值画像的描述，不对其他数据、模型或预算作总体推断。
- 单元均值隐藏图像间差异；本分析没有把原图当作额外聚类对象，也没有做图像级 bootstrap 稳定性检验。
- 稳定性只表示冻结输入噪声协议下的排序一致性，不等同解释正确率。
- wrapper 时间来自共享 DCU 负载，不能作为受控硬件效率排名。
- 两组差异不构成方法优劣或因果结论；需与原六方法统计、方法配置和报告中的限制一起解释。
