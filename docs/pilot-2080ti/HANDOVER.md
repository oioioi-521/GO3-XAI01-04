# XAI01-04 试跑交接文档

> 项目：XAI01-04 归因方法模式发现（`project-pool.pdf` 第 63–65 页）
> 试跑目录：`xai01-04-pilot-20260911/`
> 交接日期：2026-09-11　撰写：试跑执行方
> 本文档汇总本次试跑（P0–P4 + 外推）的**实测结果、结论、组合爆炸应对方案、局限与待办**，供接手同学复现与继续推进。

---

## 0. 一句话总结

在限时 GPU 预算内验证了整套“方法 × 模型 × 数据集 × 指标”的测量管线：**2 数据集 × 3 模型 × 6 方法 × 16 图 = 576 单元零失败**，并额外完成两个选做前沿方法（MA-GIG / FourierShap）的试跑级探针（96 单元零失败），得到可直接外推的成本表与初步指标证据。结论是：**组合爆炸不靠堆算力，而靠“分层矩阵 + 冻结超参 + 包含式递增 + 缓存复用 + K 分级”控制**；完整 54 条件在 500 图上约 14.6 GPU 小时，一次夜间批量可完成。前沿方法方面：FourierShap 便宜稳定、可作候选；MA-GIG 最贵且本 OOD 操作化下质量不稳定、不建议直接进主矩阵。

---

## 1. 试跑范围与边界

- 只依据 `project-pool.pdf` 第 63–65 页 XAI01-04 及第 57/59/61 页的指标定义、`intro-mlproj.pdf` 的 XAI Track 说明。
- 目标是**测清各环节开销、建立可外推成本表**，不是训练模型，也不是宣布某方法最优。
- 本轮不做统计显著性结论（样本 16 张/数据集不足）。
- 所有 VOC2007 / CHNCXR 结果均标记 `quality_scope=ood_pipeline_diagnostic`：ImageNet 预训练头对这两个数据集属域外，**不是任务忠实性或医学结论**。

---

## 2. 机器与环境（实测快照）

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti，11 GiB（桌面显示卡，保留约 1.5 GiB 余量） |
| CPU / 内存 | Intel i7-8700K（6 核 12 线程）/ 约 46 GiB |
| 驱动 / CUDA runtime | 610.57.04 / CUDA runtime 12.8（Python 侧） |
| PyTorch / torchvision | 2.8.0 / 0.23.0 |
| 归因库 | Captum 0.9.0（IG / KernelShap / Lime / FeatureAblation / LayerGradCam） |
| 说明 | 驱动显示的 CUDA 13.3 不代表 Python 实际 runtime；实测占用见各 JSONL |

快照文件：`planning_baseline.json`、`environment_snapshot.json`、`environment_snapshot_p2.json`。

---

## 3. 数据与目标语义

| 数据集 | 规模 / 抽样 | 标签 | 目标类别 |
|---|---|---|---|
| VOC2007 | 官方 `VOCtrainval_06-Nov-2007.tar` 的 trainval，种子 20260911 无放回抽 **16 张** | 多标签（20 类） | 取模型自身 ImageNet top-1 预测，逐条记录 |
| CHNCXR Shenzhen | NLM 深圳胸片集，按标签分层抽 **16 张**（8 normal / 8 abnormal） | normal / abnormal | 同上（域外探针） |

- 采样只做一次、全组共用；来源 URL、SHA256、抽样规则见 `data_manifest.json`。
- 正式实验前必须完成“数据任务 ↔ 模型类别”的映射（ImageNet 用对应类；VOC 需类别映射或适配分类器；CHNCXR 需可靠任务检查点与患者级划分）。本轮未做微调。
- 归因 target 在后续所有方法/扰动/忠实性中**固定为原图预测类**；不得用扰动后的 top-1 重定 target。

---

## 4. 已完成阶段与产物

| 阶段 | 内容 | 记录数 | 结果 |
|---|---|---:|---|
| P0 环境与首通 | 版本/权重/数据来源；6 方法首通 | 6 | 真 CUDA 执行，结果有限 |
| P1 合成图校准 | 6 方法 × 3 模型 × 3 重复，合成图计时 | 54 | 全部 ok，见 `calibration_report.md` |
| P2 小规模完整评价 | 2 数据集 × 3 模型 × 6 方法 × 16 图 | **576（全部 ok）** | 见 `eval_report.md` |
| P3 采样成本曲线 | IG steps / RISE masks / SHAP·LIME samples / internal batch | 20 | 见 `cost_curve_report.md` |
| P4 随机方法种子复核 | RISE/KS/LIME × 8 图 × 3 种子 | 24 | 见 `recheck_report.md` |
| P5 前沿方法探针 | MA-GIG / FourierShap × 2 数据集 × 3 模型 × 8 图 | **96（全部 ok）** | 见 `frontier_report.md`、`frontier_cost_report.md` |
| 外推 | 100/500/1000 图与完整矩阵成本 | — | `extrapolation.md`、`matrix_timing_full.md` |

P2 每单元包含：1 次原图归因 + 缓存热图上的 Insertion/Deletion（20 步 / 21 节点梯形 AUC）+ K=2 高斯噪声（raw [0,1]，sigma=0.01）重新归因的 cosine；指标复用同一份缓存归因，**不是乘法项**。

---

## 5. 核心结果

### 5.1 效率（秒/图，K=2，含忠实性 + 稳定性 + IO）

| 方法 | 仅归因 | 2 次扰动归因 | 忠实性 | 单图总 | 相对 Grad-CAM |
|---|---:|---:|---:|---:|---:|
| Grad-CAM | 0.013 | 0.017 | 0.086 | **0.14** | 1× |
| Ablation | 0.086 | 0.172 | 0.082 | **0.36** | ~2.6× |
| IG | 0.106 | 0.205 | 0.082 | **0.41** | ~3× |
| RISE | 0.774 | 1.544 | 0.082 | **2.42** | ~18× |
| KernelSHAP | 0.891 | 1.763 | 0.082 | **2.78** | ~20× |
| LIME | 0.920 | 1.842 | 0.081 | **2.86** | ~21× |

**要点**：RISE / KernelSHAP / LIME 三种占全矩阵约 **90%** 时间，却只占一半条件；忠实性成本与方法几乎无关（约 0.08 s，42 次批量前向）；稳定性成本 ≈ K × 原归因。

### 5.2 忠实性 / 稳定性（每个方法在 3 模型 × 2 数据集共 6 个单元上的均值）

| 方法 | Insertion AUC | Deletion AUC | 稳定性 cosine |
|---|---:|---:|---:|
| LIME | 0.198 | 0.032 | 0.952 |
| RISE | 0.160 | 0.034 | **0.999** |
| KernelSHAP | 0.145 | 0.040 | 0.961 |
| Ablation | 0.141 | 0.060 | 0.878 |
| Grad-CAM | 0.093 | 0.063 | 0.926 |
| IG | **0.043** | 0.019 | **0.509** |

- VOC2007 上 Insertion 明显高于 CHNCXR（分类头域外，概率曲线接近平坦）。
- IG 稳定性最低（约 0.35–0.63，梯度噪声大）；RISE 接近 1.0（随机掩码对小幅输入噪声不敏感）。
- 逐条件数值见 `eval_report.md` / `eval_units.csv`；这是域外管线证据，不能当任务质量结论。

### 5.3 随机方法种子方差（P4，跨种子 cosine）

| 方法 | 均值 | 范围 |
|---|---:|---|
| RISE | 0.999 | 0.998–0.999 |
| LIME | 0.929 | 0.866–0.978 |
| KernelSHAP | **0.417** | 0.181–0.734 |

说明：KernelSHAP 的 Monte-Carlo 噪声远大于输入扰动敏感性，正式实验需为它单独加种子/提高采样数；这部分与 5.2 的 K=2 稳定性是两码事。

### 5.4 成本曲线（P3，ResNet-50，单图，秒）

| 方法 | 低档 | 中档 | 高档 |
|---|---:|---:|---:|
| IG `n_steps` | 16: 0.106 | 32: 0.096 | 64: 0.182 |
| RISE `n_masks` | 128: 0.149 | 512: 0.595 | 2048: 2.380 |
| KernelSHAP `n_samples` | 128: 0.854 | 512: 0.701 | 2048: 2.783 |
| LIME `n_samples` | 128: 0.224 | 512: 0.740 | 2048: 2.965 |
| Ablation `grid` | 7: 0.072 | 8: 0.082 | — |
| internal batch=8/16/32 | RISE 0.677/0.597/0.570 | KernelSHAP 0.764/0.700/0.670 | — |

**要点**：成本随采样数近似线性；2048 约为 512 的 3–4 倍；内部 batch 16→32 仅 5–10% 收益，不值得为它改变归因数值。

### 5.5 组合爆炸：实测外推（含 30% 工程余量）

| 方案 | 条件数 | 每图合计 | 100 图 | 500 图 | 1000 图 |
|---|---:|---:|---:|---:|---:|
| **推荐主体**（4 方法 × 2 模型 × 2 数据集） | 16 | 24.1 s | 52 min | **4.35 h** | 8.70 h |
| 完整实测矩阵（6 × 3 × 2） | 36 | 53.85 s | 1.94 h | 9.72 h | 19.45 h |
| **PDF 完整候选**（6 × 3 × 3） | 54 | 80.78 s | 2.92 h | **14.58 h** | 29.17 h |

逐条件表见 `matrix_timing_full.md` / `matrix_timing_full.csv`。

### 5.6 前沿方法探针（MA-GIG / FourierShap，2 数据集 × 3 模型 × 8 图）

两个方法均按“参考前沿论文思路”做**试跑级操作化**（`frontier_pilot_approximation`），不是作者原实现。

| 方法 | 归因耗时/图 | Insertion AUC | 稳定性 cosine | 显存峰值 | 说明 |
|---|---:|---:|---:|---:|---|
| MA-GIG（32 步） | 4.1 s | 0.046 | **0.008** | ~2.4 GiB | 隐空间 Guided-IG；路径末端才跳变，积分不收敛 |
| FourierShap（512 采样） | 0.76 s | 0.175 | 0.871 | ~2.0–4.5 GiB | 谱代理 + 闭式 Shapley |

**成本曲线**
- MA-GIG：约 **0.12 s/步**；4/8/16/32/64/128/200 步 = 0.53/0.86/1.84/3.94/7.75/15.7/**24.7** s/图。论文默认 200 步。
- FourierShap：约 **0.0011 s/采样**；128/512/2048 = 0.23/0.61/2.24 s/图。

**结论**
- **MA-GIG**：最贵且在本 OOD 设置下质量最不稳定。低步数路径积分不收敛（32 步稳定性 cosine≈0.01）；即使 200 步，单图 signed_sum≈0.22（应约为 f(x)−f(black)≈0.66），说明该操作化需要按官方仓库（指定 VAE/分类器/数据集）复现后才能判断。**不建议直接进主矩阵。**
- **FourierShap**：便宜、稳定，Insertion AUC 与 LIME/KernelSHAP 同量级，**可作为主矩阵候选**；PDF 未给引用，代码依据为 arXiv:2410.06300。
- MA-GIG 依据 ICML 2026（arXiv:2605.02167，官方代码 `leekwoon/ma-gig`），VAE 用等价的 `stabilityai/sd-vae-ft-mse`。

---

## 6. 结论：如何应对组合爆炸

1. **分层矩阵，不做全笛卡尔积**：主体 4 方法（Grad-CAM / IG / KernelSHAP / RISE）× 2 模型（ResNet-50 / VGG-16）× 2 数据集 = 16 条件；LIME / Ablation / DenseNet-121 只进小样本补充矩阵。KernelSHAP 与 LIME 机制重叠，二选一可省约 1/6 时间。
2. **冻结超参，超参只在校准图扫**：主矩阵固定 224×224、黑基线、IG 32 步、SHAP/LIME/RISE 512 采样。
3. **包含式递增**：debug 32 → 100 张选优 → 保留子集 500 张确认；原图保持包含关系。
4. **缓存复用 + 断点续跑**：归因/端点/扰动图全部缓存，忠实性、稳定性、Pareto、相关性、ANOVA 共享同一批逐图记录；seed/target/权重/配置变化时严格更换缓存键。
5. **K 与 seed 分级**：主体 K=2；K 2→10 会让全矩阵成本约 3.5 倍，是比图像数更陡的杠杆；随机方法先测 seed 方差（P4 已测），只对候选加种子。
6. **按成本分配样本**：廉价方法可给 500 张，昂贵方法给 100 张并如实注明子集（效率本身是评价维度）。
7. **统计上不再乘维度**：图像是配对重复测量单元、嵌套于数据集；先用配对差异/效应量选候选，再在保留子集确认。

---

## 7. 局限与风险

- 16 张/数据集是**计时子集**，不是评估集；不做 ANOVA / Pareto / 显著性声明。
- 分类头域外；CHNCXR 仅作 OOD 管线探针，未做结核分类。
- 计时含 CPU 回归与同步等待，报告为墙钟；不同机器/占用会有波动。
- 结果目录建议不入 git；归因缓存与原始记录是否保留需与组内约定。
- 归因 target 取预测类而非真值；正式“方法 × 模型”质量交互研究需固定可比较的语义目标或把目标选择列为条件。

---

## 8. 复现命令

```bash
cd xai01-04-pilot-20260911

# P1 合成图 6×3 校准
.venv/bin/python pilot_runner.py --calibration --resume --global-deadline-sec 900 --task-timeout-sec 180

# P2 小规模端到端评价（2 数据集 × 3 模型 × 6 方法 × 16 图）
.venv/bin/python pilot_eval.py --evaluate --resume --max-images 16 --global-deadline-sec 2400 --task-timeout-sec 180

# P3 成本曲线
.venv/bin/python pilot_eval.py --cost-curve --resume --global-deadline-sec 1200 --task-timeout-sec 300

# P4 随机方法种子复核
.venv/bin/python pilot_eval.py --recheck --resume --replicates 3 --max-images 8 --global-deadline-sec 900

# 汇总（外推 + 完整矩阵用时 + 结果索引），不占 GPU
.venv/bin/python summarize_pilot.py
.venv/bin/python full_matrix_timing.py
```

---

## 9. 关键文件清单

**记录（原始，追加写）**
- `calibration_records.jsonl`（P1，54）、`smoke_records.jsonl`、`rise_smoke_records.jsonl`
- `eval_records.jsonl`（P2，576）、`cost_curve_records.jsonl`（P3，20）、`recheck_records.jsonl`（P4，24）
- `frontier_records.jsonl`（P5，96）、`frontier_cost_records.jsonl`（P5 成本，10）

**报告 / 汇总（中文）**
- `RESULT_INDEX.md`（总索引）、`eval_report.md`、`cost_curve_report.md`、`recheck_report.md`、`calibration_report.md`
- `frontier_report.md`（前沿方法）、`frontier_cost_report.md`
- `extrapolation.md`、`matrix_timing_full.md`
- 本文件 `HANDOVER.md`

**表 / 数据**
- `eval_units.csv`（method,model,dataset,metric,mean,std,n,config_hash）
- `eval_per_image.csv`（image_id,dataset,model,method,metric,value,time_ms）
- `frontier_units.csv`、`frontier_per_image.csv`、`frontier_summary.json`
- `matrix_timing_full.csv`、`extrapolation.json`、`eval_summary.json`、`data_manifest.json`

**脚本**
- `pilot_runner.py`（P0/P1）、`pilot_eval.py`（P2/P3/P4）、`pilot_frontier.py`（P5）
- `summarize_pilot.py`、`full_matrix_timing.py`

**环境快照**
- `planning_baseline.json`、`environment_snapshot.json`、`environment_snapshot_p2.json`
- `environment_snapshot_frontier.json`、`vae_cache/`（SD VAE 权重，约 320 MB）、`hardware_baseline_nvidia.txt`

---

## 10. 待办 / 下一步

1. **两个选做前沿方法（已完成试跑级探针，见 §5.6）**：
   - **MA-GIG**：ICML 2026《Manifold-Aligned Guided Integrated Gradients》（arXiv:2605.02167，官方代码 `leekwoon/ma-gig`），已在 `pilot_frontier.py` 中移植隐空间 Guided-IG 路径。结论：本 OOD 操作化下路径积分不收敛、稳定性≈0，**不进主矩阵**。若要正式采用，需按官方仓库（指定 VAE/分类器/数据集）完整复现后再评估。
   - **FourierShap**：无同名论文，按 NeurIPS 2025《SHAP values via sparse Fourier representation》（arXiv:2410.06300）做谱代理操作化。结论：便宜、稳定，**可作为主矩阵候选**。仍需与指导教师确认 PDF 所指确为该文。
2. **正式实验前置**：确定主体 16 条件的图像规模（100 → 500）、数据集任务映射（VOC/CHNCXR 的类别与划分）、是否加入 ImageNet。
3. **统计阶段**：从同一批 `eval_records.jsonl`（必要时并入 `frontier_records.jsonl`）直接构造 ANOVA / Pareto / 相关性 / 聚类 / 决策树，无需重新归因。
4. **版本控制**：`results/`、`data/`、`vae_cache/`、`weights_cache/` 不进 git；仅提交脚本与 schema（见指导文件 §7.2）。

---

*本文档所有数字均来自 `xai01-04-pilot-20260911/` 内的 JSONL / CSV / 报告，可由第 8 节命令复现。*
