# XAI01-04 试跑：两个前沿方法结论（MA-GIG / FourierShap）

生成时间（UTC）：`2026-09-11T14:47:44.194274+00:00`

> 本文档总结选做前沿方法 MA-GIG 与 FourierShap 的试跑级探针结果与结论。
> 两者均为“参考前沿论文思路”的**试跑级操作化**（`frontier_pilot_approximation`），**不是作者原实现**；
> 所有质量数字均为 VOC2007 / CHNCXR 上的 `ood_pipeline_diagnostic`，不构成对论文方法本身的评价。

## 1. 方法来源

| 方法 | 来源 | 本试跑实现 |
|---|---|---|
| MA-GIG | Manifold-Aligned Guided Integrated Gradients，ICML 2026，arXiv:2605.02167；官方代码 `github.com/leekwoon/ma-gig` | 移植官方 `cleanig` 隐空间 Guided-IG 路径；VAE 用公开等价的 `stabilityai/sd-vae-ft-mse` |
| FourierShap | PDF 无引用；最接近 NeurIPS 2025《SHAP values via sparse Fourier representation》，arXiv:2410.06300 | 7×7 特征分组；OMP 拟合稀疏多线性（Walsh–Fourier）代理；闭式 Shapley `phi_i = Σ_{T∋i} c_T/|T|` |

## 2. 实验协议

- 数据：VOC2007 16 张中的前 8 张 + CHNCXR 8 张，模型 ResNet-50 / VGG-16 / DenseNet-121。
- 目标：模型自身 ImageNet top-1 预测类（逐条记录）。
- 指标：Insertion/Deletion AUC（20 步，缓存热图）、K=2 输入扰动稳定性 cosine（sigma=0.01）、同步墙钟与显存。
- MA-GIG 矩阵用 32 步（论文默认 200 步，另做成本曲线）；FourierShap 用 512 采样（另做 128/2048 曲线）。
- 记录数：矩阵 2 数据集 × 3 模型 × 2 方法 × 8 图 = **96，全部 ok**；成本曲线 10 条。

## 3. 成本结果

| 方法 | 单图归因 | 单图总耗时（含忠实性+稳定性+IO） | 每单位成本 |
|---|---:|---:|---:|
| FourierShap 512 采样 | 0.6–1.0 s | ~1.9–3.2 s | ~1.1 ms/采样 |
| MA-GIG 32 步 | ~3.9–4.5 s | ~12–13 s | ~0.12 s/步 |
| MA-GIG 200 步（论文默认） | ~24.7 s | ~75 s | ~0.12 s/步 |

矩阵级（8 方法 × 3 模型 × 3 数据集 = 72 条件，含 ImageNet 外推，含 30% 余量）：

| 范围 | 每图合计 | 100 图 | 500 图 |
|---|---:|---:|---:|
| 标准 6 方法（PDF 主体，54 条件） | 80.8 s | 2.92 小时 | 14.58 小时 |
| FourierShap（9 条件） | 21.3 s | 46.2 分 | 3.85 小时 |
| MA-GIG 32 步（9 条件） | 111.9 s | 4.04 小时 | 20.20 小时 |
| MA-GIG 200 步（9 条件） | 673.4 s | 24.32 小时 | 121.59 小时 |
| 前沿 2 方法合计（MA-GIG 32 步，18 条件） | 133.2 s | 4.81 小时 | 24.05 小时 |
| **合计 72（MA-GIG 32 步）** | 214.0 s | 7.73 小时 | 38.63 小时 |
| **合计 72（MA-GIG 200 步）** | 775.5 s | 28.01 小时 | 140.03 小时 |

## 4. 质量结果（12 条件均值）

| 方法 | Insertion AUC | Deletion AUC | 稳定性 cosine |
|---|---:|---:|---:|
| FourierShap 512 | ~0.18 | ~0.03 | ~0.87 |
| MA-GIG 32 步 | ~0.05 | ~0.03 | ~0.01 |

## 5. 结论与建议

1. **FourierShap 建议纳入候选**：成本与 RISE/KernelSHAP/LIME 同量级，质量同量级，稳定性可接受；按 512 采样即可，2048 采样约 2.2 s/图。需先与指导教师确认 PDF 所指就是 arXiv:2410.06300。
2. **MA-GIG 不建议直接进入主矩阵**：
   - 成本最高：论文默认 200 步单图总耗时约 75 s，72 条件 500 图约需 140.03 小时；
   - 本 OOD 操作化下质量不稳定：32 步稳定性 cosine≈0.01，路径积分不收敛；即使 200 步，单图 signed_sum≈0.22，而 f(x)−f(black)≈0.66。
   - 若确需比较，应按官方仓库（指定 VAE、分类器、数据集、200 步）完整复现后单独评估，不与主矩阵混跑。
3. **组合爆炸结论不变**：真正进入主体矩阵的前沿方法最多是 FourierShap；MA-GIG 作为附录/复现项而非矩阵项。

## 6. 局限

- 16 张/数据集中的 8 张，仅计时/管线探针，不做统计显著性结论。
- 分类头域外；MA-GIG 的 VAE 为公开等价权重而非官方指定镜像；FourierShap 为论文思路的操作化，两者均不等于作者原实现。
- 质量差异可能部分来自步数、VAE、数据域，而非方法本身；MA-GIG 的负面结果需在官方设置下复核。

## 7. 复现命令

```bash
cd xai01-04-pilot-20260911
export HF_HOME="$PWD/vae_cache"
.venv/bin/python pilot_frontier.py --cost --resume
.venv/bin/python pilot_frontier.py --matrix --resume --max-images 8 --ma-gig-steps 32 --fourier-samples 512
.venv/bin/python frontier_tables.py
```

## 8. 文件

- 记录：`frontier_records.jsonl`（96）、`frontier_cost_records.jsonl`（10）
- 报告：`frontier_report.md`、`frontier_cost_report.md`、`frontier_matrix_timing.md`、本文 `frontier_conclusions.md`
- 表：`frontier_units.csv`、`frontier_per_image.csv`、`frontier_matrix_timing.csv`
- 脚本：`pilot_frontier.py`、`frontier_tables.py`
- 环境快照：`environment_snapshot_frontier.json`
