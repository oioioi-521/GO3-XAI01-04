# XAI01-04 试跑：前沿方法探针（MA-GIG / FourierShap）

生成时间（UTC）：`2026-09-11T14:31:29.211379+00:00`

两个方法均为**试跑级操作化**（`frontier_pilot_approximation`），不是作者原实现；
VOC2007 / CHNCXR 上的结果沿用 `ood_pipeline_diagnostic` 范围，不是任务质量或医学结论。

## 环境

- 设备：`cuda`；GPU：`NVIDIA GeForce RTX 2080 Ti`
- PyTorch / torchvision / CUDA runtime：`2.8.0` / `0.23.0` / `12.8`
- MA-GIG VAE：`stabilityai/sd-vae-ft-mse`；步数 32，fraction 0.1，slerp False
- FourierShap：网格 7×7，采样 512，顶层项 64
- 每数据集图像数：`8`；稳定性 K=2，sigma 0.01

## 数据集 × 模型 × 方法汇总

| 数据集 | 模型 | 方法 | 成功/总计 | 归因耗时(s) | Insertion AUC | Deletion AUC | 稳定性 cosine | 前向样本 | 显存峰值(MiB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| chncxr_shenzhen | densenet121 | fourier_shap | 8/8 | 0.6593 | 0.0872 | 0.0194 | 0.9031 | 512.0000 | 1994.0000 |
| chncxr_shenzhen | densenet121 | ma_gig | 8/8 | 4.4008 | 0.0660 | 0.1308 | 0.0042 | 63.0000 | 2424.0000 |
| chncxr_shenzhen | resnet50 | fourier_shap | 8/8 | 0.6160 | 0.0846 | 0.0096 | 0.8288 | 512.0000 | 1976.0000 |
| chncxr_shenzhen | resnet50 | ma_gig | 8/8 | 3.9480 | 0.0043 | 0.0035 | 0.0059 | 63.0000 | 2404.0000 |
| chncxr_shenzhen | vgg16 | fourier_shap | 8/8 | 0.9948 | 0.0512 | 0.0246 | 0.6593 | 512.0000 | 4544.0000 |
| chncxr_shenzhen | vgg16 | ma_gig | 8/8 | 3.9729 | 0.0061 | 0.0081 | 0.0225 | 63.0000 | 2414.0000 |
| voc2007 | densenet121 | fourier_shap | 8/8 | 0.6597 | 0.2326 | 0.0287 | 0.9533 | 512.0000 | 1994.0000 |
| voc2007 | densenet121 | ma_gig | 8/8 | 4.4681 | 0.0799 | 0.0121 | 0.0066 | 63.0000 | 2424.0000 |
| voc2007 | resnet50 | fourier_shap | 8/8 | 0.6177 | 0.3333 | 0.0608 | 0.9362 | 512.0000 | 1418.0000 |
| voc2007 | resnet50 | ma_gig | 8/8 | 3.9030 | 0.0980 | 0.0119 | 0.0030 | 63.0000 | 1846.0000 |
| voc2007 | vgg16 | fourier_shap | 8/8 | 1.0196 | 0.2623 | 0.0427 | 0.9455 | 512.0000 | 4512.0000 |
| voc2007 | vgg16 | ma_gig | 8/8 | 3.9956 | 0.0245 | 0.0169 | 0.0056 | 63.0000 | 2384.0000 |

## 初步结论与警告

- **MA-GIG（试跑步数 32）**：单图归因约 4 s，Insertion AUC ≈ 0.05，稳定性 cosine ≈ 0.01（接近噪声）。
  隐空间 Guided-IG 路径在低步数下严重延迟：目标概率在路径末端才跳变，路径积分不收敛。
  论文默认 `num_steps=200`，试跑成本曲线测得约 24.7 s/图（0.12 s/步）；即使在 200 步，本 OOD 设置下
  路径积分仍明显偏小（单图 signed_sum ≈ 0.22，而 f(x)-f(black) ≈ 0.66）。
  结论：该操作化下 MA-GIG 是**最贵且质量最不稳定**的方法，不建议直接进主矩阵；
  若要采用，需按官方仓库（含其指定 VAE/分类器/数据集）复现后重新评估。
- **FourierShap（512 采样）**：单图归因约 0.6–1.0 s，Insertion AUC ≈ 0.05–0.33，稳定性 cosine ≈ 0.66–0.95，
  按时序比同采样数的 KernelSHAP 相当或更快（无逐样本反向），质量与 LIME/KernelSHAP 同量级，
  **可作为主矩阵候选**。2048 采样约 2.2 s/图。
- 成本曲线见 `frontier_cost_report.md` / `frontier_cost_records.jsonl`。
- 以上均为 `ood_pipeline_diagnostic`，样本仅 8 张/数据集，不构成对论文方法本身的评价。

## 方法与依据

- **MA-GIG**：Manifold-Aligned Guided Integrated Gradients，ICML 2026（arXiv:2605.02167）。移植官方 `cleanig` 的隐空间 Guided-IG 路径：在预训练 VAE 隐空间构造路径，逐步解码；图像空间路径积分。官方用 SD2 的 VAE（`lzyvegetable/stable-diffusion-2-1` 的 vae 子目录），本试跑用等价的公开 `stabilityai/sd-vae-ft-mse`。
- **FourierShap**：参考 NeurIPS 2025《SHAP values via sparse Fourier representation》（arXiv:2410.06300）的两阶段思路。图像特征按 7×7 分组，用 OMP 拟合稀疏多线性（Walsh–Fourier）代理，再由谱系数闭式计算 Shapley（`phi_i = Σ_{T∋i} c_T/|T|`）。PDF 未给引用，此为该文的试跑操作化。

## 复现命令

```bash
cd /home/hycx233/Courses/machine-learning/xai01-04-pilot-20260911
.venv/bin/python pilot_frontier.py --matrix --resume --max-images 8 --global-deadline-sec 2400 --task-timeout-sec 600
.venv/bin/python pilot_frontier.py --cost --resume --global-deadline-sec 1200 --task-timeout-sec 600
```

原始记录：`frontier_records.jsonl`；聚合：`frontier_summary.json`；CSV：`frontier_units.csv`、`frontier_per_image.csv`。
