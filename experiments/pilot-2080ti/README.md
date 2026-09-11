# XAI01-04 试跑：2080 Ti 计时与成本校准

> 本目录是 **M1/M2 阶段的限时试跑**，目的是测清归因、评价、数据准备各自的开销，建立可外推的成本表，
> 而不是训练模型或宣布方法优劣。方案见 [`../../docs/pilot-2080ti/PLAN.md`](../../docs/pilot-2080ti/PLAN.md)，
> 交接总结见 [`../../docs/pilot-2080ti/HANDOVER.md`](../../docs/pilot-2080ti/HANDOVER.md)，
> 总索引见 [`../../docs/pilot-2080ti/RESULT_INDEX.md`](../../docs/pilot-2080ti/RESULT_INDEX.md)。

## 1. 机器与环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti，11 GiB（桌面显示卡） |
| CPU / 内存 | Intel i7-8700K（6 核 12 线程）/ 约 46 GiB |
| Python / PyTorch / torchvision | 3.13 / 2.8.0 / 0.23.0 |
| CUDA runtime | 12.8 |
| 归因 / VAE | Captum 0.9.0 / diffusers 0.40.0（`stabilityai/sd-vae-ft-mse`） |

依赖见 [`requirements-pilot.txt`](requirements-pilot.txt)。

## 2. 阶段与产物

| 阶段 | 内容 | 记录数 | 记录文件（未入库，见 §5） | 报告 |
|---|---|---:|---|---|
| P0 环境与首通 | 6 方法合成图首通 | 初次 5/6；修正后 6/6 | `smoke_records.jsonl`、`rise_smoke_records.jsonl` | `smoke_report.md`、`rise_smoke_report.md` |
| P1 合成图校准 | 6 方法 × 3 模型 × 3 重复 | 54 | `calibration_records.jsonl` | `calibration_report.md` |
| P2 小规模完整评价 | 2 数据集 × 3 模型 × 6 方法 × 16 图 | 576 | `eval_records.jsonl` | `eval_report.md` |
| P3 采样成本曲线 | IG steps / RISE masks / SHAP·LIME samples / batch | 20 | `cost_curve_records.jsonl` | `cost_curve_report.md` |
| P4 随机方法种子复核 | RISE/KS/LIME × 8 图 × 3 种子 | 24 | `recheck_records.jsonl` | `recheck_report.md` |
| P5 前沿方法探针 | MA-GIG / FourierShap × 2 数据集 × 3 模型 × 8 图 | 96 | `frontier_records.jsonl` | `frontier_report.md`、`frontier_conclusions.md` |
| 外推 | 100/500/1000 图与完整矩阵 | — | — | `extrapolation.md`、`matrix_timing_full.md`、`frontier_matrix_timing.md` |

汇总报告在 `../../docs/pilot-2080ti/reports/`，聚合 JSON 在 `summaries/`，表在 `tables/`。

## 3. 数据

试跑使用两个真实数据集（采样记录见 [`data_manifest.json`](data_manifest.json)）：

- **VOC2007**：官方 `VOCtrainval_06-Nov-2007.tar` 的 trainval 子集，种子 `20260911` 无放回抽 16 张。
- **CHNCXR Shenzhen**：NLM 深圳胸片集，按标签分层抽 16 张（8 normal / 8 abnormal）。

> 原始图像**不入库**（见指导文件 §7.2）。复现时把它们按 `data/voc2007/images/`、`data/chncxr/images/`
> 放到本目录下（`data_manifest.json` 的 `runner_paths` 指定相对路径）。

**重要声明**：ImageNet 预训练分类头对 VOC2007（多标签）与 CHNCXR（医学影像）均属域外，
因此所有质量行标记 `quality_scope=ood_pipeline_diagnostic`，**不是任务忠实性结论或医学结论**；
归因目标取模型自身 ImageNet top-1 预测，逐条记录。

## 4. 复现

```bash
cd experiments/pilot-2080ti
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-pilot.txt

# P0/P1（合成图）
python pilot_runner.py --calibration --resume --global-deadline-sec 900 --task-timeout-sec 180

# P2 小规模完整评价
python pilot_eval.py --evaluate --resume --max-images 16 --global-deadline-sec 2400 --task-timeout-sec 180

# P3/P4
python pilot_eval.py --cost-curve --resume --global-deadline-sec 1200 --task-timeout-sec 300
python pilot_eval.py --recheck --resume --replicates 3 --max-images 8 --global-deadline-sec 900

# P5 前沿方法（首次会下载 SD VAE，约 320 MB）
HF_HOME="$PWD/vae_cache" python pilot_frontier.py --cost --resume
HF_HOME="$PWD/vae_cache" python pilot_frontier.py --matrix --resume --max-images 8 --ma-gig-steps 32 --fourier-samples 512

# 汇总（不占 GPU）
python summarize_pilot.py
python full_matrix_timing.py
python frontier_tables.py
```

脚本的 `--resume` 会跳过已成功的组合；超时/失败会记为 `timeout`/`error`，不会当成正常耗时。

## 5. 不进版本库的内容

原始逐条记录（`*_records.jsonl`）、权重/VAE/绘图缓存（`weights_cache/`、`vae_cache/`、`mpl_cache/`）、
归因缓存（`artifacts/`）、虚拟环境与日志均不入库（已在仓库 `.gitignore` 中忽略）。
入库的是脚本、`data_manifest.json`、报告与聚合表。

汇总脚本（`summarize_pilot.py`、`full_matrix_timing.py`、`frontier_tables.py`）读取**同一次运行**生成的
`*_records.jsonl`；若不重跑、只做代码走查，直接看 `../../docs/pilot-2080ti/` 下已入库的报告与表即可。

## 6. 关键结论（摘要，详细见 docs）

- **成本结构**：RISE / KernelSHAP / LIME 三种占全矩阵约 90% 时间，却只占一半条件；忠实性成本与方法几乎无关；
  稳定性成本 ≈ K × 原归因。
- **外推**：推荐主体 16 条件（4 方法 × 2 模型 × 2 数据集）500 图约 4.35 h；
  PDF 完整 54 条件（6 × 3 × 3）500 图约 14.58 h（含 30% 余量）。前者来自本次 VOC+CHNCXR
  两个计时数据集；正式主线仍按组内决定使用 ImageNet+VOC，ImageNet 成本属于工程外推。
- **组合爆炸应对**：分层矩阵 + 冻结超参 + 包含式递增 + 缓存复用 + K 分级，详见 `HANDOVER.md` §6。
- **前沿方法**：FourierShap 成本可控，可进入下一轮复核；MA-GIG 当前 32 步操作化的输入扰动稳定性很低，
  200 步只测了单图归因成本，不能据此判断收敛性，故不建议直接进完整主矩阵（见 `frontier_conclusions.md`）。

## 7. 代码说明

| 文件 | 作用 |
|---|---|
| `pilot_runner.py` | P0/P1：合成图计时校准 |
| `pilot_eval.py` | P2/P3/P4：真实数据评价、成本曲线、种子复核 |
| `pilot_frontier.py` | P5：参考 MA-GIG 隐空间 Guided-IG 思路与 FourierShap 思路的试跑级探针 |
| `summarize_pilot.py` | 成本外推与结果索引 |
| `full_matrix_timing.py` | PDF 完整矩阵用时表 |
| `frontier_tables.py` | 并入前沿方法后的 72 条件大表与结论文档 |

## 8. 引用

- IG：Sundararajan et al., *Axiomatic Attribution for Deep Networks*, ICML 2017
- SHAP / KernelSHAP：Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017
- Grad-CAM：Selvaraju et al., *Grad-CAM*, ICCV 2017
- RISE：Petsiuk et al., *RISE*, BMVC 2018
- LIME：Ribeiro et al., *"Why Should I Trust You?"*, KDD 2016
- MA-GIG：Kim et al., *Manifold-Aligned Guided Integrated Gradients*, ICML 2026, arXiv:2605.02167
- FourierShap（操作化依据）：Gorji et al., *SHAP values via sparse Fourier representation*, NeurIPS 2025, arXiv:2410.06300
