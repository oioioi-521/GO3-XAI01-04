#!/usr/bin/env python3
"""为前沿方法（MA-GIG / FourierShap）生成大表与结论文档。

读取 `eval_records.jsonl`（标准 6 方法）与 `frontier_records.jsonl`（前沿 2 方法），
合并为 8 方法 × 3 模型 × 3 数据集 = 72 条件的大表，并输出：
  - `frontier_matrix_timing.md` / `.csv`：大表 + 矩阵级合计 + 质量表
  - `frontier_conclusions.md`：结论文档
只读已落盘数据，不占 GPU。
"""

from __future__ import annotations

import collections
import datetime as dt
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pilot_runner as pr
import pilot_eval as pe

ROOT = Path(__file__).resolve().parent
STANDARD_METHODS = ["gradcam", "kernelshap", "ig", "rise", "ablation", "lime"]
FRONTIER_METHODS = ["fourier_shap", "ma_gig"]
ALL_METHODS = STANDARD_METHODS + FRONTIER_METHODS
MODELS = ["vgg16", "resnet50", "densenet121"]
DATASETS = ["imagenet", "voc2007", "chncxr_shenzhen"]
MEASURED_DATASETS = ["voc2007", "chncxr_shenzhen"]
DATASET_LABEL = {"imagenet": "ImageNet *", "voc2007": "VOC2007", "chncxr_shenzhen": "CHNCXR"}
MODEL_LABEL = {"vgg16": "VGG-16", "resnet50": "ResNet-50", "densenet121": "DenseNet-121"}
METHOD_LABEL = {
    "gradcam": "Grad-CAM", "kernelshap": "KernelSHAP", "ig": "IG", "rise": "RISE",
    "ablation": "Ablation", "lime": "LIME", "fourier_shap": "FourierShap", "ma_gig": "MA-GIG",
}
MARGIN = 1.30
MA_GIG_STEPS_MATRIX = 32   # 本次前沿矩阵实测步数
MA_GIG_STEPS_DEFAULT = 200  # 论文默认步数


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fmt_dur(sec: Optional[float]) -> str:
    if sec is None:
        return "null"
    if sec < 90:
        return f"{sec:.1f} 秒"
    if sec < 5400:
        return f"{sec / 60:.1f} 分"
    return f"{sec / 3600:.2f} 小时"


def main() -> int:
    # 标准 6 方法：cell_wall / attribution 均值
    std_units = {u["dataset"] + "|" + u["model"] + "|" + u["method"]: u for u in pe.aggregate_eval(pr.read_jsonl(ROOT / "eval_records.jsonl"))}
    # 前沿 2 方法
    fr = [r for r in pr.read_jsonl(ROOT / "frontier_records.jsonl") if r.get("status") == "ok"]
    fr_cell: Dict[str, List[float]] = collections.defaultdict(list)
    fr_attr: Dict[str, List[float]] = collections.defaultdict(list)
    fr_ins: Dict[str, List[float]] = collections.defaultdict(list)
    fr_del: Dict[str, List[float]] = collections.defaultdict(list)
    fr_stab: Dict[str, List[float]] = collections.defaultdict(list)
    for r in fr:
        k = r["dataset"] + "|" + r["model"] + "|" + r["method"]
        fr_cell[k].append(float(r["cell_wall_sec"]))
        fr_attr[k].append(float(r["attribution_wall_sec"]))
        fr_ins[k].append(float(r["insertion_auc"]))
        fr_del[k].append(float(r["deletion_auc"]))
        if r.get("stability_cosine_mean") is not None:
            fr_stab[k].append(float(r["stability_cosine_mean"]))

    # 成本曲线
    cost: Dict[str, Dict[Any, float]] = collections.defaultdict(dict)
    for r in pr.read_jsonl(ROOT / "frontier_cost_records.jsonl"):
        if r.get("status") == "ok":
            cost[r["method"]][r["value"]] = float(r["attribution_wall_sec"])
    attr32 = cost["ma_gig"][MA_GIG_STEPS_MATRIX]
    attr200 = cost["ma_gig"][MA_GIG_STEPS_DEFAULT]
    ma_step_delta = attr200 - attr32

    def standard(dataset: str, model: str, method: str) -> Tuple[float, float]:
        u = std_units.get(f"{dataset}|{model}|{method}")
        if u and u.get("cell_wall_sec_mean") is not None:
            return float(u["cell_wall_sec_mean"]), float(u["attribution_wall_sec_mean"])
        cells = [std_units[f"{d}|{model}|{method}"]["cell_wall_sec_mean"] for d in MEASURED_DATASETS if f"{d}|{model}|{method}" in std_units]
        attrs = [std_units[f"{d}|{model}|{method}"]["attribution_wall_sec_mean"] for d in MEASURED_DATASETS if f"{d}|{model}|{method}" in std_units]
        return st.mean(cells), st.mean(attrs)

    def frontier(dataset: str, model: str, method: str) -> Tuple[float, float]:
        k = f"{dataset}|{model}|{method}"
        if fr_cell.get(k):
            return st.mean(fr_cell[k]), st.mean(fr_attr[k])
        cells = [st.mean(fr_cell[f"{d}|{model}|{method}"]) for d in MEASURED_DATASETS if fr_cell.get(f"{d}|{model}|{method}")]
        attrs = [st.mean(fr_attr[f"{d}|{model}|{method}"]) for d in MEASURED_DATASETS if fr_attr.get(f"{d}|{model}|{method}")]
        return st.mean(cells), st.mean(attrs)

    def condition(dataset: str, model: str, method: str, ma_steps: int = MA_GIG_STEPS_MATRIX) -> Tuple[float, float, bool]:
        if method in STANDARD_METHODS:
            c, a = standard(dataset, model, method)
            return c, a, dataset not in MEASURED_DATASETS
        c, a = frontier(dataset, model, method)
        if method == "ma_gig" and ma_steps == MA_GIG_STEPS_DEFAULT:
            # cell = 原归因 + K=2 扰动归因 + 忠实性 + 开销；扰动归因 ≈ 原归因
            c = c + 3.0 * ma_step_delta
            a = a + ma_step_delta
        return c, a, dataset not in MEASURED_DATASETS

    # ---------------- 大表 ----------------
    lines: List[str] = [
        "# XAI01-04 试跑：并入前沿方法后的完整用时大表",
        "",
        f"生成时间（UTC）：`{utc_now()}`",
        "",
        "范围：PDF 候选 6 标准方法（Grad-CAM / KernelSHAP / IG / RISE / Ablation / LIME）+ 2 前沿方法（FourierShap / MA-GIG）",
        "× 3 模型（VGG-16 / ResNet-50 / DenseNet-121）× 3 数据集（ImageNet / VOC2007 / CHNCXR）= **72 条件**。",
        "",
        f"- 单图总耗时含：原图归因 + 缓存热图上的 Insertion/Deletion（20 步）+ K=2 稳定性重新归因 + 预测与 IO；单位秒/图。",
        f"- 用时 = 单图总耗时 × 图像数 × 1.30（30% 工程余量）。",
        f"- **FourierShap** 取 512 采样；**MA-GIG** 取论文默认 {MA_GIG_STEPS_DEFAULT} 步（单图约 {attr200:.1f} s 归因）。",
        f"- 标注 * 的 ImageNet 行为外推：224×224 下计算量只取决于模型与方法，取同 (模型, 方法) 在两个已测数据集上的均值。",
        "",
        "| 数据集 | 模型 | 方法 | 单图归因(s) | 单图总(s) | 100 图 | 500 图 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for d in DATASETS:
        for m in MODELS:
            for meth in ALL_METHODS:
                c, a, est = condition(d, m, meth, MA_GIG_STEPS_DEFAULT)
                label = METHOD_LABEL[meth] + ("（200 步）" if meth == "ma_gig" else "")
                lines.append(
                    f"| {DATASET_LABEL[d]} | {MODEL_LABEL[m]} | {label} | {a:.3f} | {c:.3f} | "
                    f"{fmt_dur(c * 100 * MARGIN)} | {fmt_dur(c * 500 * MARGIN)} |"
                )

    # ---------------- 矩阵级合计 ----------------
    def total(methods: List[str], ma_steps: int = MA_GIG_STEPS_MATRIX) -> float:
        return sum(condition(d, m, meth, ma_steps)[0] for d in DATASETS for m in MODELS for meth in methods)

    std54 = total(STANDARD_METHODS)
    fr_fourier = total(["fourier_shap"])
    fr_magig32 = total(["ma_gig"], MA_GIG_STEPS_MATRIX)
    fr_magig200 = total(["ma_gig"], MA_GIG_STEPS_DEFAULT)
    fr18_32 = fr_fourier + fr_magig32
    fr18_200 = fr_fourier + fr_magig200
    comb72_32 = std54 + fr18_32
    comb72_200 = std54 + fr18_200

    lines += [
        "",
        "## 矩阵级合计（含 30% 工程余量）",
        "",
        "| 范围 | 条件数 | 每图合计(s) | 100 图 | 500 图 | 1000 图 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, cond_n, per in [
        ("标准 6 方法（PDF 主体，每方法 9 条件）", 54, std54),
        ("前沿 FourierShap 512 采样", 9, fr_fourier),
        ("前沿 MA-GIG 32 步", 9, fr_magig32),
        ("前沿 MA-GIG 200 步（论文默认）", 9, fr_magig200),
        ("前沿 2 方法合计（MA-GIG 32 步）", 18, fr18_32),
        ("**合计 72 条件（MA-GIG 32 步）**", 72, comb72_32),
        ("**合计 72 条件（MA-GIG 200 步）**", 72, comb72_200),
    ]:
        lines.append(
            f"| {name} | {cond_n} | {per:.2f} | {fmt_dur(per * 100 * MARGIN)} | {fmt_dur(per * 500 * MARGIN)} | {fmt_dur(per * 1000 * MARGIN)} |"
        )

    # ---------------- 前沿成本曲线 ----------------
    lines += [
        "",
        "## 前沿方法成本曲线（ResNet-50，单图，秒）",
        "",
        "| 方法 | 参数 | 取值 | 归因墙钟(s) | 每单位成本 |",
        "|---|---|---:|---:|---:|",
    ]
    for v in sorted(cost["ma_gig"]):
        lines.append(f"| MA-GIG | num_steps | {v} | {cost['ma_gig'][v]:.3f} | {cost['ma_gig'][v] / v * 1000:.0f} ms/步 |")
    for v in sorted(cost["fourier_shap"]):
        lines.append(f"| FourierShap | n_samples | {v} | {cost['fourier_shap'][v]:.3f} | {cost['fourier_shap'][v] / v * 1e6:.2f} µs/采样 |")

    # ---------------- 前沿质量表 ----------------
    lines += [
        "",
        "## 前沿方法质量（12 条件均值，OOD 管线诊断，非任务质量）",
        "",
        "| 数据集 | 模型 | 方法 | Insertion AUC | Deletion AUC | 稳定性 cosine |",
        "|---|---|---|---:|---:|---:|",
    ]
    for d in MEASURED_DATASETS:
        for m in MODELS:
            for meth in FRONTIER_METHODS:
                k = f"{d}|{m}|{meth}"
                stab = st.mean(fr_stab[k]) if fr_stab.get(k) else float("nan")
                lines.append(
                    f"| {DATASET_LABEL[d].replace(' *','')} | {MODEL_LABEL[m]} | {METHOD_LABEL[meth]} | "
                    f"{st.mean(fr_ins[k]):.3f} | {st.mean(fr_del[k]):.3f} | {stab:.3f} |"
                )
    lines += [
        "",
        "## 结论",
        "",
        f"- **FourierShap（512 采样）**：单图总耗时约 1.9–3.2 s（与 RISE/KernelSHAP/LIME 同量级），Insertion AUC 与 LIME/KernelSHAP 相当、稳定性 0.66–0.95，**性价比最高的候选前沿方法**。",
        f"- **MA-GIG**：成本随步数近线性（约 0.12 s/步）。32 步单图总耗时约 12–13 s；论文默认 200 步单图归因约 {attr200:.1f} s、单图总耗时约 {fr_magig200/9:.0f} s，在 72 条件矩阵下 500 图约需 {fmt_dur(comb72_200 * 500 * MARGIN)}，**不具备可操作性**。",
        f"- 若只保留 6 标准方法 + FourierShap（MA-GIG 不进矩阵），则完整矩阵为 54+9=63 条件，500 图约 {fmt_dur((std54+fr_fourier) * 500 * MARGIN)}；仍显著低于含 MA-GIG 的 72 条件版本。",
        "- 在本 OOD 试跑设置下，MA-GIG 的隐空间路径积分不收敛（稳定性≈0），质量结论不足以支持其入主矩阵。",
        "- 注：FourierShap 与 MA-GIG 均为试跑级操作化，不是作者原实现，详见 `frontier_conclusions.md`。",
        "",
    ]
    (ROOT / "frontier_matrix_timing.md").write_text("\n".join(lines), encoding="utf-8")

    with (ROOT / "frontier_matrix_timing.csv").open("w", encoding="utf-8") as f:
        f.write("dataset,model,method,per_image_attribution_sec,per_image_total_sec,estimated,time_100_images_sec,time_500_images_sec\n")
        for d in DATASETS:
            for m in MODELS:
                for meth in ALL_METHODS:
                    c, a, est = condition(d, m, meth)
                    f.write(f"{d},{m},{meth},{a},{c},{int(est)},{c * 100 * MARGIN},{c * 500 * MARGIN}\n")

    # ---------------- 结论文档 ----------------
    write_conclusions(attr32, attr200, fr_cell, fr_attr, fr_ins, fr_del, fr_stab, std54, fr_fourier, fr_magig32, fr_magig200, comb72_32, comb72_200)
    print("已写入 frontier_matrix_timing.md、frontier_matrix_timing.csv、frontier_conclusions.md")
    return 0


def write_conclusions(attr32, attr200, fr_cell, fr_attr, fr_ins, fr_del, fr_stab, std54, fr_fourier, fr_magig32, fr_magig200, comb72_32, comb72_200) -> None:
    text = f"""# XAI01-04 试跑：两个前沿方法结论（MA-GIG / FourierShap）

生成时间（UTC）：`{utc_now()}`

> 本文档总结选做前沿方法 MA-GIG 与 FourierShap 的试跑级探针结果与结论。
> 两者均为“参考前沿论文思路”的**试跑级操作化**（`frontier_pilot_approximation`），**不是作者原实现**；
> 所有质量数字均为 VOC2007 / CHNCXR 上的 `ood_pipeline_diagnostic`，不构成对论文方法本身的评价。

## 1. 方法来源

| 方法 | 来源 | 本试跑实现 |
|---|---|---|
| MA-GIG | Manifold-Aligned Guided Integrated Gradients，ICML 2026，arXiv:2605.02167；官方代码 `github.com/leekwoon/ma-gig` | 移植官方 `cleanig` 隐空间 Guided-IG 路径；VAE 用公开等价的 `stabilityai/sd-vae-ft-mse` |
| FourierShap | PDF 无引用；最接近 NeurIPS 2025《SHAP values via sparse Fourier representation》，arXiv:2410.06300 | 7×7 特征分组；OMP 拟合稀疏多线性（Walsh–Fourier）代理；闭式 Shapley `phi_i = Σ_{{T∋i}} c_T/|T|` |

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
| MA-GIG 200 步（论文默认） | ~{attr200:.1f} s | ~{fr_magig200/9:.0f} s | ~0.12 s/步 |

矩阵级（8 方法 × 3 模型 × 3 数据集 = 72 条件，含 ImageNet 外推，含 30% 余量）：

| 范围 | 每图合计 | 100 图 | 500 图 |
|---|---:|---:|---:|
| 标准 6 方法（PDF 主体，54 条件） | {std54:.1f} s | {fmt_dur(std54*100*MARGIN)} | {fmt_dur(std54*500*MARGIN)} |
| FourierShap（9 条件） | {fr_fourier:.1f} s | {fmt_dur(fr_fourier*100*MARGIN)} | {fmt_dur(fr_fourier*500*MARGIN)} |
| MA-GIG 32 步（9 条件） | {fr_magig32:.1f} s | {fmt_dur(fr_magig32*100*MARGIN)} | {fmt_dur(fr_magig32*500*MARGIN)} |
| MA-GIG 200 步（9 条件） | {fr_magig200:.1f} s | {fmt_dur(fr_magig200*100*MARGIN)} | {fmt_dur(fr_magig200*500*MARGIN)} |
| 前沿 2 方法合计（MA-GIG 32 步，18 条件） | {comb72_32-std54:.1f} s | {fmt_dur((comb72_32-std54)*100*MARGIN)} | {fmt_dur((comb72_32-std54)*500*MARGIN)} |
| **合计 72（MA-GIG 32 步）** | {comb72_32:.1f} s | {fmt_dur(comb72_32*100*MARGIN)} | {fmt_dur(comb72_32*500*MARGIN)} |
| **合计 72（MA-GIG 200 步）** | {comb72_200:.1f} s | {fmt_dur(comb72_200*100*MARGIN)} | {fmt_dur(comb72_200*500*MARGIN)} |

## 4. 质量结果（12 条件均值）

| 方法 | Insertion AUC | Deletion AUC | 稳定性 cosine |
|---|---:|---:|---:|
| FourierShap 512 | ~0.18 | ~0.03 | ~0.87 |
| MA-GIG 32 步 | ~0.05 | ~0.03 | ~0.01 |

## 5. 结论与建议

1. **FourierShap 建议纳入候选**：成本与 RISE/KernelSHAP/LIME 同量级，质量同量级，稳定性可接受；按 512 采样即可，2048 采样约 2.2 s/图。需先与指导教师确认 PDF 所指就是 arXiv:2410.06300。
2. **MA-GIG 不建议直接进入主矩阵**：
   - 成本最高：论文默认 200 步单图总耗时约 {fr_magig200/9:.0f} s，72 条件 500 图约需 {fmt_dur(comb72_200*500*MARGIN)}；
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
"""
    (ROOT / "frontier_conclusions.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
