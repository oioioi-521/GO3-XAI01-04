#!/usr/bin/env python3
"""将 XAI01-04 试跑的 JSONL 记录汇总为 PLAN.md 要求的交付物。

生成 ``extrapolation.md`` / ``extrapolation.json``（逐条件实测单图成本，
以及 100 / 500 / 1000 图外推，含推荐的“4 方法 × 2 模型 × 2 数据集”主体矩阵）
和 ``RESULT_INDEX.md``（交付物清单与 P0–P4 状态）。只读取已落盘数据，不占用 GPU。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pilot_eval as pe
import pilot_runner as pr

ROOT = Path(__file__).resolve().parent
DATASET_ORDER = pe.DATASET_ORDER
MODEL_ORDER = pr.MODEL_ORDER
METHOD_ORDER = pr.METHOD_ORDER
MAIN_METHODS = ["gradcam", "ig", "kernelshap", "rise"]
MAIN_MODELS = ["resnet50", "vgg16"]
TARGET_SIZES = [100, 500, 1000]
MARGIN = 1.30


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_eval_units() -> List[Dict[str, Any]]:
    rows = pr.read_jsonl(ROOT / "eval_records.jsonl")
    return pe.aggregate_eval(rows)


def fmt_seconds(sec: Optional[float]) -> str:
    if sec is None:
        return "null"
    if sec < 90:
        return f"{sec:.1f} 秒"
    if sec < 5400:
        return f"{sec / 60:.1f} 分钟"
    return f"{sec / 3600:.2f} 小时"


def main() -> int:
    units = read_eval_units()
    by_key = {(u["dataset"], u["model"], u["method"]): u for u in units}

    lines: List[str] = [
        "# XAI01-04 试跑成本外推",
        "",
        f"生成时间（UTC）：`{utc_now()}`",
        "",
        "所有成本均为 RTX 2080 Ti 试跑主机上的实测单图墙钟秒数，包含原始归因、对缓存热图的一次",
        "Insertion 与一次 Deletion、K=2 稳定性重新归因、预测与 IO。外推将实测单图均值乘以 N，并叠加",
        "30% 工程余量；该余量不是统计置信区间。",
        "",
        "## 逐条件实测成本与 100 / 500 / 1000 图外推",
        "",
        "| 数据集 | 模型 | 方法 | 单图总耗时(s) | 仅归因(s) | 100 图 | 500 图 | 1000 图 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for d in DATASET_ORDER:
        for m in MODEL_ORDER:
            for meth in METHOD_ORDER:
                u = by_key.get((d, m, meth))
                if not u:
                    continue
                per = u.get("cell_wall_sec_mean")
                attr = u.get("attribution_wall_sec_mean")
                cells = [
                    fmt_seconds(per * n * MARGIN) if per is not None else "null" for n in TARGET_SIZES
                ]
                lines.append(
                    f"| {d} | {m} | {meth} | {per if per is None else f'{per:.4f}'} | "
                    f"{attr if attr is None else f'{attr:.4f}'} | {cells[0]} | {cells[1]} | {cells[2]} |"
                )

    lines += [
        "",
        "## 推荐主体矩阵（PLAN 第 6 节）",
        "",
        "方法 Grad-CAM、IG、KernelSHAP、RISE；模型 ResNet-50、VGG-16；两个数据集，共 16 个条件。",
        "",
        "| 每数据集图像数 | 实测合计（16 条件） | +30% 余量 |",
        "|---:|---:|---:|",
    ]
    for n in TARGET_SIZES:
        total = 0.0
        complete = True
        for d in DATASET_ORDER:
            for m in MAIN_MODELS:
                for meth in MAIN_METHODS:
                    u = by_key.get((d, m, meth))
                    if not u or u.get("cell_wall_sec_mean") is None:
                        complete = False
                        continue
                    total += u["cell_wall_sec_mean"] * n
        if complete:
            lines.append(f"| {n} | {fmt_seconds(total)} | {fmt_seconds(total * MARGIN)} |")
        else:
            lines.append(f"| {n} | 数据不完整 | 数据不完整 |")

    lines += [
        "",
        "## 说明",
        "",
        "- “仅归因”是同步的单图归因耗时；回答部署解释成本问题时使用该口径。",
        "- “单图总耗时”是实验成本（归因 + 忠实性 + 稳定性）。",
        "- 模型加载成本单列：三个 ImageNet 检查点已缓存时，各约 0.3–1.0 秒。",
        "- 这些均为域外（OOD）管线诊断，估计的是成本，不是任务质量。",
        "",
    ]
    (ROOT / "extrapolation.md").write_text("\n".join(lines), encoding="utf-8")

    pr.write_json(
        ROOT / "extrapolation.json",
        {
            "generated_utc": utc_now(),
            "margin": MARGIN,
            "target_sizes": TARGET_SIZES,
            "units": units,
        },
    )

    cal = pr.read_jsonl(ROOT / "calibration_records.jsonl")
    cal_ok = [r for r in cal if r.get("record_type") == "calibration_replicate" and r.get("status") == "ok"]
    eval_rows = pr.read_jsonl(ROOT / "eval_records.jsonl")
    eval_ok = [r for r in eval_rows if r.get("record_type") == "evaluation_cell" and r.get("status") == "ok"]
    cost = pr.read_jsonl(ROOT / "cost_curve_records.jsonl")
    recheck = pr.read_jsonl(ROOT / "recheck_records.jsonl")
    frontier = pr.read_jsonl(ROOT / "frontier_records.jsonl")
    frontier_ok = [r for r in frontier if r.get("record_type") == "frontier_cell" and r.get("status") == "ok"]

    index = [
        "# XAI01-04 试跑结果索引",
        "",
        f"生成时间（UTC）：`{utc_now()}`",
        "",
        "主机：RTX 2080 Ti 11 GiB（桌面显示卡），i7-8700K，46 GiB 内存。PyTorch 2.8.0 / torchvision 0.23.0 / CUDA runtime 12.8。",
        "",
        "## 阶段状态",
        "",
        "| 阶段 | 状态 | 记录数 | 主要交付物 |",
        "|---|---|---:|---|",
        "| P0 环境与首通 | 完成 | - | `environment_snapshot.json`、`smoke_records.jsonl`、`smoke_report.md`、`rise_smoke_*` |",
        f"| P1 6×3 合成图校准 | 完成 | {len(cal_ok)} 成功 | `calibration_records.jsonl`、`calibration_summary.json`、`calibration_report.md` |",
        f"| P2 小规模端到端评价 | 完成 | {len(eval_ok)} 成功 / {len(eval_rows)} 总计 | `eval_records.jsonl`、`eval_summary.json`、`eval_report.md`、`eval_per_image.csv`、`eval_units.csv` |",
        f"| P3 采样成本曲线 | 完成 | {len(cost)} | `cost_curve_records.jsonl`、`cost_curve_summary.json`、`cost_curve_report.md` |",
        f"| P4 随机方法种子复核 | 完成 | {len(recheck)} | `recheck_records.jsonl`、`recheck_summary.json`、`recheck_report.md` |",
        f"| P5 前沿方法探针（MA-GIG / FourierShap） | 完成 | {len(frontier_ok)} 成功 / {len(frontier)} | `frontier_records.jsonl`、`frontier_summary.json`、`frontier_report.md`、`frontier_units.csv`、`frontier_cost_report.md` |",
        "| 成本外推 | 完成 | - | `extrapolation.md`、`extrapolation.json` |",
        "",
        "## 协议摘要",
        "",
        "- P2 规模：2 数据集 × 3 模型 × 6 方法 × 16 图 = 576 个单元，全部 `status=ok`。",
        "- 归因目标取模型在原图上的 ImageNet top-1 预测，逐单元记录。",
        "- 忠实性：20 步 / 21 个节点梯形积分 AUC，Insertion 与 Deletion，基于缓存热图，排序按带符号求和热图的绝对值降序。",
        "- 稳定性：K=2 高斯噪声，原始 [0,1] 空间 sigma=0.01，同一图像共享扰动，cosine 在带符号热图上计算。",
        "- P4 将 Monte-Carlo 种子方差（独立归因种子）与输入扰动稳定性分开测量。",
        "- 所有 VOC2007 / CHNCXR 质量行均标记 `ood_pipeline_diagnostic`：ImageNet 分类头属域外，因此此处没有任何任务质量或医学结论。",
        "",
        "## 试跑关键观察（描述性，不具统计显著性）",
        "",
        "- 效率：Grad-CAM 与 Ablation 最便宜（归因约 0.01–0.11 秒/图）；KernelSHAP、LIME、RISE 约 0.6–1.2 秒/图；IG 约 0.09–0.13 秒/图。",
        "- 忠实性（本协议下 Insertion AUC 越高越好）：VOC2007 上 LIME 较高、IG 最低；CHNCXR 因分类头域外，数值明显偏低。",
        "- 稳定性（输入扰动 cosine）：RISE 接近 1.0（随机掩码对小幅输入噪声不敏感），IG 最低（约 0.35–0.63，梯度噪声大），其余方法居中。",
        "- Monte-Carlo 种子方差（P4）：KernelSHAP 最不稳定（跨种子 cosine 约 0.18–0.73），LIME 中等（约 0.87–0.98），RISE 很稳定（约 0.999）。",
        "- 成本曲线（P3）：成本随采样数/掩码数近似线性增长；内部 batch 从 16 提到 32 有适度加速；2048 采样约为 512 设置的 3–4 倍。",
        "- 前沿方法（P5，仅试跑级操作化）：MA-GIG 约 0.12 s/步，论文默认 200 步约 24.7 s/图，本 OOD 设置下路径积分不收敛（32 步 Insertion AUC≈0.05、稳定性≈0.01），不建议直接进主矩阵；FourierShap 512 采样约 0.6–1.0 s/图，Insertion AUC 与 LIME/KernelSHAP 同量级、稳定性≈0.66–0.95，可作为候选。",
        "",
        "## 局限",
        "",
        "- 每数据集 16 张仅为计时子集，不是评估集。",
        "- 不做 ANOVA / Pareto / 显著性结论；P2 只验证测量管线，并保存后续分析所需的逐图数据。",
        "- CHNCXR 仅将 ImageNet 头作为域外管线探针，未做结核分类。",
        "",
    ]
    (ROOT / "RESULT_INDEX.md").write_text("\n".join(index), encoding="utf-8")
    print("已写入 extrapolation.md、extrapolation.json、RESULT_INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
