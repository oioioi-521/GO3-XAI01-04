#!/usr/bin/env python3
"""按 project-pool.pdf 的完整候选矩阵（6 方法 × 3 模型 × 3 数据集）输出用时表。

VOC2007 与 CHNCXR 的逐条件成本来自本次试跑实测（eval_records.jsonl）。
ImageNet 未在试跑中提供真实图像，但其前向/反向计算量与数据集无关（同尺寸
224×224、同模型、同方法），故用同 (模型, 方法) 在两个已测数据集上的均值外推，
并在表中以 * 标注。只读已落盘数据，不占用 GPU。
"""

from __future__ import annotations

import collections
import datetime as dt
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

import pilot_runner as pr

ROOT = Path(__file__).resolve().parent
METHODS = ["gradcam", "ig", "kernelshap", "rise", "lime", "ablation"]
MODELS = ["vgg16", "resnet50", "densenet121"]
DATASETS_MEASURED = ["voc2007", "chncxr_shenzhen"]
DATASETS_ALL = ["imagenet", "voc2007", "chncxr_shenzhen"]
DATASET_LABEL = {"imagenet": "ImageNet", "voc2007": "VOC2007", "chncxr_shenzhen": "CHNCXR-Shenzhen"}
MODEL_LABEL = {"vgg16": "VGG-16", "resnet50": "ResNet-50", "densenet121": "DenseNet-121"}
METHOD_LABEL = {"gradcam": "Grad-CAM", "ig": "IG", "kernelshap": "KernelSHAP", "rise": "RISE", "lime": "LIME", "ablation": "Ablation"}
MARGIN = 1.30


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def fmt_dur(sec: float) -> str:
    if sec < 90:
        return f"{sec:.1f} 秒"
    if sec < 5400:
        return f"{sec / 60:.1f} 分"
    return f"{sec / 3600:.2f} 小时"


def main() -> int:
    rows = [r for r in pr.read_jsonl(ROOT / "eval_records.jsonl") if r.get("status") == "ok"]
    cell: Dict[Tuple[str, str, str], List[float]] = collections.defaultdict(list)
    attr: Dict[Tuple[str, str, str], List[float]] = collections.defaultdict(list)
    for r in rows:
        k = (r["dataset"], r["model"], r["method"])
        cell[k].append(float(r["cell_wall_sec"]))
        attr[k].append(float(r["attribution_wall_sec"]))
    cmean = {k: st.mean(v) for k, v in cell.items()}
    amean = {k: st.mean(v) for k, v in attr.items()}

    def per_condition(dataset: str, model: str, method: str) -> Tuple[float, float, bool]:
        if dataset in DATASETS_MEASURED and (dataset, model, method) in cmean:
            return cmean[(dataset, model, method)], amean[(dataset, model, method)], False
        vals = [cmean[(d, model, method)] for d in DATASETS_MEASURED if (d, model, method) in cmean]
        avals = [amean[(d, model, method)] for d in DATASETS_MEASURED if (d, model, method) in amean]
        return st.mean(vals), st.mean(avals), True

    lines: List[str] = [
        "# XAI01-04 完整候选矩阵用时表",
        "",
        f"生成时间（UTC）：`{utc_now()}`",
        "",
        "矩阵来源：`project-pool.pdf` 第 63–65 页 XAI01-04 的建议实验空间，取全部候选：",
        "方法 6（Grad-CAM / IG / KernelSHAP / RISE / LIME / Ablation）× 模型 3（VGG-16 / ResNet-50 / DenseNet-121）× 数据集 3（ImageNet / VOC / CHNCXR）= **54 个条件**。",
        "",
        "单图成本为实测值，包含原始归因 + 对缓存热图的一次 Insertion/Deletion（20 步）+ K=2 稳定性重新归因（sigma=0.01）+ 预测与 IO；单位：秒/图。",
        "用时 = 单图成本 × 图像数 × 1.30（30% 工程余量）。",
        "",
        "> 说明：VOC2007 与 CHNCXR 为本次实测；**ImageNet 用 * 标注，为同 (模型, 方法) 在两个已测数据集上的均值外推**——",
        "> 因为 224×224 输入下前向/反向计算量只取决于模型与方法，与图像来自哪个数据集无关。",
        "",
        "## 逐条件用时（N=100 / N=500）",
        "",
        "| 数据集 | 模型 | 方法 | 单图总(s) | 仅归因(s) | 100 图用时 | 500 图用时 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]

    for d in DATASETS_ALL:
        for m in MODELS:
            for meth in METHODS:
                per, only, est = per_condition(d, m, meth)
                mark = " *" if est else ""
                lines.append(
                    f"| {DATASET_LABEL[d]}{mark} | {MODEL_LABEL[m]} | {METHOD_LABEL[meth]} | {per:.3f} | {only:.3f} | "
                    f"{fmt_dur(per * 100 * MARGIN)} | {fmt_dur(per * 500 * MARGIN)} |"
                )

    def matrix_total(datasets: List[str], n: int) -> float:
        return sum(per_condition(d, m, meth)[0] for d in datasets for m in MODELS for meth in METHODS) * n

    lines += [
        "",
        "## 矩阵级合计",
        "",
        "| 范围 | 条件数 | 每图合计(s) | 100 图 | 500 图 | 1000 图 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    scopes = [
        ("本次实测矩阵（VOC+CHNCXR）", DATASETS_MEASURED),
        ("PDF 完整矩阵（ImageNet+VOC+CHNCXR）", DATASETS_ALL),
    ]
    for name, ds in scopes:
        cond = sum(per_condition(d, m, meth)[0] for d in ds for m in MODELS for meth in METHODS)
        lines.append(
            f"| {name} | {len(ds) * 3 * 6} | {cond:.2f} | {fmt_dur(cond * 100 * MARGIN)} | "
            f"{fmt_dur(cond * 500 * MARGIN)} | {fmt_dur(cond * 1000 * MARGIN)} |"
        )

    main_cond = sum(per_condition(d, m, meth)[0] for d in DATASETS_ALL for m in ["vgg16", "resnet50"] for meth in ["gradcam", "ig", "kernelshap", "rise"])
    lines += [
        "",
        "## 推荐主矩阵（PLAN 第 6 节：4 方法 × 2 模型 × 3 数据集 = 24 条件）",
        "",
        "方法 Grad-CAM / IG / KernelSHAP / RISE，模型 VGG-16 / ResNet-50，数据集 ImageNet / VOC / CHNCXR。",
        "",
        "| 图像数（每条件） | 用时（含 30% 余量） |",
        "|---:|---:|",
    ]
    for n in (100, 500, 1000):
        lines.append(f"| {n} | {fmt_dur(main_cond * n * MARGIN)} |")

    lines += [
        "",
        "## 结论",
        "",
        f"- PDF 完整 54 条件：每图合计 {matrix_total(DATASETS_ALL, 1):.1f} 秒；100 图约 {fmt_dur(matrix_total(DATASETS_ALL, 100) * MARGIN)}，500 图约 {fmt_dur(matrix_total(DATASETS_ALL, 500) * MARGIN)}。",
        f"- 实测 36 条件：100 图约 {fmt_dur(matrix_total(DATASETS_MEASURED, 100) * MARGIN)}，500 图约 {fmt_dur(matrix_total(DATASETS_MEASURED, 500) * MARGIN)}。",
        f"- 推荐主矩阵 24 条件：100 图约 {fmt_dur(main_cond * 100 * MARGIN)}，500 图约 {fmt_dur(main_cond * 500 * MARGIN)}。",
        "- 昂贵方法（RISE/KernelSHAP/LIME）占矩阵绝大部分时间；K 是比图像数更陡的杠杆（K 2→10 约为 3.5 倍）。",
        "- 全 54 条件在 500 图上约半天 GPU 墙钟，属于一次夜间批量可完成的范围。",
        "",
    ]
    (ROOT / "matrix_timing_full.md").write_text("\n".join(lines), encoding="utf-8")

    with (ROOT / "matrix_timing_full.csv").open("w", encoding="utf-8") as f:
        f.write("dataset,model,method,per_image_total_sec,per_image_attribution_sec,estimated,time_100_images_sec,time_500_images_sec\n")
        for d in DATASETS_ALL:
            for m in MODELS:
                for meth in METHODS:
                    per, only, est = per_condition(d, m, meth)
                    f.write(f"{d},{m},{meth},{per},{only},{int(est)},{per * 100 * MARGIN},{per * 500 * MARGIN}\n")

    print("已写入 matrix_timing_full.md、matrix_timing_full.csv")
    for d in DATASETS_ALL:
        for m in MODELS:
            for meth in METHODS:
                per, only, est = per_condition(d, m, meth)
                print(f"{DATASET_LABEL[d]:18} {MODEL_LABEL[m]:12} {METHOD_LABEL[meth]:10} {per:6.3f}s {'(估)' if est else '    '} 100图={fmt_dur(per*100*MARGIN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
