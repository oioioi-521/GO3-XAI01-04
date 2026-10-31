"""Render paired DCU statistics without loading raw data or running models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .validate_dcu_team_handoff import DATASETS, METHODS, METRICS, MODELS

LABEL = dict(ig="IG", gradcam="Grad-CAM", occlusion="Occlusion", rise="RISE",
             kernelshap="KernelSHAP", lime="LIME-8000")
MODEL_LABEL = dict(resnet50="ResNet50", densenet121="DenseNet121", vgg16="VGG16")
DATA_LABEL = dict(imagenet="ImageNet", voc="VOC (single target)")
COLORS = ["#1971c2", "#e67700", "#2b8a3e", "#9c36b5", "#c92a2a", "#087f8c"]


def effect_grid(rows, dataset, model, metric):
    grid, significant = np.zeros((6, 6)), np.zeros((6, 6), dtype=bool)
    selected = [r for r in rows if r["cohort"] == "model_correct"
                and (r["dataset"], r["model"], r["metric"]) == (dataset, model, metric)]
    if len(selected) != 15:
        raise ValueError("incomplete 15-pair quality panel")
    seen = set()
    for r in selected:
        a, b = METHODS.index(r["a"]), METHODS.index(r["b"])
        if a == b or tuple(sorted((a, b))) in seen:
            raise ValueError("duplicate/self pair")
        seen.add(tuple(sorted((a, b))))
        grid[a, b], grid[b, a] = r["rank_biserial_a"], -r["rank_biserial_a"]
        significant[a, b] = significant[b, a] = r["p_holm"] < .05
    return grid, significant


def render(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = []

    def save(fig, name):
        path = output / name
        fig.savefig(path, dpi=170, bbox_inches="tight", metadata={"Software": "D DCU statistics"})
        plt.close(fig)
        files.append(path)

    summaries = {(r["dataset"], r["model"], r["method"], r["metric"]): r
                 for r in data["summaries"] if r["cohort"] == "model_correct"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    titles = ["Raw MoRF AUC (lower better)", "Noise Spearman (higher better)",
              "Recorded wrapper time, ms (log; not a speedup)"]
    for i, dataset in enumerate(DATASETS):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            for k, model in enumerate(MODELS):
                rows = [summaries[dataset, model, m, metric] for m in METHODS]
                means = np.array([r["mean"] for r in rows])
                limits = np.array([r["mean_ci95"] for r in rows])
                ax.errorbar(means, np.arange(6) + (k - 1) * .2,
                            xerr=np.maximum(0, np.array([means - limits[:, 0], limits[:, 1] - means])),
                            fmt="o", ms=4, capsize=2, color=COLORS[k], label=MODEL_LABEL[model])
            ax.set_yticks(range(6), [LABEL[m] for m in METHODS])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.2)
            ax.set_title(DATA_LABEL[dataset] + "\n" + titles[j], fontsize=11)
            if j == 2:
                ax.set_xscale("log")
            if j == 0:
                ax.set_xlim(left=0)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.5, .95), frameon=False)
    fig.suptitle("Six methods: model-correct images, class-stratified 95% mean CIs", y=1.01, fontsize=16)
    fig.text(.5, .005, "ImageNet n=408 / 371 / 363; VOC n=387 / 388 / 400 (RN50 / DN121 / VGG16). "
             "Marginal CIs; fixed budgets; two shared-load rounds.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, .91))
    save(fig, "six_method_means.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, dataset in enumerate(DATASETS):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            rows = [r for r in data["pareto"] if r["cohort"] == "model_correct"
                    and (r["dataset"], r["model"]) == (dataset, model)]
            for r in rows:
                k = METHODS.index(r["method"])
                ax.scatter(r["faithfulness"], r["stability"], s=85,
                           color=COLORS[k], marker="o" if r["quality_front"] else "x", zorder=3)
                if r["recorded_time_3d_front"]:
                    ax.scatter(r["faithfulness"], r["stability"], s=175,
                               facecolors="none", edgecolors=COLORS[k], linewidths=1, zorder=2)
            front = sorted([r for r in rows if r["quality_front"]], key=lambda r: r["faithfulness"])
            ax.plot([r["faithfulness"] for r in front], [r["stability"] for r in front],
                    color="#555555", lw=1, alpha=.6, zorder=1)
            ax.set_title(f"{DATA_LABEL[dataset]} / {MODEL_LABEL[model]} (n={rows[0]['n']})")
            ax.set_xlabel("Raw MoRF AUC (lower better)")
            ax.set_ylabel("Noise Spearman (higher better)")
            ax.grid(alpha=.15)
            ax.margins(.12)
    legend = [Line2D([0], [0], color=c, marker="o", ls="", label=LABEL[m]) for m, c in zip(METHODS, COLORS)]
    fig.legend(handles=legend, loc="upper center", ncol=6, bbox_to_anchor=(.5, .955), frameon=False)
    fig.suptitle("Quality trade-offs within each dataset/model (model-correct cohort)", y=1.01, fontsize=16)
    fig.text(.5, .01, "Filled dot = quality frontier; x = quality-dominated; outer ring = 3D frontier with recorded time.\n"
             "Frontiers use means, not significance tests; timing is two-round/shared-load and not causal hardware efficiency.", ha="center")
    fig.tight_layout(rect=(0, .055, 1, .915))
    save(fig, "quality_pareto.png")

    fig, axes = plt.subplots(4, 3, figsize=(16, 18))
    labels = [LABEL[m] for m in METHODS]
    for i, (dataset, metric) in enumerate((d, m) for d in DATASETS for m in METRICS[:2]):
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            grid, sig = effect_grid(data["method_pairs"], dataset, model, metric)
            im = ax.imshow(grid, cmap="RdBu", vmin=-1, vmax=1)
            for a in range(6):
                for b in range(6):
                    text = "-" if a == b else f"{grid[a, b]:.2f}" + ("*" if sig[a, b] else "")
                    ax.text(b, a, text, ha="center", va="center", fontsize=8,
                            color="white" if abs(grid[a, b]) > .65 else "black")
            ax.set_xticks(range(6), labels, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(6), labels, fontsize=8)
            ax.set_title(f"{DATA_LABEL[dataset]} / {MODEL_LABEL[model]}\n"
                         + ("Faithfulness" if metric == METRICS[0] else "Stability"), fontsize=11)
    fig.suptitle("Paired rank-biserial effect: positive favors row method", y=.995, fontsize=16)
    fig.text(.5, .005, "* Two-sided Wilcoxon Holm p < .05; one 180-comparison quality family across both metrics and six strata.\n"
             "Model-correct cohort. Rank effects are not mean differences; see report for the DenseNet121/ImageNet exception.", ha="center")
    fig.tight_layout(rect=(0, .045, .95, .98))
    cax = fig.add_axes((.96, .25, .012, .5))
    fig.colorbar(im, cax=cax, label="Rank-biserial (row advantage)")
    save(fig, "paired_quality_effects.png")

    grid, sig = np.zeros((6, 6)), np.zeros((6, 6), dtype=bool)
    strata = [(d, m) for d in DATASETS for m in MODELS]
    for i, method in enumerate(METHODS):
        for j, (dataset, model) in enumerate(strata):
            r = next(r for r in data["correlations"] if r["cohort"] == "model_correct"
                     and (r["dataset"], r["model"], r["method"]) == (dataset, model, method)
                     and (r["metric_a"], r["metric_b"]) == METRICS[:2])
            if r["status"] != "ok":
                grid[i, j] = np.nan
            else:
                grid[i, j] = r["spearman_rho"]
                sig[i, j] = r["spearman_p_holm"] < .05
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, "NA" if np.isnan(grid[i, j]) else f"{grid[i, j]:.3f}" + ("*" if sig[i, j] else ""),
                    ha="center", va="center", color="black", fontsize=11)
    ax.set_xticks(range(6), [("IN" if d == "imagenet" else "VOC") + "\n" + MODEL_LABEL[m] for d, m in strata])
    ax.set_yticks(range(6), labels)
    ax.set_title("Within-method AUC vs noise stability: exploratory Spearman correlation", pad=18)
    fig.colorbar(im, ax=ax, label="Spearman rho (AUC is lower-better)")
    fig.text(.5, .005, "Model-correct images. * Holm p < .05 across all 216 Pearson/Spearman endpoints in this cohort.\n"
             "Positive rho means larger AUC (worse faithfulness) accompanies higher stability; not a causal effect or pooled method ranking.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    save(fig, "faithfulness_stability_correlation.png")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/dcu_20261015/statistics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/dcu_20261015"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    for path in render(data, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
