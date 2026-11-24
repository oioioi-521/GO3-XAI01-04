"""Render the formal IG/Grad-CAM faithfulness and efficiency summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ("ig", "gradcam")
MODELS = ("vgg16", "resnet50", "densenet121")
DATASETS = ("imagenet", "voc")
METRICS = ("faithfulness_morf_auc_raw", "efficiency_time_ms")
METHOD_LABELS = {"ig": "Integrated Gradients", "gradcam": "Grad-CAM"}
MODEL_LABELS = {
    "vgg16": "VGG-16",
    "resnet50": "ResNet-50",
    "densenet121": "DenseNet-121",
}
COLORS = {"ig": "#2563EB", "gradcam": "#F59E0B"}


def _validated_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[
        frame["method"].isin(METHODS)
        & frame["model"].isin(MODELS)
        & frame["dataset"].isin(DATASETS)
        & frame["metric"].isin(METRICS)
    ].copy()
    expected = {
        (method, model, dataset, metric)
        for method in METHODS
        for model in MODELS
        for dataset in DATASETS
        for metric in METRICS
    }
    actual = set(
        selected[["method", "model", "dataset", "metric"]]
        .itertuples(index=False, name=None)
    )
    if actual != expected or len(selected) != len(expected):
        raise ValueError("units.csv does not contain exactly the 24 formal A summaries")
    if set(selected["n"]) != {460}:
        raise ValueError("every formal unit must contain exactly 460 images")
    return selected.sort_values(["dataset", "metric", "model", "method"])


def _draw_panel(ax, frame: pd.DataFrame, dataset: str, metric: str) -> None:
    y = np.arange(len(MODELS), dtype=float)
    offsets = {"ig": -0.13, "gradcam": 0.13}
    for method in METHODS:
        subset = frame[
            (frame["dataset"] == dataset)
            & (frame["metric"] == metric)
            & (frame["method"] == method)
        ].set_index("model").loc[list(MODELS)]
        means = subset["mean"].to_numpy(float)
        stds = subset["std"].to_numpy(float)
        lower = np.minimum(stds, means)
        ax.errorbar(
            means,
            y + offsets[method],
            xerr=np.vstack([lower, stds]),
            fmt="o",
            markersize=6,
            capsize=3,
            linewidth=1.3,
            color=COLORS[method],
            label=METHOD_LABELS[method],
            zorder=3,
        )
        for value, y_value in zip(means, y + offsets[method]):
            label = f"{value:.3f}" if metric == METRICS[0] else f"{value:.1f}"
            ax.annotate(
                label,
                (value, y_value),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=COLORS[method],
            )

    ax.set_yticks(y, [MODEL_LABELS[model] for model in MODELS])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if metric == METRICS[0]:
        ax.set_xlim(left=0)
        ax.set_xlabel("Raw MoRF deletion AUC (lower is better)")
        ax.set_title(f"{dataset.upper()} · faithfulness")
    else:
        ax.set_xscale("log")
        ax.set_xlim(5, 420)
        ax.set_xlabel("Attribution wall time (ms, log scale)")
        ax.set_title(f"{dataset.upper()} · efficiency")


def render(units_path: Path, summary_path: Path, output_path: Path) -> None:
    frame = _validated_summary(units_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(summary_path, index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#6B7280",
        "axes.labelcolor": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
    })
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), constrained_layout=False)
    for row, dataset in enumerate(DATASETS):
        for column, metric in enumerate(METRICS):
            _draw_panel(axes[row, column], frame, dataset, metric)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.5, 0.94))
    figure.suptitle("Integrated Gradients vs Grad-CAM — formal evaluation",
                    fontsize=16, fontweight="bold", y=0.985)
    figure.text(
        0.5,
        0.015,
        "n=460 images per method/model/dataset unit; points show mean ± sample SD. "
        "Timing excludes one warm-up and excludes prediction, MoRF scoring, and file I/O.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.11, right=0.96, top=0.88, bottom=0.09,
                           hspace=0.38, wspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=Path("results/units.csv"))
    parser.add_argument(
        "--summary", type=Path, default=Path("analysis/a_ig_gradcam_summary.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/a_ig_gradcam_summary.png")
    )
    args = parser.parse_args()
    render(args.units, args.summary, args.output)


if __name__ == "__main__":
    main()
