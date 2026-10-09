"""Render scoped midterm D plots from existing formal summaries.

The four-method ImageNet faithfulness plot is descriptive all-image evidence.
The two-objective Pareto plot uses only A's same-GPU IG/Grad-CAM units and
evaluates dominance separately for each model/dataset pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("resnet50", "densenet121", "vgg16")
DATASETS = ("imagenet", "voc")
METHODS = ("ig", "gradcam", "occlusion", "rise")
FAITHFULNESS = "faithfulness_morf_auc_raw"
EFFICIENCY = "efficiency_time_ms"
LABELS = {"ig": "IG", "gradcam": "Grad-CAM", "occlusion": "Occlusion", "rise": "RISE"}
MODEL_LABELS = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "vgg16": "VGG-16"}
COLORS = {"ig": "#2563EB", "gradcam": "#D97706", "occlusion": "#059669", "rise": "#9333EA"}


def _checked_rows(frame: pd.DataFrame, expected: set[tuple], columns: list[str], label: str) -> pd.DataFrame:
    required = set(columns) | {"mean", "std", "n", "config_hash"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{label}: missing columns {sorted(required - set(frame.columns))}")
    keys = list(frame[columns].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"{label}: missing, extra, or duplicated formal summary keys")
    if not (frame["n"] == 460).all():
        raise ValueError(f"{label}: expected n=460 for every summary")
    if not np.isfinite(frame[["mean", "std"]].to_numpy(float)).all() or (frame["std"] < 0).any():
        raise ValueError(f"{label}: invalid mean or sample SD")
    if not frame["config_hash"].astype(str).str.fullmatch(r"[0-9a-f]{12}").all():
        raise ValueError(f"{label}: invalid config hash")
    return frame.copy()


def load_points(a_path: Path, b_path: Path, rise_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(a_path)
    a = a[a["method"].isin(("ig", "gradcam"))]
    a_expected = {(method, model, dataset, metric) for method in ("ig", "gradcam")
                  for model in MODELS for dataset in DATASETS
                  for metric in (FAITHFULNESS, EFFICIENCY)}
    a = _checked_rows(a, a_expected, ["method", "model", "dataset", "metric"], "A")
    b = pd.read_csv(b_path)
    b = b[(b["method"] == "occlusion") & (b["dataset"] == "imagenet")
          & (b["metric"] == FAITHFULNESS)]
    b = _checked_rows(b, {(model,) for model in MODELS}, ["model"], "B/ImageNet")
    rise_source = json.loads(rise_path.read_text(encoding="utf-8"))
    rise = pd.DataFrame(rise_source["rows"])
    rise = _checked_rows(rise, {(model,) for model in MODELS}, ["model"], "C/RISE/ImageNet")
    rise["method"] = "rise"
    rise["dataset"] = "imagenet"
    rise["metric"] = FAITHFULNESS
    four = pd.concat([a[(a["dataset"] == "imagenet") & (a["metric"] == FAITHFULNESS)],
                      b, rise], ignore_index=True)
    four = _checked_rows(four, {(method, model) for method in METHODS for model in MODELS},
                         ["method", "model"], "four-method ImageNet")
    if not four["mean"].between(0, 1).all():
        raise ValueError("four-method ImageNet: AUC outside [0,1]")
    return four, a


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": "#6B7280", "axes.labelcolor": "#111827"})


def plot_four_methods(four: pd.DataFrame, output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=True)
    for ax, model in zip(axes, MODELS):
        unit = four[four["model"] == model].set_index("method").loc[list(METHODS)]
        y = np.arange(len(METHODS))
        for index, method in enumerate(METHODS):
            mean = float(unit.loc[method, "mean"])
            sd = float(unit.loc[method, "std"])
            ax.errorbar(mean, index, xerr=[[min(sd, mean)], [min(sd, 1 - mean)]],
                        fmt="o", color=COLORS[method], capsize=3, markersize=6)
            ax.annotate(f"{mean:.3f}", (mean, index), xytext=(5, -13),
                        textcoords="offset points", fontsize=8, color=COLORS[method])
        ax.set_yticks(y, [LABELS[method] for method in METHODS])
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_xlabel("Raw MoRF AUC (lower is better)")
        ax.set_title(MODEL_LABELS[model])
        ax.grid(axis="x", color="#D1D5DB", alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("ImageNet faithfulness · four methods", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015,
             "n=460 each; mean ± sample SD (bars clipped to [0,1]). All-image sensitivity view, "
             "not paired correct-only ranking. RISE uses C's summary.",
             ha="center", fontsize=8.5, color="#4B5563")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.20, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _paired_pareto(a: pd.DataFrame) -> pd.DataFrame:
    points = a.pivot(index=["method", "model", "dataset"], columns="metric", values="mean")
    points = points.reset_index()
    lookup = {}
    for dataset in DATASETS:
        for model in MODELS:
            pair = points[(points["dataset"] == dataset) & (points["model"] == model)]
            if len(pair) != 2:
                raise ValueError(f"A: incomplete method pair for {model}/{dataset}")
            for _, row in pair.iterrows():
                other = pair[pair["method"] != row["method"]].iloc[0]
                lookup[(row["method"], model, dataset)] = not (
                    other[FAITHFULNESS] <= row[FAITHFULNESS] and
                    other[EFFICIENCY] <= row[EFFICIENCY] and
                    (other[FAITHFULNESS] < row[FAITHFULNESS] or
                     other[EFFICIENCY] < row[EFFICIENCY])
                )
    points["within_pair_pareto"] = [lookup[key] for key in
                                    points[["method", "model", "dataset"]].itertuples(index=False, name=None)]
    return points


def plot_a_pareto(a: pd.DataFrame, output: Path) -> pd.DataFrame:
    points = _paired_pareto(a)
    _style()
    model_colors = {"resnet50": "#2563EB", "densenet121": "#059669", "vgg16": "#D97706"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        for model in MODELS:
            pair = points[(points["dataset"] == dataset) & (points["model"] == model)]
            pair = pair.set_index("method").loc[["gradcam", "ig"]]
            x = pair[EFFICIENCY].to_numpy(float)
            y = pair[FAITHFULNESS].to_numpy(float)
            ax.plot(x, y, color=model_colors[model], alpha=0.55, linewidth=1)
            for method, marker in (("gradcam", "s"), ("ig", "o")):
                ax.scatter(float(pair.loc[method, EFFICIENCY]),
                           float(pair.loc[method, FAITHFULNESS]), marker=marker,
                           color=model_colors[model], s=55, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Attribution time (ms, log scale; lower is better)")
        ax.set_ylabel("Raw MoRF AUC (lower is better)")
        ax.set_title(dataset.upper())
        ax.grid(color="#D1D5DB", alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color=model_colors[model], label=MODEL_LABELS[model])
              for model in MODELS]
    legend += [Line2D([0], [0], marker="s", linestyle="None", color="#374151", label="Grad-CAM"),
               Line2D([0], [0], marker="o", linestyle="None", color="#374151", label="IG")]
    fig.legend(handles=legend, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.91))
    fig.suptitle("IG vs Grad-CAM · preliminary two-objective Pareto", fontsize=15,
                 fontweight="bold", y=0.98)
    fig.text(0.5, 0.015,
             "A's RTX 2080 Ti only. 460-image all-sample means; dominance evaluated within each model/dataset pair. "
             "Stability is excluded.", ha="center", fontsize=8.5, color="#4B5563")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.19, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=Path("analysis/a_ig_gradcam_summary.csv"))
    parser.add_argument("--b", type=Path,
                        default=Path("results/analysis/occlusion_eval_acceptance/units.csv"))
    parser.add_argument("--rise", type=Path, default=Path("analysis/midterm_rise_imagenet.json"))
    parser.add_argument("--four-output", type=Path,
                        default=Path("analysis/midterm_four_method_faithfulness.png"))
    parser.add_argument("--pareto-output", type=Path,
                        default=Path("analysis/midterm_a_ig_gradcam_pareto.png"))
    args = parser.parse_args()
    four, a = load_points(args.a, args.b, args.rise)
    plot_four_methods(four, args.four_output)
    pareto = plot_a_pareto(a, args.pareto_output)
    print(f"four_method_units={len(four)}, a_pareto_units={len(pareto)}, "
          f"within_pair_front={int(pareto['within_pair_pareto'].sum())}")


if __name__ == "__main__":
    main()
