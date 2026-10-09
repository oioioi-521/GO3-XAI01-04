"""Plot the six validated 460-image KernelSHAP formal eval units.

Run the independent formal validator first. This plot is within-method only;
it is not a cross-method ranking or a statistical significance test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ("resnet50", "densenet121", "vgg16")
DATASETS = ("imagenet", "voc")
METRICS = (
    "faithfulness_morf_auc_raw",
    "stability_spearman",
    "efficiency_time_ms",
    "stability_valid_rate",
)
MODEL_LABELS = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "vgg16": "VGG-16"}
PANELS = (
    ("faithfulness_morf_auc_raw", "Faithfulness", "MoRF deletion AUC (lower is better)"),
    ("stability_spearman", "Stability", "Perturbation Spearman (higher is better)"),
    ("efficiency_time_ms", "Efficiency", "Attribution time (ms, lower is better)"),
)


def validated_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"method", "model", "dataset", "metric", "mean", "std", "n", "config_hash"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing summary columns: {sorted(required - set(frame.columns))}")
    selected = frame[frame["method"] == "kernelshap"].copy()
    expected = {
        (model, dataset, metric)
        for model in MODELS for dataset in DATASETS for metric in METRICS
    }
    actual = list(selected[["model", "dataset", "metric"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("require exactly four formal metrics for each of six KernelSHAP units")
    if not (selected["n"] == 460).all():
        raise ValueError("every plotted metric must have n=460")
    if not np.isfinite(selected[["mean", "std"]].to_numpy(float)).all():
        raise ValueError("summary contains non-finite values")
    if (selected["std"] < 0).any():
        raise ValueError("summary contains negative sample standard deviation")
    for metric, minimum, maximum in (
        ("faithfulness_morf_auc_raw", 0, 1),
        ("stability_spearman", -1, 1),
        ("stability_valid_rate", 0, 1),
        ("efficiency_time_ms", 0, float("inf")),
    ):
        values = selected.loc[selected["metric"] == metric, "mean"]
        if not values.between(minimum, maximum).all():
            raise ValueError(f"out-of-range {metric} mean")
    if not selected["config_hash"].astype(str).str.fullmatch(r"[0-9a-f]{12}").all():
        raise ValueError("summary contains a missing or invalid config hash")
    return selected.sort_values(["dataset", "metric", "model"])


def render(units_path: Path, output_path: Path) -> None:
    frame = validated_summary(units_path)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = {"imagenet": "#2563EB", "voc": "#D97706"}
    for row_index, dataset in enumerate(DATASETS):
        for column_index, (metric, title, axis_label) in enumerate(PANELS):
            ax = axes[row_index, column_index]
            subset = frame[(frame["dataset"] == dataset) & (frame["metric"] == metric)]
            subset = subset.set_index("model").loc[list(MODELS)]
            means = subset["mean"].to_numpy(float)
            stds = subset["std"].to_numpy(float)
            y = np.arange(len(MODELS))
            lower = np.minimum(stds, means)
            upper = np.minimum(stds, 1 - means) if metric == "stability_spearman" else stds
            ax.errorbar(means, y, xerr=np.vstack([lower, upper]),
                        fmt="o", color=colors[dataset], capsize=3, linewidth=1.3)
            ax.set_yticks(y, [MODEL_LABELS[model] for model in MODELS])
            ax.invert_yaxis()
            ax.grid(axis="x", color="#D1D5DB", alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if metric == "stability_spearman":
                ax.set_xlim(left=max(0, float(np.min(means - stds)) - 0.005), right=1.001)
            else:
                ax.set_xlim(left=0)
            ax.set_xlabel(axis_label)
            ax.set_title(f"{dataset.upper()} · {title}")
    fig.suptitle("KernelSHAP · six formal evaluation units", fontsize=16, fontweight="bold")
    fig.supxlabel(
        "Each point is the mean ± sample SD of 460 images. Within-method descriptive results only; "
        "GPU timing is not cross-device comparable.", fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, default=Path("results/kernelshap_eval/units.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/analysis/kernelshap_formal_summary.png"))
    args = parser.parse_args()
    render(args.units, args.output)


if __name__ == "__main__":
    main()
