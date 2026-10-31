"""Plot the frozen three-objective Pareto points required by experiment 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .plot_dcu_statistics import COLORS, DATA_LABEL, LABEL, MODEL_LABEL
from .validate_dcu_team_handoff import DATASETS, METHODS, MODELS


def render(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig = plt.figure(figsize=(18, 11))
    for i, dataset in enumerate(DATASETS):
        for j, model in enumerate(MODELS):
            rows = [r for r in data["pareto"] if r["cohort"] == "model_correct"
                    and (r["dataset"], r["model"]) == (dataset, model)]
            if len(rows) != 6 or {r["method"] for r in rows} != set(METHODS):
                raise ValueError("each panel must contain exactly the six frozen methods")
            rows.sort(key=lambda r: METHODS.index(r["method"]))
            values = np.array([[r["faithfulness"], -r["stability"], r["recorded_ms"]] for r in rows])
            if not np.isfinite(values).all() or (values[:, 2] <= 0).any():
                raise ValueError("finite quality scores and positive recorded times required")
            front = [not any(np.all(other <= point) and np.any(other < point) for other in values)
                     for point in values]
            if front != [r["recorded_time_3d_front"] for r in rows]:
                raise ValueError("published three-objective front differs from its means")
            ax = fig.add_subplot(2, 3, i * 3 + j + 1, projection="3d")
            for k, row in enumerate(rows):
                xyz = (row["faithfulness"], row["stability"], np.log10(row["recorded_ms"]))
                ax.scatter(*xyz, color=COLORS[k], marker="o" if front[k] else "x",
                           s=65, depthshade=False, linewidths=1.6)
                if front[k]:
                    ax.scatter(*xyz, facecolors="none", edgecolors=COLORS[k], s=170,
                               depthshade=False, linewidths=1.1)
            ax.set_title(f"{DATA_LABEL[dataset]} / {MODEL_LABEL[model]}\nn={rows[0]['n']}", fontsize=11)
            ax.set_xlabel("Raw AUC (lower)", labelpad=6)
            ax.set_ylabel("Noise Spearman (higher)", labelpad=7)
            ax.set_zlabel("log10(mean ms) (lower)", labelpad=6, fontsize=9)
            ax.view_init(elev=22, azim=-58)
            ax.tick_params(labelsize=8)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(MaxNLocator(4))
    methods = [Line2D([0], [0], color=c, marker="o", ls="", label=LABEL[m])
               for m, c in zip(METHODS, COLORS)]
    fig.legend(handles=methods, loc="upper center", bbox_to_anchor=(.5, .945), ncol=6, frameon=False)
    fig.suptitle("Three-objective Pareto sets | model-correct cohort", y=.985, fontsize=18)
    fig.text(.5, .025, "Ringed point = three-objective non-dominated; x = dominated. Fronts use means, not significance tests.\n"
             "Time is log10 of the mean recorded wrapper ms (not mean log time); the monotone display preserves dominance.\n"
             "Compare methods only within each panel. Shared DCU load / two rounds: no causal hardware speed claim.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.015, right=.94, bottom=.12, top=.86, wspace=.16, hspace=.20)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", metadata={"Software": "A final report; frozen DCU means"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("analysis/dcu_20261015/statistics.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis/a_m5_20261031/recorded_time_pareto_3d.png"))
    args = parser.parse_args()
    render(json.loads(args.input.read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    main()
