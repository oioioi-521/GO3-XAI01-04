"""Cluster six attribution methods in the frozen DCU metric space.

CPU-only analysis. The checked-in D statistics JSON contains the exact unit
means derived from the frozen handoff ZIP; this script uses those means rather
than per-image rows because the clustering unit is the method, not the image.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

_CACHE_DIR = Path(tempfile.gettempdir()) / "go3-xai01-04-e-method-clustering-cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform


EXPECTED_ZIP_SHA256 = "14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc"
METRICS = (
    ("faithfulness_morf_auc_raw", "raw_morf_auc", -1, "Raw MoRF AUC (lower is better)"),
    ("stability_spearman", "noise_stability_spearman", 1, "Noise Spearman (higher is better)"),
    ("efficiency_time_ms", "wrapper_time_ms", -1, "Wrapper time (lower is better)"),
)
DISPLAY_NAMES = {
    "ig": "IG",
    "gradcam": "Grad-CAM",
    "occlusion": "Occlusion",
    "rise": "RISE",
    "kernelshap": "KernelSHAP",
    "lime": "LIME",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_labels(labels: np.ndarray, methods: list[str]) -> dict[str, int]:
    groups: dict[int, list[str]] = {}
    for method, label in zip(methods, labels, strict=True):
        groups.setdefault(int(label), []).append(method)
    ordered = sorted(groups.values(), key=lambda group: min(group))
    return {method: i + 1 for i, group in enumerate(ordered) for method in group}


def silhouette_mean(distance: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette from a precomputed distance matrix (small n = 6)."""
    groups = sorted(set(map(int, labels)))
    if len(groups) < 2 or len(groups) >= len(labels):
        return 0.0
    scores = []
    for i, label in enumerate(labels):
        own = np.flatnonzero(labels == label)
        other_groups = [g for g in groups if g != int(label)]
        if len(own) == 1:
            scores.append(0.0)
            continue
        a = float(distance[i, own[own != i]].mean())
        b = min(float(distance[i, labels == other].mean()) for other in other_groups)
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return float(np.mean(scores))


def load_means(path: Path) -> tuple[list[str], list[tuple[str, str]], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    receipt = payload.get("receipt", {})
    require(receipt.get("zip_sha256") == EXPECTED_ZIP_SHA256,
            "source statistics do not identify the frozen DCU handoff ZIP")
    require(receipt.get("units") == 36, "source statistics do not record all 36 units")
    methods = sorted({row["method"] for row in payload["summaries"]
                      if row.get("cohort") == "model_correct"})
    cells = sorted({(row["dataset"], row["model"]) for row in payload["summaries"]
                    if row.get("cohort") == "model_correct"})
    expected_methods = sorted(DISPLAY_NAMES)
    require(methods == expected_methods, f"expected six fixed methods; got {methods}")
    require(len(cells) == 6, f"expected six dataset/model conditions; got {cells}")
    means = {}
    for row in payload["summaries"]:
        if row.get("cohort") != "model_correct":
            continue
        key = (row["dataset"], row["model"], row["method"], row["metric"])
        if row["metric"] in {m[0] for m in METRICS}:
            require(key not in means, f"duplicate summary row: {key}")
            means[key] = float(row["mean"])
    expected_keys = {(dataset, model, method, metric[0])
                     for dataset, model in cells for method in methods for metric in METRICS}
    require(means.keys() == expected_keys, "incomplete or unexpected metric summary matrix")
    require(np.isfinite(list(means.values())).all(), "non-finite unit means")
    return methods, cells, {"receipt": receipt, "means": means}


def make_features(methods: list[str], cells: list[tuple[str, str]], means: dict,
                  include_time: bool) -> tuple[np.ndarray, list[str], dict]:
    selected_metrics = METRICS if include_time else METRICS[:2]
    columns: list[str] = []
    columns_meta = []
    feature_rows = []
    for dataset, model in cells:
        for source_metric, short_name, direction, label in selected_metrics:
            values = np.array([means[(dataset, model, method, source_metric)]
                               for method in methods], dtype=float)
            sd = float(values.std(ddof=0))
            require(sd > 0, f"zero variance in {dataset}/{model}/{source_metric}")
            oriented_z = direction * (values - values.mean()) / sd
            name = f"{dataset}__{model}__{short_name}__oriented_z"
            columns.append(name)
            columns_meta.append({
                "column": name,
                "dataset": dataset,
                "model": model,
                "source_metric": source_metric,
                "direction_multiplier": direction,
                "standardization": "population z-score across six methods within this condition",
                "interpretation": "higher means relatively better within this condition",
            })
            feature_rows.append(oriented_z)
    matrix = np.column_stack(feature_rows)
    require(matrix.shape == (len(methods), len(columns)), "unexpected feature matrix shape")
    return matrix, columns, {"columns": columns_meta, "metric_groups": [m[1] for m in selected_metrics]}


def analyze_matrix(methods: list[str], matrix: np.ndarray, columns: list[str],
                   metric_groups: list[str]) -> dict:
    tree = linkage(matrix, method="ward", metric="euclidean", optimal_ordering=True)
    distances = squareform(pdist(matrix, metric="euclidean"))
    k_candidates = []
    for k in range(2, min(5, len(methods))):
        raw_labels = fcluster(tree, t=k, criterion="maxclust")
        labels = canonical_labels(raw_labels, methods)
        sil = silhouette_mean(distances, raw_labels)
        k_candidates.append({
            "k": k,
            "silhouette": sil,
            "clusters": {str(i): [m for m in methods if labels[m] == i]
                         for i in sorted(set(labels.values()))},
        })
    best = max(k_candidates, key=lambda item: (item["silhouette"], -item["k"]))
    cluster_id = {method: int(cid) for cid, members in best["clusters"].items() for method in members}

    profiles = {}
    for cid, members in best["clusters"].items():
        idx = [methods.index(method) for method in members]
        profiles[cid] = {
            group: float(matrix[np.ix_(idx, [i for i, col in enumerate(columns) if f"__{group}__" in col])].mean())
            for group in metric_groups
        }
    separation = []
    if len(profiles) == 2:
        left, right = sorted(profiles)
        for group in metric_groups:
            separation.append({
                "metric_group": group,
                "absolute_mean_oriented_z_gap": abs(profiles[left][group] - profiles[right][group]),
                "cluster_1_mean_oriented_z": profiles[left][group],
                "cluster_2_mean_oriented_z": profiles[right][group],
            })
        separation.sort(key=lambda item: item["absolute_mean_oriented_z_gap"], reverse=True)

    return {
        "n_methods": len(methods),
        "n_features": int(matrix.shape[1]),
        "distance": "Euclidean on condition-wise direction-aligned z-scores",
        "linkage": "Ward",
        "merge_heights": [float(x) for x in tree[:, 2]],
        "linkage_matrix": tree.tolist(),
        "pairwise_distances": {
            methods[i]: {methods[j]: float(distances[i, j]) for j in range(len(methods))}
            for i in range(len(methods))
        },
        "candidate_k": k_candidates,
        "selection_rule": "Choose the highest mean silhouette among k=2,3,4; ties favor smaller k. The selected k=2 cut also matches the largest terminal Ward merge-height jump.",
        "selected_k": int(best["k"]),
        "selected_silhouette": float(best["silhouette"]),
        "clusters": best["clusters"],
        "direction_aligned_z_profiles_by_metric": profiles,
        "between_cluster_metric_gaps": separation,
    }


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def render_dendrograms(methods: list[str], results: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), constrained_layout=True)
    titles = ("Faithfulness + stability + time (18 features)",
              "Faithfulness + stability only (12 features)")
    for ax, result, title in zip(axes, results, titles, strict=True):
        tree = np.asarray(result["linkage_matrix"], dtype=float)
        tree_for_plot = tree.copy()
        threshold = float((tree_for_plot[-1, 2] + tree_for_plot[-2, 2]) / 2)
        dendrogram(
            tree_for_plot,
            labels=[DISPLAY_NAMES[m] for m in methods],
            color_threshold=threshold,
            above_threshold_color="#6B7280",
            ax=ax,
        )
        ax.axhline(threshold, color="#6B7280", linestyle="--", linewidth=1, alpha=.75)
        labels = result["clusters"]
        cluster_for_method = {method: int(cluster) for cluster, members in labels.items() for method in members}
        palette = {1: "#2463A6", 2: "#C25A31", 3: "#5F7F3A", 4: "#80559B"}
        for tick in ax.get_xmajorticklabels():
            method = next(m for m, display in DISPLAY_NAMES.items() if display == tick.get_text())
            tick.set_color(palette.get(cluster_for_method[method], "#222222"))
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_ylabel("Ward distance")
        ax.set_xlabel("Attribution method")
        ax.grid(axis="y", color="#D9DEE7", linewidth=.6, alpha=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.98, .98,
                f"selected k={result['selected_k']}  |  silhouette={result['selected_silhouette']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#D9DEE7"})
    fig.suptitle("Six-method clustering from frozen DCU unit means", fontsize=15, weight="bold")
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-json", type=Path,
                        default=Path("analysis/dcu_20261015/statistics.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/e_method_clustering_20261122"))
    args = parser.parse_args()
    require(args.statistics_json.is_file(), f"statistics JSON not found: {args.statistics_json}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    methods, cells, source = load_means(args.statistics_json)
    means = source["means"]
    raw_header = ["dataset", "model", "method"] + [metric[1] for metric in METRICS]
    raw_rows = []
    for dataset, model in cells:
        for method in methods:
            raw_rows.append([dataset, model, method] + [
                f"{means[(dataset, model, method, metric[0])]:.17g}" for metric in METRICS
            ])
    write_csv(args.output_dir / "unit_means.csv", raw_header, raw_rows)

    all_matrix, all_columns, all_meta = make_features(methods, cells, means, include_time=True)
    quality_matrix, quality_columns, quality_meta = make_features(methods, cells, means, include_time=False)
    all_result = analyze_matrix(methods, all_matrix, all_columns, all_meta["metric_groups"])
    quality_result = analyze_matrix(methods, quality_matrix, quality_columns, quality_meta["metric_groups"])

    feature_rows = [[method] + [f"{v:.12g}" for v in all_matrix[i]]
                    for i, method in enumerate(methods)]
    write_csv(args.output_dir / "method_features_18d.csv", ["method"] + all_columns, feature_rows)
    quality_rows = [[method] + [f"{v:.12g}" for v in quality_matrix[i]]
                    for i, method in enumerate(methods)]
    write_csv(args.output_dir / "method_features_quality_12d.csv", ["method"] + quality_columns, quality_rows)

    payload = {
        "input": {
            "path": str(args.statistics_json),
            "statistics_json_sha256": hashlib.sha256(args.statistics_json.read_bytes()).hexdigest(),
            "cohort": "model_correct",
            "source_handoff_zip_sha256_recorded_in_statistics": source["receipt"]["zip_sha256"],
            "source_units": source["receipt"]["units"],
            "input_kind": "checked-in derived unit-mean statistics; raw handoff ZIP was not available in this checkout",
            "note": "The frozen archive SHA is verified through the D statistics receipt; this run does not claim to re-read or re-hash the archive itself.",
        },
        "methods": methods,
        "environment": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "matplotlib")
        },
        "display_names": DISPLAY_NAMES,
        "conditions": [{"dataset": dataset, "model": model} for dataset, model in cells],
        "feature_construction": {
            "primary": all_meta,
            "without_time": quality_meta,
            "per_method_row_count": len(methods),
            "normalization_reason": "Each of the six dataset/model conditions contributes equal-variance, direction-aligned metric features; raw scales are never pooled across datasets or models.",
        },
        "with_time": all_result,
        "without_time": quality_result,
        "interpretation_limits": [
            "Only six method labels are clustered; this is descriptive structure, not population-level inference or a significance test.",
            "Unit means conceal image-level heterogeneity and the original sample sizes vary by model-correct cohort.",
            "Recorded wrapper time came from shared DCU batches and is not a controlled hardware ranking.",
            "Noise stability measures agreement under the specified input noise protocol, not explanation correctness.",
        ],
    }
    (args.output_dir / "clustering_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render_dendrograms(methods, [all_result, quality_result],
                       args.output_dir / "six_method_dendrograms.png")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "with_time": {"k": all_result["selected_k"], "silhouette": all_result["selected_silhouette"], "clusters": all_result["clusters"]},
        "without_time": {"k": quality_result["selected_k"], "silhouette": quality_result["selected_silhouette"], "clusters": quality_result["clusters"]},
        "metric_gaps_without_time": quality_result["between_cluster_metric_gaps"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
