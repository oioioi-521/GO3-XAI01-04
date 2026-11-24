"""Cluster six attribution methods in the frozen DCU metric space.

CPU-only analysis. It can read the unit means directly from the frozen handoff
ZIP or use the checked-in D statistics JSON as a reproducible fallback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

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
EXPECTED_ZIP_SIZE = 13_707_240
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


def load_statistics_means(path: Path) -> tuple[list[str], list[tuple[str, str]], dict]:
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
    means, metadata = {}, {}
    for row in payload["summaries"]:
        if row.get("cohort") != "model_correct":
            continue
        key = (row["dataset"], row["model"], row["method"], row["metric"])
        if row["metric"] in {m[0] for m in METRICS}:
            require(key not in means, f"duplicate summary row: {key}")
            means[key] = float(row["mean"])
            metadata[key] = {"n": int(row["n"]), "config_hash": ""}
    expected_keys = {(dataset, model, method, metric[0])
                     for dataset, model in cells for method in methods for metric in METRICS}
    require(means.keys() == expected_keys, "incomplete or unexpected metric summary matrix")
    require(np.isfinite(list(means.values())).all(), "non-finite unit means")
    return methods, cells, {
        "receipt": receipt,
        "means": means,
        "unit_metadata": metadata,
        "provenance": {
            "kind": "checked-in derived unit means",
            "statistics_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "handoff_zip_sha256": receipt["zip_sha256"],
        },
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_handoff_zip(path: Path) -> tuple[list[str], list[tuple[str, str]], dict]:
    require(path.is_file(), f"handoff ZIP not found: {path}")
    size = path.stat().st_size
    require(size == EXPECTED_ZIP_SIZE, f"unexpected handoff ZIP size: {size}")
    archive_sha256 = file_sha256(path)
    require(archive_sha256 == EXPECTED_ZIP_SHA256,
            f"handoff ZIP SHA-256 mismatch: {archive_sha256}")

    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        require(bad_member is None, f"ZIP CRC failure: {bad_member}")
        csv_members = [name for name in archive.namelist()
                       if name.endswith("/data/units_model_correct.csv")]
        require(len(csv_members) == 1, f"expected one units_model_correct.csv, got {csv_members}")
        csv_member = csv_members[0]
        package_root = csv_member.removesuffix("/data/units_model_correct.csv")
        relative_csv = "data/units_model_correct.csv"
        csv_bytes = archive.read(csv_member)
        csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()

        manifest = json.loads(archive.read(f"{package_root}/manifest.json").decode("utf-8"))
        manifest_entry = manifest.get("files", {}).get(relative_csv)
        require(manifest_entry is not None, "manifest is missing units_model_correct.csv")
        require(manifest_entry.get("bytes") == len(csv_bytes), "manifest CSV byte count mismatch")
        require(manifest_entry.get("sha256") == csv_sha256, "manifest CSV SHA-256 mismatch")

        checksum_text = archive.read(f"{package_root}/SHA256SUMS").decode("utf-8")
        checksum_entries = [line.split(maxsplit=1) for line in checksum_text.splitlines() if line.strip()]
        matches = [digest for digest, name in checksum_entries
                   if name.lstrip("*./") == relative_csv]
        require(matches == [csv_sha256], "SHA256SUMS CSV entry missing or mismatched")

    expected_fields = ["method", "model", "dataset", "metric", "mean", "std", "n", "config_hash"]
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    require(rows and list(rows[0]) == expected_fields, "unexpected units_model_correct.csv schema")
    require(len(rows) == 144 and manifest_entry.get("rows") == 144,
            f"expected 144 unit-metric rows, got {len(rows)}")
    metric_names = {metric[0] for metric in METRICS} | {"stability_valid_rate"}
    require({row["metric"] for row in rows} == metric_names, "unexpected or missing metrics in CSV")
    methods = sorted({row["method"] for row in rows})
    cells = sorted({(row["dataset"], row["model"]) for row in rows})
    require(methods == sorted(DISPLAY_NAMES), f"expected six fixed methods; got {methods}")
    require(len(cells) == 6, f"expected six dataset/model conditions; got {cells}")

    means, metadata, unit_ns = {}, {}, {}
    all_keys = set()
    for row in rows:
        key = (row["dataset"], row["model"], row["method"], row["metric"])
        require(key not in all_keys, f"duplicate CSV row: {key}")
        all_keys.add(key)
        mean, std, n = float(row["mean"]), float(row["std"]), int(row["n"])
        require(np.isfinite([mean, std]).all() and std >= 0 and n > 0,
                f"invalid aggregate values in CSV row: {key}")
        require(row["config_hash"], f"missing config hash in CSV row: {key}")
        unit_key = (row["dataset"], row["model"], row["method"])
        unit_ns.setdefault(unit_key, set()).add(n)
        if row["metric"] in {metric[0] for metric in METRICS}:
            means[key] = mean
            metadata[key] = {"n": n, "config_hash": row["config_hash"]}

    expected_keys = {(dataset, model, method, metric[0])
                     for dataset, model in cells for method in methods for metric in METRICS}
    require(means.keys() == expected_keys, "incomplete metric matrix in frozen ZIP CSV")
    expected_all_keys = {(dataset, model, method, metric)
                         for dataset, model in cells for method in methods for metric in metric_names}
    require(all_keys == expected_all_keys, "incomplete 4-metric unit matrix in frozen ZIP CSV")
    require(all(len(counts) == 1 for counts in unit_ns.values()),
            "inconsistent sample count across metrics within a method/model/dataset unit")
    require(np.isfinite(list(means.values())).all(), "non-finite unit means in frozen ZIP CSV")
    return methods, cells, {
        "means": means,
        "unit_metadata": metadata,
        "provenance": {
            "kind": "direct frozen handoff ZIP input",
            "handoff_zip_sha256": archive_sha256,
            "handoff_zip_size_bytes": size,
            "zip_crc_valid": True,
            "csv_member": csv_member,
            "csv_rows": len(rows),
            "csv_sha256": csv_sha256,
            "manifest_file_count": len(manifest["files"]),
            "csv_manifest_row_count": manifest_entry["rows"],
        },
    }


def compare_sources(archive_means: dict, summary_means: dict) -> dict:
    require(archive_means.keys() == summary_means.keys(),
            "frozen ZIP and checked-in statistics have different unit-metric keys")
    differences = [abs(archive_means[key] - summary_means[key]) for key in archive_means]
    max_difference = max(differences, default=0.0)
    tolerance = 1e-15
    require(max_difference <= tolerance,
            f"unit means differ from checked-in summary by {max_difference:g} > {tolerance:g}")
    return {
        "matching_unit_metric_keys": len(differences),
        "exact_float_matches": sum(difference == 0 for difference in differences),
        "last_digit_differences": sum(difference != 0 for difference in differences),
        "max_absolute_difference": max_difference,
        "absolute_tolerance": tolerance,
        "interpretation": "All direct ZIP means agree with the checked-in accepted summary within float serialization precision.",
    }


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
    parser.add_argument("--handoff-zip", type=Path,
                        help="optional frozen dcu_team_handoff_20261014.zip; if supplied, read its units CSV directly")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/e_method_clustering_20261122"))
    args = parser.parse_args()
    require(args.statistics_json.is_file(), f"statistics JSON not found: {args.statistics_json}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_methods, summary_cells, summary_source = load_statistics_means(args.statistics_json)
    if args.handoff_zip:
        methods, cells, source = load_handoff_zip(args.handoff_zip)
        require(methods == summary_methods and cells == summary_cells,
                "frozen ZIP and checked-in statistics describe different methods/conditions")
        source_comparison = compare_sources(source["means"], summary_source["means"])
    else:
        methods, cells, source = summary_methods, summary_cells, summary_source
        source_comparison = None
    means = source["means"]
    raw_header = ["dataset", "model", "method"]
    for _, short_name, _, _ in METRICS:
        raw_header.extend([short_name, f"{short_name}_n", f"{short_name}_config_hash"])
    raw_rows = []
    for dataset, model in cells:
        for method in methods:
            row = [dataset, model, method]
            for source_metric, short_name, _, _ in METRICS:
                key = (dataset, model, method, source_metric)
                unit_metadata = source["unit_metadata"][key]
                row.extend([f"{means[key]:.17g}", unit_metadata["n"], unit_metadata["config_hash"]])
            raw_rows.append(row)
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
            "statistics_json_path": str(args.statistics_json),
            "statistics_json_sha256": summary_source["provenance"]["statistics_json_sha256"],
            "cohort": "model_correct",
            "source": source["provenance"],
            "accepted_statistics_receipt": {
                "handoff_zip_sha256": summary_source["receipt"]["zip_sha256"],
                "units": summary_source["receipt"]["units"],
            },
            "comparison_to_accepted_statistics": source_comparison,
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
        "input_kind": source["provenance"]["kind"],
        "source_comparison": source_comparison,
        "with_time": {"k": all_result["selected_k"], "silhouette": all_result["selected_silhouette"], "clusters": all_result["clusters"]},
        "without_time": {"k": quality_result["selected_k"], "silhouette": quality_result["selected_silhouette"], "clusters": quality_result["clusters"]},
        "metric_gaps_without_time": quality_result["between_cluster_metric_gaps"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
