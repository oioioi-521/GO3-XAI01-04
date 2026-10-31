"""Compute pairwise metric correlations on matched per-image observations.

Example:
    python -m analysis.correlation --input results/per_image.csv \
        --output-dir analysis/out/correlation
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .schema import read_csv, validate_per_image


OBSERVATION_KEY = ["image_id", "dataset", "model", "method"]


def metric_correlations(
    per_image: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return Pearson matrix, Spearman matrix and pairwise test details."""

    validate_per_image(per_image)
    wide = per_image.pivot(
        index=OBSERVATION_KEY, columns="metric", values="value"
    ).sort_index(axis=1)
    if wide.shape[1] < 2:
        raise ValueError("correlation analysis requires at least two metrics")
    pearson = wide.corr(method="pearson", min_periods=3)
    spearman = wide.corr(method="spearman", min_periods=3)

    rows: list[dict[str, object]] = []
    metrics = list(wide.columns.astype(str))
    for left_index, left in enumerate(metrics):
        for right in metrics[left_index + 1 :]:
            pair = wide[[left, right]].dropna()
            row: dict[str, object] = {
                "metric_a": left,
                "metric_b": right,
                "n": int(len(pair)),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
            }
            if len(pair) >= 3 and pair[left].nunique() > 1 and pair[right].nunique() > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pearson_test = stats.pearsonr(pair[left], pair[right])
                    spearman_test = stats.spearmanr(pair[left], pair[right])
                row.update(
                    {
                        "pearson_r": float(pearson_test.statistic),
                        "pearson_p": float(pearson_test.pvalue),
                        "spearman_rho": float(spearman_test.statistic),
                        "spearman_p": float(spearman_test.pvalue),
                    }
                )
            rows.append(row)
    return pearson, spearman, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = read_csv(args.input, "per_image")
    pearson, spearman, pairs = metric_correlations(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pearson.to_csv(args.output_dir / "pearson.csv")
    spearman.to_csv(args.output_dir / "spearman.csv")
    pairs.to_csv(args.output_dir / "pairwise_tests.csv", index=False)
    print(f"metric_pairs={len(pairs)}, output={args.output_dir}")


if __name__ == "__main__":
    main()
