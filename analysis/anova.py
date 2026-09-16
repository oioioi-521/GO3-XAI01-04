"""Factorial ANOVA and a matched-block Friedman fallback.

The ANOVA is fitted on per-image values for one metric.  It reports Type-II
sums of squares and partial eta-squared.  Assumption checks are diagnostics,
not an automatic licence to discard results.  When assumptions are poor, the
matched-block Friedman result provides a transparent non-parametric comparison
across methods.

Example:
    python -m analysis.anova --input results/per_image.csv \
        --metric faithfulness_morf_auc --output-dir analysis/out/faithfulness
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from .schema import read_csv, validate_per_image


FACTOR_COLUMNS = ["method", "model", "dataset"]


def _metric_rows(per_image: pd.DataFrame, metric: str) -> pd.DataFrame:
    validate_per_image(per_image)
    selected = per_image[per_image["metric"].astype(str) == metric].copy()
    if selected.empty:
        raise ValueError(f"metric not found: {metric}")
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    missing_levels = [column for column in FACTOR_COLUMNS if selected[column].nunique() < 2]
    if missing_levels:
        raise ValueError(
            "factorial ANOVA requires at least two levels for: "
            + ", ".join(missing_levels)
        )
    return selected


def assumption_checks(frame: pd.DataFrame, residuals: pd.Series) -> dict[str, Any]:
    """Return normality and homoscedasticity diagnostics."""

    clean_residuals = pd.Series(residuals).dropna().astype(float)
    # scipy.shapiro warns and becomes overly sensitive for very large samples.
    shapiro_sample = clean_residuals.iloc[:5000]
    shapiro = stats.shapiro(shapiro_sample)
    groups = [
        group["value"].to_numpy(dtype=float)
        for _, group in frame.groupby(FACTOR_COLUMNS, sort=True)
        if len(group) >= 2
    ]
    if len(groups) >= 2:
        levene = stats.levene(*groups, center="median")
        levene_result: dict[str, float | None] = {
            "statistic": float(levene.statistic),
            "p_value": float(levene.pvalue),
        }
    else:
        levene_result = {"statistic": None, "p_value": None}
    return {
        "shapiro_residuals": {
            "n_used": int(len(shapiro_sample)),
            "statistic": float(shapiro.statistic),
            "p_value": float(shapiro.pvalue),
        },
        "levene_cells": {"groups_used": len(groups), **levene_result},
    }


def run_factorial_anova(
    per_image: pd.DataFrame, metric: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit ``method * model * dataset`` and return table plus diagnostics."""

    selected = _metric_rows(per_image, metric)
    formula = "value ~ C(method) * C(model) * C(dataset)"
    model = ols(formula, data=selected).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        table = anova_lm(model, typ=2).reset_index(names="effect")
    residual_sum = float(
        table.loc[table["effect"] == "Residual", "sum_sq"].iloc[0]
    )
    table["partial_eta_sq"] = np.where(
        table["effect"] == "Residual",
        np.nan,
        table["sum_sq"] / (table["sum_sq"] + residual_sum),
    )
    diagnostics = {
        "metric": metric,
        "formula": formula,
        "rows": int(len(selected)),
        "levels": {
            column: sorted(selected[column].astype(str).unique().tolist())
            for column in FACTOR_COLUMNS
        },
        "assumptions": assumption_checks(selected, model.resid),
        "note": (
            "If residual normality or equal-variance diagnostics are poor, report "
            "the limitation and compare with the matched-block Friedman result."
        ),
    }
    return table, diagnostics


def run_friedman_by_method(per_image: pd.DataFrame, metric: str) -> dict[str, Any]:
    """Compare methods on blocks matched by image, model and dataset.

    Only complete blocks containing every observed method are used.  The result
    therefore states the retained block count explicitly.
    """

    validate_per_image(per_image)
    selected = per_image[per_image["metric"].astype(str) == metric].copy()
    if selected.empty:
        raise ValueError(f"metric not found: {metric}")
    methods = sorted(selected["method"].astype(str).unique())
    if len(methods) < 3:
        return {
            "metric": metric,
            "status": "not_run",
            "reason": "Friedman requires at least three methods",
            "methods": methods,
        }
    wide = selected.pivot_table(
        index=["image_id", "model", "dataset"],
        columns="method",
        values="value",
        aggfunc="first",
    ).reindex(columns=methods)
    complete = wide.dropna()
    if len(complete) < 2:
        return {
            "metric": metric,
            "status": "not_run",
            "reason": "fewer than two complete matched blocks",
            "methods": methods,
            "complete_blocks": int(len(complete)),
        }
    result = stats.friedmanchisquare(
        *(complete[method].to_numpy(dtype=float) for method in methods)
    )
    ranks = complete.rank(axis=1, method="average", ascending=True).mean(axis=0)
    return {
        "metric": metric,
        "status": "ok",
        "methods": methods,
        "complete_blocks": int(len(complete)),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_ranks_lower_value_is_better": {
            method: float(ranks[method]) for method in methods
        },
        "note": "Metric direction must be checked before interpreting mean ranks.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = read_csv(args.input, "per_image")
    table, diagnostics = run_factorial_anova(frame, args.metric)
    friedman = run_friedman_by_method(frame, args.metric)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "anova.csv", index=False)
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "friedman.json").write_text(
        json.dumps(friedman, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ANOVA rows={len(table)}, output={args.output_dir}")


if __name__ == "__main__":
    main()
