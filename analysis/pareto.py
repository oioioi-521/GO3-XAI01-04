"""Compute a configurable Pareto frontier from unit-level metrics.

Metric direction is explicit because the project mixes scores where larger is
better with costs where smaller is better.  For example, MoRF deletion AUC,
Max-Sensitivity and wall-clock time are all minimized.

Example:
    python -m analysis.pareto --input analysis/data_long.csv \
        --metric faithfulness_morf_auc:min \
        --metric max_sensitivity:min \
        --metric efficiency_time_ms:min \
        --output analysis/out/pareto.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import read_csv, validate_units


UNIT_ID = ["method", "model", "dataset"]


def parse_metric_specs(specs: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"metric must be NAME:min or NAME:max: {spec}")
        name, direction = spec.rsplit(":", 1)
        name, direction = name.strip(), direction.strip().lower()
        if not name or direction not in {"min", "max"}:
            raise ValueError(f"invalid metric specification: {spec}")
        if name in result:
            raise ValueError(f"duplicate metric specification: {name}")
        result[name] = direction
    if len(result) < 2:
        raise ValueError("Pareto analysis requires at least two metrics")
    return result


def pareto_front(units: pd.DataFrame, metric_directions: dict[str, str]) -> pd.DataFrame:
    """Return complete units with normalized scores and a frontier flag."""

    validate_units(units)
    requested = list(metric_directions)
    available = set(units["metric"].astype(str))
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"units table does not contain metrics: {missing}")

    selected = units[units["metric"].isin(requested)].copy()
    wide = selected.pivot(index=UNIT_ID, columns="metric", values="mean")
    wide = wide.reindex(columns=requested).dropna().reset_index()
    if wide.empty:
        raise ValueError("no experiment unit has all requested Pareto metrics")

    score_columns: list[str] = []
    for metric, direction in metric_directions.items():
        values = wide[metric].astype(float)
        minimum, maximum = float(values.min()), float(values.max())
        score_name = f"score__{metric}"
        if np.isclose(maximum, minimum):
            score = pd.Series(1.0, index=values.index)
        elif direction == "max":
            score = (values - minimum) / (maximum - minimum)
        else:
            score = (maximum - values) / (maximum - minimum)
        wide[score_name] = score
        score_columns.append(score_name)

    scores = wide[score_columns].to_numpy(dtype=float)
    dominated = np.zeros(len(wide), dtype=bool)
    for index, candidate in enumerate(scores):
        no_worse = np.all(scores >= candidate, axis=1)
        strictly_better = np.any(scores > candidate, axis=1)
        dominated[index] = bool(np.any(no_worse & strictly_better))
    wide["is_pareto"] = ~dominated
    wide["pareto_score_mean"] = wide[score_columns].mean(axis=1)
    return wide.sort_values(
        ["is_pareto", "pareto_score_mean"], ascending=[False, False]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--metric", action="append", required=True, help="NAME:min or NAME:max"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    units = read_csv(args.input, "units")
    result = pareto_front(units, parse_metric_specs(args.metric))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"complete_units={len(result)}, pareto_units={int(result['is_pareto'].sum())}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
