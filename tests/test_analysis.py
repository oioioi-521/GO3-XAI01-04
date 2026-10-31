from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.aggregate import aggregate_per_image
from analysis.anova import run_factorial_anova, run_friedman_by_method
from analysis.correlation import metric_correlations
from analysis.pareto import pareto_front
from analysis.schema import ResultValidationError, validate_per_image


def make_per_image() -> pd.DataFrame:
    rows = []
    methods = ["kernelshap", "rise", "lime"]
    models = ["resnet50", "vgg16"]
    datasets = ["imagenet", "voc"]
    for image_index in range(4):
        for method_index, method in enumerate(methods):
            for model_index, model in enumerate(models):
                for dataset_index, dataset in enumerate(datasets):
                    base = method_index + model_index * 0.2 + dataset_index * 0.1
                    noise = image_index * 0.01 + (image_index % 2) * method_index * 0.005
                    common = {
                        "image_id": f"img-{image_index}",
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                    }
                    rows.append(
                        {
                            **common,
                            "metric": "faithfulness_morf_auc",
                            "value": base + noise,
                            "time_ms": "",
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "metric": "efficiency_time_ms",
                            "value": (method_index + 1) * 100 + image_index,
                            "time_ms": (method_index + 1) * 100 + image_index,
                        }
                    )
    return pd.DataFrame(rows)


def test_validation_rejects_duplicate_observation() -> None:
    frame = make_per_image()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ResultValidationError, match="duplicate"):
        validate_per_image(duplicated)


def test_aggregate_preserves_runner_hashes() -> None:
    frame = make_per_image()
    initial = aggregate_per_image(frame)
    initial["config_hash"] = [f"hash-{index}" for index in range(len(initial))]
    result = aggregate_per_image(frame, initial)
    assert list(result.columns) == [
        "method",
        "model",
        "dataset",
        "metric",
        "mean",
        "std",
        "n",
        "config_hash",
    ]
    assert result["config_hash"].str.startswith("hash-").all()
    assert set(result["n"]) == {4}


def test_factorial_anova_and_friedman_run_on_complete_fixture() -> None:
    frame = make_per_image()
    table, diagnostics = run_factorial_anova(frame, "faithfulness_morf_auc")
    assert "partial_eta_sq" in table.columns
    assert "C(method)" in set(table["effect"])
    assert diagnostics["rows"] == 48
    friedman = run_friedman_by_method(frame, "faithfulness_morf_auc")
    assert friedman["status"] == "ok"
    assert friedman["complete_blocks"] == 16


def test_correlations_use_only_matched_metric_pairs() -> None:
    pearson, spearman, pairs = metric_correlations(make_per_image())
    assert pearson.shape == (2, 2)
    assert spearman.shape == (2, 2)
    assert pairs.iloc[0]["n"] == 48


def test_pareto_marks_dominated_unit() -> None:
    rows = []
    values = {
        "fast": {"faith": 0.5, "stability": 0.2, "time": 10.0},
        "faithful": {"faith": 0.1, "stability": 0.4, "time": 30.0},
        "dominated": {"faith": 0.6, "stability": 0.5, "time": 40.0},
    }
    for method, metrics in values.items():
        for metric, mean in metrics.items():
            rows.append(
                {
                    "method": method,
                    "model": "resnet50",
                    "dataset": "imagenet",
                    "metric": metric,
                    "mean": mean,
                    "std": 0.01,
                    "n": 10,
                    "config_hash": "fixture",
                }
            )
    result = pareto_front(
        pd.DataFrame(rows), {"faith": "min", "stability": "min", "time": "min"}
    ).set_index("method")
    assert bool(result.loc["fast", "is_pareto"])
    assert bool(result.loc["faithful", "is_pareto"])
    assert not bool(result.loc["dominated", "is_pareto"])
    assert np.isfinite(result["pareto_score_mean"]).all()
