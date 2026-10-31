"""Tampering checks for the independent KernelSHAP formal gate."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from analysis.validate_kernelshap_formal import (
    FormalValidationError,
    _check_tables,
    _check_trace,
    _seed,
)


def _fixture():
    targets = {"image-a": 1, "image-b": 2}
    trace = []
    per_image = []
    values = {
        "efficiency_time_ms": [10.0, 20.0],
        "faithfulness_morf_auc_raw": [0.1, 0.2],
        "stability_spearman": [0.8, 0.8],
        "stability_valid_rate": [1.0, 1.0],
    }
    for index, (image_id, target) in enumerate(targets.items()):
        for repeat in range(5):
            trace.append({
                "image_id": image_id, "dataset": "imagenet", "model": "resnet50",
                "method": "kernelshap", "repeat": str(repeat),
                "seed": str(_seed("imagenet", image_id, repeat)),
                "target": str(target), "protocol_version": "rgb-gaussian-spearman-v1",
                "config_hash": "stability", "sigma": "0.005", "status": "valid",
                "spearman": "0.8", "score": "0.8", "top10_jaccard": "0.5",
                "original_pred": str(target), "perturbed_pred": str(target),
                "prediction_preserved": "True", "original_target_score": "0.8",
                "perturbed_target_score": "0.75", "target_score_abs_delta": "0.05",
                "perturbation_mae_pixel": "0.004", "attribution_time_ms": "5.0",
            })
        for metric, metric_values in values.items():
            per_image.append({
                "image_id": image_id, "dataset": "imagenet", "model": "resnet50",
                "method": "kernelshap", "metric": metric,
                "value": str(metric_values[index]),
                "time_ms": str(metric_values[index]) if metric == "efficiency_time_ms" else "",
            })
    units = []
    for metric, metric_values in values.items():
        units.append({
            "dataset": "imagenet", "model": "resnet50", "method": "kernelshap",
            "metric": metric, "mean": str(np.mean(metric_values)),
            "std": str(np.std(metric_values, ddof=1)), "n": "2",
            "config_hash": "base" if metric in
            {"efficiency_time_ms", "faithfulness_morf_auc_raw"} else "stability",
        })
    return targets, trace, per_image, units


def test_trace_requires_paired_seeds_and_unique_repeats():
    targets, trace, _, _ = _fixture()
    means, rates, statuses = _check_trace(trace, "resnet50", "imagenet", targets, "stability")
    assert means == {"image-a": 0.8, "image-b": 0.8}
    assert rates == {"image-a": 1.0, "image-b": 1.0}
    assert statuses == {"valid": 10}

    altered = copy.deepcopy(trace)
    altered[0]["seed"] = "0"
    with pytest.raises(FormalValidationError, match="paired seed"):
        _check_trace(altered, "resnet50", "imagenet", targets, "stability")

    altered = copy.deepcopy(trace)
    altered[1]["repeat"] = "0"
    with pytest.raises(FormalValidationError, match="duplicate trace key"):
        _check_trace(altered, "resnet50", "imagenet", targets, "stability")


def test_recomputed_metric_and_unit_summaries_reject_tampering():
    targets, trace, per_image, units = _fixture()
    means, rates, _ = _check_trace(trace, "resnet50", "imagenet", targets, "stability")
    summary = _check_tables(per_image, units, "resnet50", "imagenet", targets,
                            "base", "stability", means, rates)
    assert summary["stability_spearman"]["n"] == 2

    altered = copy.deepcopy(per_image)
    next(row for row in altered if row["metric"] == "stability_spearman")["value"] = "0.7"
    with pytest.raises(FormalValidationError, match="stability mean"):
        _check_tables(altered, units, "resnet50", "imagenet", targets,
                      "base", "stability", means, rates)

    altered_units = copy.deepcopy(units)
    next(row for row in altered_units if row["metric"] == "stability_spearman")["std"] = "0.1"
    with pytest.raises(FormalValidationError, match="sample std"):
        _check_tables(per_image, altered_units, "resnet50", "imagenet", targets,
                      "base", "stability", means, rates)
