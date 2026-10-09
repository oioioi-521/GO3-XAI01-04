"""Completeness checks for the KernelSHAP formal summary figure."""

import pandas as pd
import pytest

from analysis.plot_kernelshap_formal import DATASETS, METRICS, MODELS, render, validated_summary


def _unit_rows():
    return [
        {"method": "kernelshap", "model": model, "dataset": dataset,
         "metric": metric, "mean": 1.0, "std": 0.1, "n": 460,
         "config_hash": "0123456789ab"}
        for dataset in DATASETS for model in MODELS for metric in METRICS
    ]


def test_plot_requires_all_six_units(tmp_path):
    path = tmp_path / "units.csv"
    rows = _unit_rows()
    pd.DataFrame(rows[:-1]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="exactly four formal metrics"):
        validated_summary(path)
    rows[-1]["n"] = 40
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="n=460"):
        validated_summary(path)


def test_plot_renders_complete_summary(tmp_path):
    path = tmp_path / "units.csv"
    output = tmp_path / "formal.png"
    pd.DataFrame(_unit_rows()).to_csv(path, index=False)
    render(path, output)
    assert output.is_file() and output.stat().st_size > 10000
