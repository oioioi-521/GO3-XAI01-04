"""Scope and dominance checks for the two midterm D plots."""

from pathlib import Path

import pandas as pd
import pytest

from analysis.plot_midterm_d import (
    DATASETS, EFFICIENCY, FAITHFULNESS, MODELS, _paired_pareto, load_points,
)


ROOT = Path(__file__).resolve().parents[1]


def test_midterm_inputs_require_three_complete_b_units(tmp_path):
    b_path = tmp_path / "b.csv"
    b_rows = [{"method": "occlusion", "model": model, "dataset": "imagenet",
               "metric": FAITHFULNESS, "mean": 0.2, "std": 0.1, "n": 460,
               "config_hash": "0123456789ab"} for model in MODELS]
    pd.DataFrame(b_rows[:-1]).to_csv(b_path, index=False)
    with pytest.raises(ValueError, match="missing, extra, or duplicated"):
        load_points(ROOT / "analysis/a_ig_gradcam_summary.csv", b_path,
                    ROOT / "analysis/midterm_rise_imagenet.json")
    pd.DataFrame(b_rows).to_csv(b_path, index=False)
    four, a = load_points(ROOT / "analysis/a_ig_gradcam_summary.csv", b_path,
                          ROOT / "analysis/midterm_rise_imagenet.json")
    assert len(four) == 12
    assert len(a) == 24
    assert set(four["dataset"]) == {"imagenet"}


def test_pareto_is_within_model_dataset_and_detects_dominance():
    a = pd.read_csv(ROOT / "analysis/a_ig_gradcam_summary.csv")
    points = _paired_pareto(a)
    assert len(points) == 12
    assert points["within_pair_pareto"].all()
    # If one method becomes worse on both objectives in one unit, it leaves
    # only that unit's local frontier; other model/dataset pairs stay intact.
    changed = a.copy()
    mask = ((changed["method"] == "ig") & (changed["model"] == "resnet50")
            & (changed["dataset"] == "imagenet"))
    changed.loc[mask & (changed["metric"] == FAITHFULNESS), "mean"] = 0.5
    changed.loc[mask & (changed["metric"] == EFFICIENCY), "mean"] = 500.0
    changed_points = _paired_pareto(changed)
    dominated = changed_points[(changed_points["method"] == "ig")
                               & (changed_points["model"] == "resnet50")
                               & (changed_points["dataset"] == "imagenet")]
    assert not bool(dominated.iloc[0]["within_pair_pareto"])
    assert int(changed_points["within_pair_pareto"].sum()) == 11
