"""Numerical regression tests only; synthetic arrays are not course results."""
import itertools
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from analysis.dcu_statistics import (
    adjust_families, friedman, holm, interval, method_matrix, paired_test,
    pareto_flags, stratified_boot_means,
)
from analysis.validate_dcu_team_handoff import ALL_METRICS, METHODS, check_aggregates, object_hash, paired_seed, validate
from analysis.plot_dcu_statistics import effect_grid


def test_holm_known_values_and_family_scope():
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])
    rows = [dict(family="a", p_value=.01), dict(family="a", p_value=.04),
            dict(family="b", p_value=.03)]
    adjust_families(rows)
    assert [r["p_holm"] for r in rows] == pytest.approx([.02, .04, .03])
    assert [r["family_size"] for r in rows] == [2, 2, 1]
    with pytest.raises(ValueError):
        holm([float("nan")])


def test_all_tied_pairs_are_not_significant():
    result = paired_test(np.arange(20), np.arange(20), -1)
    assert result["p_value"] == result["sign_p_value"] == 1
    assert result["rank_biserial_a"] == result["cohen_dz_a"] == 0
    assert result["ties"] == 20 and result["paired_superiority_a"] == .5


def test_pair_direction_and_swap():
    a, b = np.arange(20.), np.arange(20.) + np.linspace(1, 5, 20)
    first, swap = paired_test(a, b, -1), paired_test(b, a, -1)
    assert first["mean_advantage_a"] > 0 and first["rank_biserial_a"] == 1
    assert first["mean_advantage_a"] == -swap["mean_advantage_a"]
    assert first["rank_biserial_a"] == -swap["rank_biserial_a"]
    assert first["p_value"] == swap["p_value"]


def test_zeros_preserved_and_nonfinite_rejected():
    result = paired_test(np.r_[np.ones(30), 0, 0], np.zeros(32), 1)
    assert result["wins_a"] == 30 and result["ties"] == 2
    assert result["cohen_dz_a"] is not None
    constant = paired_test(np.ones(20), np.zeros(20), 1)
    assert constant["cohen_dz_a"] is None  # Not arbitrary infinity/JSON NaN.
    with pytest.raises(ValueError):
        paired_test([1, np.nan], [0, 0], 1)


def test_bootstrap_preserves_pairs_and_class_counts():
    values = np.array([[0., 10.], [0., 10.], [9., 19.]])
    labels = ["a", "a", "b"]
    boot = stratified_boot_means(values, labels, 100, 1)
    assert np.allclose(boot, [3., 13.])  # Class counts stay 2:1.
    assert np.allclose(boot[:, 1] - boot[:, 0], 10)
    assert interval(boot[:, 1] - boot[:, 0]) == pytest.approx([10, 10])
    assert np.array_equal(boot, stratified_boot_means(values, labels, 100, 1))


def test_bootstrap_reproducible_with_variation():
    x = np.arange(20.).reshape(10, 2)
    first = stratified_boot_means(x, ["a"] * 10, 100, 7)
    assert first[:, 0].std() > 0
    assert np.array_equal(first, stratified_boot_means(x, ["a"] * 10, 100, 7))
    with pytest.raises(ValueError):
        stratified_boot_means(x, ["a"], 100, 7)


def test_raw_pareto_keeps_equal_candidates_and_batch_axis():
    costs = np.array([[0., 1.], [1., 0.], [2., 2.], [0., 1.]])
    expected = [True, True, False, True]
    assert pareto_flags(costs).tolist() == expected
    assert pareto_flags(np.stack([costs, costs])).tolist() == [expected, expected]
    with pytest.raises(ValueError):
        pareto_flags([[np.nan, 1]])


def test_friedman_all_tied_and_strong_order():
    tied = friedman(np.ones((30, 3)), ["a", "b", "c"], -1, 99, 7)
    assert tied["p_value"] == 1 and tied["kendall_w"] == 0
    x = np.tile(np.arange(6.), (30, 1))
    result = friedman(x, list("abcdef"), -1, 99, 7)
    assert result["p_value"] == .01 and result["kendall_w"] == pytest.approx(1.)
    assert result["mean_ranks_best_is_1"]["a"] == 1
    assert result == friedman(x, list("abcdef"), -1, 99, 7)


def test_method_matrix_aligns_ids_not_row_order_and_rejects_missing():
    rows = [dict(image_id=i, dataset="voc", model="resnet50", method=m, metric=metric,
                 value=float(j)) for i in ["b", "a"] for m in METHODS
            for j, metric in enumerate(ALL_METRICS)]
    frame = pd.DataFrame(rows).sample(frac=1, random_state=42)
    wide = method_matrix(frame, "voc", "resnet50")
    assert wide.index.tolist() == ["a", "b"]
    assert wide.columns.tolist() == list(itertools.product(METHODS, ALL_METRICS))
    with pytest.raises(ValueError, match="incomplete"):
        method_matrix(frame.iloc[1:], "voc", "resnet50")
    with pytest.raises(ValueError, match="duplicate"):
        method_matrix(pd.concat([frame, frame.iloc[:1]]), "voc", "resnet50")


def test_receipt_aggregates_use_sample_std_and_reject_duplicates():
    frame = pd.DataFrame([dict(image_id=str(i), dataset="voc", model="resnet50",
        method="ig", metric="m", value=float(i)) for i in [1, 2, 3]])
    summary = pd.DataFrame([dict(dataset="voc", model="resnet50", method="ig",
        metric="m", mean=2., std=1., n=3)])
    check_aggregates(frame, summary)
    with pytest.raises(ValueError, match="duplicate"):
        check_aggregates(pd.concat([frame, frame.iloc[:1]]), summary)
    summary["std"] = np.sqrt(2 / 3)
    with pytest.raises(ValueError, match="std"):
        check_aggregates(frame, summary)


def test_paired_seed_known_value_and_cross_method_independence():
    assert paired_seed("imagenet", "ILSVRC2012_val_00000025", 0) == "6152198569299538860"
    assert paired_seed("imagenet", "a", 0) != paired_seed("imagenet", "a", 1)


def test_bad_zip_sha_and_traversal_rejected_without_execution(tmp_path):
    path = tmp_path / "bad.zip"
    with ZipFile(path, "w") as z:
        z.writestr("team-handoff-20261014/../outside.py", "raise RuntimeError('must never execute')")
    with pytest.raises(ValueError, match="SHA"):
        validate(path, tmp_path)
    with pytest.raises(ValueError, match="unsafe"):
        validate(path, tmp_path, expected_sha=None)


def test_config_identity_order_independent_but_value_sensitive():
    assert object_hash({"a": 1, "b": [2, 3]}) == object_hash({"b": [2, 3], "a": 1})
    assert object_hash({"a": 1}) != object_hash({"a": 2})


def test_effect_grid_preserves_row_orientation_and_family_significance():
    rows = [dict(cohort="model_correct", dataset="voc", model="resnet50", metric="m",
                 a=a, b=b, rank_biserial_a=.7, p_holm=.01)
            for a, b in itertools.combinations(METHODS, 2)]
    grid, sig = effect_grid(rows, "voc", "resnet50", "m")
    assert np.array_equal(grid, -grid.T)
    assert grid[0, 5] == .7 and sig[0, 5] and not sig[0, 0]
    with pytest.raises(ValueError, match="incomplete"):
        effect_grid(rows[:-1], "voc", "resnet50", "m")
    with pytest.raises(ValueError, match="duplicate"):
        effect_grid(rows[:-1] + rows[:1], "voc", "resnet50", "m")


def test_checked_in_derived_tables_have_finite_complete_summary_scope():
    # Checks the published derived artifact, not independent model execution.
    path = Path(__file__).resolve().parents[1] / "analysis/dcu_20261015/statistics.json"
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda v: pytest.fail(v))
    counts = {"summaries": 432, "method_omnibus": 54, "method_pairs": 810,
              "pareto": 108, "correlations": 324, "model_omnibus": 36, "model_pairs": 108}
    assert all(len(data[k]) == n for k, n in counts.items())
    assert data["receipt"]["config_snapshots_checked"] == 90
    for r in data["method_pairs"]:
        assert r["wins_a"] + r["losses_a"] + r["ties"] == r["n"]
        assert r["family_size"] == (90 if r["metric"] == "efficiency_time_ms" else 180)
        assert 0 <= r["p_value"] <= r["p_holm"] <= 1
    for r in data["summaries"]:
        assert r["std"] >= 0 and r["mean_ci95"][0] <= r["mean_ci95"][1]
        assert all(np.isfinite(v) for v in [r["mean"], r["std"], *r["mean_ci95"]])
