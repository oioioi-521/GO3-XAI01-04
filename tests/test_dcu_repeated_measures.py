"""Known-reference and synthetic tests, never substituted for formal observations."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.linalg import helmert

from analysis.dcu_repeated_measures import QUALITY, analyze, effect_coordinates, repeated_cube, rm_effect
from analysis.validate_dcu_team_handoff import METHODS, MODELS


def test_stata_published_two_repeated_factors():
    # StataCorp FAQ: Winer et al. table 7.13, noise==1, three subjects.
    # https://www.stata.com/support/faqs/statistics/repeated-measures-anova/
    # Published table is period-major/dial-minor: subject, period, dial.
    y = np.array([[45, 53, 60, 40, 52, 57, 28, 37, 46],
                  [35, 41, 50, 30, 37, 47, 25, 32, 41],
                  [60, 65, 75, 58, 54, 70, 40, 47, 50]], float).reshape(3, 3, 3)
    result = {k: rm_effect(z) for k, z in effect_coordinates(y).items()}
    for effect, ss, error, eps, p in [
        ("method", 1020.6666667, 39.7777778, .5481, .0147),
        ("model", 1124.6666667, 123.7777778, .5419, .0441),
        ("method_model", 9.3333333, 54.2222222, .2532, .6187),
    ]:
        r = result[effect]
        assert r["ss_effect"] == pytest.approx(ss, abs=1e-6)
        assert r["ss_error"] == pytest.approx(error, abs=1e-6)
        assert r["epsilon_gg"] == pytest.approx(eps, abs=5e-5)
        assert r["p_value"] == pytest.approx(p, abs=5e-5)


def test_two_level_effect_matches_paired_t_squared():
    rng = np.random.default_rng(27)
    y = rng.normal(size=(40, 2)) + [0, .7]
    z = y @ helmert(2).T
    r = rm_effect(z)
    t = stats.ttest_rel(y[:, 0], y[:, 1])
    assert r["F"] == pytest.approx(t.statistic ** 2)
    assert r["p_value"] == pytest.approx(t.pvalue)
    assert r["epsilon_gg"] == 1


def test_additive_data_have_no_interaction():
    y = np.arange(20.)[:, None, None] + np.arange(3.)[None, :, None] + np.arange(6.)[None, None, :]
    r = rm_effect(effect_coordinates(y)["method_model"])
    assert r["status"] == "all_zero" and r["p_value"] == 1
    assert r["F"] is None


def test_nonzero_constant_has_explicit_unestimable_error():
    r = rm_effect(np.ones((20, 5)))
    assert r["status"] == "zero_error_variance" and r["p_value"] is None


def test_level_permutation_and_subject_intercepts_preserve_tests():
    rng = np.random.default_rng(123)
    y = rng.normal(size=(30, 3, 6))
    original = {k: rm_effect(z) for k, z in effect_coordinates(y).items()}
    altered = y[:, [2, 0, 1]][:, :, [4, 1, 3, 2, 5, 0]] + rng.normal(size=(30, 1, 1)) * 100
    for k, z in effect_coordinates(altered).items():
        other = rm_effect(z)
        for field in ("ss_effect", "ss_error", "F", "epsilon_gg", "p_value"):
            assert other[field] == pytest.approx(original[k][field], abs=1e-10)


@pytest.mark.parametrize("y", [np.ones((2, 3, 6)), np.ones((4, 1, 6)), np.full((5, 3, 6), np.nan)])
def test_invalid_arrays_rejected(y):
    with pytest.raises(ValueError):
        effect_coordinates(y)


def synthetic_frames():
    rng, rows, labels = np.random.default_rng(48), [], {}
    for dataset in ("imagenet", "voc"):
        for i in range(12):
            labels[dataset, str(i)] = str(i % 3)
            for k, model in enumerate(MODELS):
                for m, method in enumerate(METHODS):
                    for j, metric in enumerate(QUALITY):
                        rows.append(dict(image_id=str(i), dataset=dataset, model=model, method=method,
                                         metric=metric, value=float(rng.normal() + k * m * .4)))
    frame = pd.DataFrame(rows)
    return {"common_correct": frame, "all": frame.copy()}, pd.Series(labels)


def test_duplicate_and_missing_cells_rejected():
    frames, _ = synthetic_frames()
    frame = frames["all"]
    with pytest.raises(ValueError, match="duplicate"):
        repeated_cube(pd.concat([frame, frame.iloc[:1]]), "imagenet")
    with pytest.raises(ValueError, match="incomplete"):
        repeated_cube(frame.iloc[1:], "imagenet")


def test_complete_output_families_and_reproducibility():
    frames, classes = synthetic_frames()
    data = analyze(frames, classes, resamples=40)
    assert len(data["omnibus"]) == 24
    assert len(data["direct_did"]) == 360
    assert len(data["cell_means"]) == len(data["centered_interactions"]) == 144
    assert all(r["family_size"] == 12 for r in data["omnibus"])
    assert all(r["family_size"] == 180 for r in data["direct_did"])
    for r in data["omnibus"]:
        assert r["p_holm"] >= r["p_value"]
        assert 0 <= r["partial_eta_squared"] <= 1
    assert data == analyze(frames, classes, resamples=40)
