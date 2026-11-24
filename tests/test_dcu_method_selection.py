"""Independent small examples for selection direction, eligibility and leakage."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from analysis.dcu_method_selection import (
    FEATURES, METRIC, METHODS, OBJECTIVES, PROFILES, PROTOCOL_PATH,
    baseline_choices, bootstrap_difference, export_tree, features, fit_policy,
    grouped_split, leaf_rows, loss_table, oracle, plot_outputs, predict_export, prepare_cases, selection_scores,
)


def synthetic_per(n=40):
    rows = []
    for dataset in ("imagenet", "voc"):
        for model in ("resnet50", "densenet121", "vgg16"):
            for image in range(n):
                for j, method in enumerate(METHODS):
                    for objective, metric in METRIC.items():
                        # Full-access: ig best faithfulness, rise best stability,
                        # gradcam fastest. No-gradient: occlusion best fidelity/time.
                        value = (1. if j == 3 else .5) if objective == "stability" else (float(j) if objective == "faithfulness" else float((j - 1) % 6))
                        rows.append(dict(dataset=dataset, model=model, image_id=str(image),
                                         method=method, metric=metric, value=value))
    return pd.DataFrame(rows)


def test_objective_direction_and_native_gap():
    values = np.array([[.8, .3, .5], [.4, .4, .6]])
    names = ("a", "b", "c")
    assert oracle(loss_table(values, "faithfulness"), names)[0].tolist() == ["b", "a"]
    assert oracle(loss_table(values, "stability"), names)[0].tolist() == ["a", "c"]
    scores = selection_scores(loss_table(values, "faithfulness"), ["c", "c"], names)
    assert scores["rank_regret"].tolist() == [.5, 1.]
    assert scores["native_gap"] == pytest.approx([.2, .2])


def test_ties_have_no_arbitrary_evaluation_penalty():
    loss = np.array([[0., 5e-13, 1.], [4., 4., 4.]])
    names = ("a", "b", "c")
    label, ties = oracle(loss, names)
    assert label.tolist() == ["a", "a"]
    assert ties.sum(axis=1).tolist() == [2, 3]
    scores = selection_scores(loss, ["b", "c"], names)
    assert scores["hit"].tolist() == [1., 1.]
    assert scores["rank_regret"].tolist() == [0., 0.]
    assert scores["native_gap"].tolist() == [0., 0.]


@pytest.mark.parametrize("bad", [np.array([[np.nan, 1.]]), np.array([[np.inf, 1.]])])
def test_bad_metrics_rejected(bad):
    with pytest.raises(ValueError):
        loss_table(bad, "efficiency")
    with pytest.raises(ValueError):
        selection_scores(bad, ["a"], ("a", "b"))


def test_interface_restriction_and_pair_coverage():
    cases = prepare_cases(synthetic_per(2))
    full = cases[cases.profile.eq("full")]
    restricted = cases[cases.profile.eq("no_grad_logits")]
    assert set(full[full.objective.eq("faithfulness")].label) == {"ig"}
    assert set(restricted[restricted.objective.eq("faithfulness")].label) == {"occlusion"}
    assert not set(restricted.label) & {"ig", "gradcam"}
    assert len(cases) == 2 * 3 * 2 * 3 * 2


def test_missing_or_duplicate_method_cannot_silently_change_labels():
    per = synthetic_per(1)
    with pytest.raises(ValueError, match="incomplete"):
        prepare_cases(per.drop(index=0))
    with pytest.raises(ValueError, match="duplicate"):
        prepare_cases(pd.concat([per, per.iloc[:1]]))


def test_split_keeps_every_model_priority_and_interface_of_image_together():
    cases = prepare_cases(synthetic_per())
    tr, te = grouped_split(cases)
    train, test = cases.iloc[tr], cases.iloc[te]
    assert set(train.group).isdisjoint(test.group)
    assert len(train) + len(test) == len(cases)
    assert cases.groupby(["dataset", "image_id"]).group.nunique().eq(1).all()
    assert cases[cases.image_id.eq("0")].group.nunique() == 2
    assert np.array_equal(te, grouped_split(cases)[1])


def test_outcomes_identifiers_and_correctness_are_not_features():
    cases = prepare_cases(synthetic_per(1))
    altered = cases.copy()
    altered["label"] = "wrong"
    altered["image_id"], altered["group"] = "new-id", "new-group"
    for name in METHODS:
        altered[name] = 1e8
    assert len(FEATURES) == 5
    assert np.array_equal(features(cases), features(altered))
    altered["dataset"] = "future-task"
    with pytest.raises(ValueError, match="unknown dataset"):
        features(altered)


def test_rank_regret_is_unit_scale_invariant_and_bounds():
    loss = np.array([[1., 2., 3.], [2., 1., 3.]])
    a = selection_scores(loss, ["c", "a"], ("a", "b", "c"))
    b = selection_scores(loss * 1000 + 42, ["c", "a"], ("a", "b", "c"))
    assert a["rank_regret"].tolist() == [1., .5]
    assert np.array_equal(a["rank_regret"], b["rank_regret"])
    with pytest.raises(ValueError, match="unavailable"):
        selection_scores(loss, ["ig", "a"], ("a", "b", "c"))


def test_baselines_use_training_labels_and_training_ranks_only():
    cases = prepare_cases(synthetic_per(4))
    part = cases[cases.profile.eq("full")]
    mode, rank = baseline_choices(part, METHODS)
    assert mode == rank == dict(faithfulness="ig", stability="rise", efficiency="gradcam")
    changed = part.copy()
    changed.loc[changed.objective.eq("faithfulness"), "label"] = "lime"
    assert baseline_choices(changed, METHODS)[0]["faithfulness"] == "lime"
    assert baseline_choices(changed, METHODS)[1]["faithfulness"] == "ig"


def test_fitted_and_portable_policy_match_and_remain_eligible():
    cases = prepare_cases(synthetic_per(40))
    tr, te = grouped_split(cases)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for profile, names in PROFILES.items():
        tree = fit_policy(cases.iloc[tr], profile, protocol)
        held = cases.iloc[te]
        held = held[held.profile.eq(profile)]
        predicted = tree.predict(features(held))
        assert np.array_equal(predicted, predict_export(export_tree(tree), features(held)))
        assert set(predicted) <= set(names)
        assert tree.get_depth() <= 3
        assert (predicted == held.label).all()


def test_cluster_interval_known_constant_and_misalignment_rejected():
    rows = []
    for case in range(8):
        for policy in ("tree", "goal_mode", "goal_rank"):
            rows.append(dict(profile="full", objective="faithfulness", policy=policy,
                case_id=case, group=f"g{case // 2}", dataset="voc",
                hit=1. if policy == "tree" else 0., rank_regret=0. if policy == "tree" else .5))
    data = pd.DataFrame(rows)
    result = bootstrap_difference(data, 100)
    assert result[result.metric.eq("hit")].ci_low.tolist() == [1., 1.]
    assert result[result.metric.eq("rank_regret")].ci_high.tolist() == [-.5, -.5]
    assert result.n_original_images.eq(4).all()
    with pytest.raises(ValueError, match="identity"):
        bootstrap_difference(data.drop(index=1), 20)
    altered = data.copy()
    altered.loc[1, "group"] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        bootstrap_difference(altered, 20)


def test_plot_binary_questions_and_edge_labels_have_same_direction(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    cases = prepare_cases(synthetic_per(40))
    tr, te = grouped_split(cases)
    train, test = cases.iloc[tr], cases.iloc[te]
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    trees = {p: fit_policy(train, p, protocol) for p in PROFILES}
    leaves = pd.concat([leaf_rows(t, train, test, p, protocol) for p, t in trees.items()])
    metrics = pd.DataFrame([dict(profile=p, objective=o, policy=b, hit_rate=.5, mean_rank_regret=.2)
        for p in PROFILES for o in OBJECTIVES for b in ("tree", "goal_mode", "goal_rank")])
    checked = []
    def inspect(fig, path, **kwargs):
        if "method_selection_tree" in str(path):
            for ax in fig.axes:
                labels = [a.get_text().strip() for a in ax.texts]
                assert "no" in labels and "yes" in labels
                assert not {"True", "False"} & set(labels)
            checked.append(str(path))
    monkeypatch.setattr(Figure, "savefig", inspect)
    plot_outputs(trees, leaves, metrics, tmp_path)
    assert len(checked) == 2
