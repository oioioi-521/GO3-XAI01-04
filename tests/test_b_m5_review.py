"""Selection behavior on artificial scalar fixtures, not formal image evidence."""
import pandas as pd
import pytest

from analysis.b_m5_review import METRICS, METHODS, select_cases


def fixture():
    rows = []
    for image in ["10", "2", "30"]:
        for metric in METRICS:
            for i, method in enumerate(METHODS):
                rows.append(dict(image_id=image, dataset="voc", model="resnet50", method=method,
                                 metric=metric, value=str(i + 1)))
    predictions = pd.DataFrame([dict(image_id=i, dataset="voc", model="resnet50", confidence=c)
                                for i, c in [("10", "0.9"), ("2", "0.9"), ("30", "0.6")]])
    cohorts = predictions.drop(columns="confidence").assign(model_correct="1", common_correct="1")
    trace = pd.DataFrame(columns=["image_id", "dataset", "model", "method", "status"])
    return pd.DataFrame(rows), predictions, trace, cohorts


def test_numeric_ties_and_disjoint_selection_survive_row_permutation():
    args = fixture()
    selected, _ = select_cases(*args)
    assert [r["image_id"] for r in selected] == ["2", "10", "30"]
    shuffled = tuple(frame.sample(frac=1, random_state=7) for frame in args)
    assert select_cases(*shuffled)[0] == selected


def test_incomplete_six_method_matrix_rejected():
    per, *others = fixture()
    with pytest.raises(ValueError, match="incomplete"):
        select_cases(per.iloc[1:], *others)


def test_degenerate_candidate_requires_hit_cohort_and_counts_repeats():
    per, pred, trace, cohorts = fixture()
    cohorts = pd.concat([cohorts, pd.DataFrame([
        dict(image_id="7", dataset="voc", model="vgg16", model_correct="1", common_correct="0"),
        dict(image_id="1", dataset="voc", model="densenet121", model_correct="0", common_correct="0")])])
    trace = pd.DataFrame([dict(image_id=i, dataset="voc", model=m, method="gradcam", status="both_constant")
                          for i, m in [("7", "vgg16"), ("1", "densenet121")] for _ in range(5)])
    selected, _ = select_cases(per, pred, trace, cohorts)
    assert selected[-1] == dict(role="constant_gradcam", dataset="voc", model="vgg16", image_id="7", invalid_repeats=5)
