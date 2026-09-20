import pytest
import torch

from experiments.metrics import faithfulness_morf_auc, raw_true_target_deletion_auc, summarize_by_correct


class Scores(torch.nn.Module):
    def forward(self, image):
        value = image[:, 0].mean((1, 2))
        return torch.stack((-value - 10, value), 1)


@pytest.mark.parametrize("activation", ["softmax", "sigmoid"])
def test_raw_deletion_auc_is_finite_and_bounded(activation):
    image = torch.ones(1, 3, 4, 4); baseline = torch.zeros_like(image); attr = torch.ones(4, 4)
    score = raw_true_target_deletion_auc(Scores(), image, 0, attr, baseline, [0, .5, 1], activation)
    assert 0 <= score <= 1


def test_ratio_can_exceed_one_for_low_original_probability():
    image = torch.ones(1, 3, 4, 4); baseline = torch.zeros_like(image); attr = torch.ones(4, 4)
    assert faithfulness_morf_auc(Scores(), image, 0, attr, baseline, [0, .5, 1]) >= 1


def test_group_summary_keeps_incorrect_rows():
    summary = summarize_by_correct([.1, .8, .2], [True, False, True])
    assert summary["correct"] == {"n": 2, "mean": pytest.approx(.15)}
    assert summary["incorrect"] == {"n": 1, "mean": pytest.approx(.8)}
