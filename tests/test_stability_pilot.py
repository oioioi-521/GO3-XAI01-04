from __future__ import annotations

import numpy as np
import torch

from experiments.stability import perturb_rgb, stability_similarity, stable_seed
from preprocessing.dataset import MEAN, STD


def _normalized_midgray() -> torch.Tensor:
    pixels = torch.full((1, 3, 4, 4), 0.5)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def test_seed_is_stable_and_repeat_specific() -> None:
    first = stable_seed("imagenet", "image-1", 0)
    assert first == stable_seed("imagenet", "image-1", 0)
    assert first != stable_seed("imagenet", "image-1", 1)
    assert first != stable_seed("voc", "image-1", 0)


def test_rgb_perturbation_is_deterministic() -> None:
    image = _normalized_midgray()
    first, first_mae = perturb_rgb(image, sigma=0.02, seed=42)
    second, second_mae = perturb_rgb(image, sigma=0.02, seed=42)
    different, _ = perturb_rgb(image, sigma=0.02, seed=43)
    assert torch.equal(first, second)
    assert first_mae == second_mae
    assert not torch.equal(first, different)
    assert 0 < first_mae < 0.02


def test_similarity_reports_valid_and_constant_maps() -> None:
    gradient = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = stability_similarity(gradient, gradient.copy())
    assert result.spearman == 1.0
    assert result.score == 1.0
    assert result.top_jaccard == 1.0
    assert result.status == "valid"

    constant = np.zeros((4, 4), dtype=np.float32)
    result = stability_similarity(constant, gradient)
    assert np.isnan(result.spearman)
    assert result.score == 0.0
    assert np.isnan(result.top_jaccard)
    assert result.status == "candidate_constant"
