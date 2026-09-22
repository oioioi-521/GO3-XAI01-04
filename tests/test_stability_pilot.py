from __future__ import annotations

import numpy as np
import torch

from analysis.pilot_stability import _perturb_rgb, _similarity, _stable_seed
from preprocessing.dataset import MEAN, STD


def _normalized_midgray() -> torch.Tensor:
    pixels = torch.full((1, 3, 4, 4), 0.5)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (pixels - mean) / std


def test_seed_is_stable_and_repeat_specific() -> None:
    first = _stable_seed("imagenet", "image-1", 0)
    assert first == _stable_seed("imagenet", "image-1", 0)
    assert first != _stable_seed("imagenet", "image-1", 1)
    assert first != _stable_seed("voc", "image-1", 0)


def test_rgb_perturbation_is_deterministic() -> None:
    image = _normalized_midgray()
    first, first_mae = _perturb_rgb(image, sigma=0.02, seed=42)
    second, second_mae = _perturb_rgb(image, sigma=0.02, seed=42)
    different, _ = _perturb_rgb(image, sigma=0.02, seed=43)
    assert torch.equal(first, second)
    assert first_mae == second_mae
    assert not torch.equal(first, different)
    assert 0 < first_mae < 0.02


def test_similarity_reports_valid_and_constant_maps() -> None:
    gradient = np.arange(16, dtype=np.float32).reshape(4, 4)
    correlation, overlap, status = _similarity(gradient, gradient.copy())
    assert correlation == 1.0
    assert overlap == 1.0
    assert status == "valid"

    constant = np.zeros((4, 4), dtype=np.float32)
    correlation, overlap, status = _similarity(constant, gradient)
    assert np.isnan(correlation)
    assert np.isnan(overlap)
    assert status == "candidate_constant"
