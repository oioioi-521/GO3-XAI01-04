"""Lightweight metrics used by the M2 pipeline demonstration."""

from __future__ import annotations

from typing import Iterable

import torch


@torch.inference_mode()
def faithfulness_morf_auc(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: int,
    attribution: torch.Tensor,
    baseline: torch.Tensor,
    fractions: Iterable[float],
) -> float:
    """Area under the MoRF deletion curve (lower is better).

    Pixels are replaced from most to least relevant. Values are normalized by
    the unperturbed target probability, which makes units easier to compare.
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if baseline.ndim == 3:
        baseline = baseline.unsqueeze(0)
    fractions = [float(value) for value in fractions]
    if not fractions or fractions[0] != 0.0 or fractions[-1] != 1.0:
        raise ValueError("deletion fractions must start at 0.0 and end at 1.0")
    if any(b <= a for a, b in zip(fractions, fractions[1:])):
        raise ValueError("deletion fractions must be strictly increasing")

    height, width = attribution.shape
    order = attribution.flatten().argsort(descending=True)
    original_probability = model(image).softmax(1)[0, target].clamp_min(1e-12)
    curve = []
    total = height * width
    for fraction in fractions:
        count = round(fraction * total)
        pixel_mask = torch.zeros(total, dtype=torch.bool, device=image.device)
        pixel_mask[order[:count].to(image.device)] = True
        pixel_mask = pixel_mask.view(1, 1, height, width)
        perturbed = torch.where(pixel_mask, baseline, image)
        probability = model(perturbed).softmax(1)[0, target]
        curve.append(probability / original_probability)
    x = torch.tensor(fractions, device=image.device)
    y = torch.stack(curve)
    return float(torch.trapz(y, x).item())
