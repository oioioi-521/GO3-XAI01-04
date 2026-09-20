"""Lightweight metrics used by the M2 pipeline demonstration."""

from __future__ import annotations

from typing import Iterable

import torch


def _validate_deletion_inputs(image, target, attribution, baseline, fractions):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if baseline.ndim == 3:
        baseline = baseline.unsqueeze(0)
    fractions = [float(value) for value in fractions]
    if not fractions or fractions[0] != 0.0 or fractions[-1] != 1.0 or any(b <= a for a, b in zip(fractions, fractions[1:])):
        raise ValueError("deletion fractions must be strictly increasing from 0.0 to 1.0")
    if attribution.ndim != 2 or not torch.isfinite(attribution).all():
        raise ValueError("attribution must be a finite (H, W) tensor")
    return image, int(target), attribution, baseline, fractions


def _probability(logits: torch.Tensor, target: int, output_activation: str) -> torch.Tensor:
    if logits.ndim != 2 or not 0 <= target < logits.shape[1]:
        raise ValueError("target is outside model output dimensions")
    if output_activation == "softmax":
        return logits.softmax(1)[0, target]
    if output_activation == "sigmoid":
        return logits.sigmoid()[0, target]
    raise ValueError("output_activation must be 'softmax' or 'sigmoid'")


@torch.inference_mode()
def faithfulness_morf_auc_raw(model, image, target, attribution, baseline, fractions, output_activation="softmax") -> float:
    """Bounded MoRF deletion AUC of the true target probability (lower is better)."""
    image, target, attribution, baseline, fractions = _validate_deletion_inputs(image, target, attribution, baseline, fractions)
    height, width = attribution.shape
    order = attribution.flatten().argsort(descending=True)
    values = []
    for fraction in fractions:
        mask = torch.zeros(height * width, dtype=torch.bool, device=image.device)
        mask[order[:round(fraction * height * width)].to(image.device)] = True
        perturbed = torch.where(mask.view(1, 1, height, width), baseline, image)
        values.append(_probability(model(perturbed), target, output_activation))
    score = float(torch.trapz(torch.stack(values), torch.tensor(fractions, device=image.device)).item())
    if not torch.isfinite(torch.tensor(score)):
        raise ValueError("deletion AUC is not finite")
    return min(1.0, max(0.0, score))


def raw_true_target_deletion_auc(*args, **kwargs) -> float:
    """Temporary compatibility wrapper; removed when the runner switches names."""
    return faithfulness_morf_auc_raw(*args, **kwargs)


def summarize_by_correct(scores: Iterable[float], correct: Iterable[bool]) -> dict[str, dict[str, float | int]]:
    """Return explicit correct/error group summaries without silently dropping errors."""
    groups = {True: [], False: []}
    for score, is_correct in zip(scores, correct):
        if not torch.isfinite(torch.tensor(float(score))):
            raise ValueError("scores must be finite")
        groups[bool(is_correct)].append(float(score))
    return {("correct" if key else "incorrect"): {"n": len(values), "mean": (sum(values) / len(values) if values else float("nan"))} for key, values in groups.items()}


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
