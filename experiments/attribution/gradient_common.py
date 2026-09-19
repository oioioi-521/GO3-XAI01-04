"""Validation and normalization helpers shared by gradient-based methods."""

from __future__ import annotations

from numbers import Integral
from typing import Optional

import torch

from experiments.attribution.occlusion import normalized_black_baseline


def single_rgb_image(image: torch.Tensor, method_name: str) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            f"{method_name} expects exactly one image shaped (C,H,W) or (1,C,H,W)"
        )
    if image.shape[1] != 3:
        raise ValueError(f"{method_name} expects a three-channel RGB image")
    if not torch.is_floating_point(image) or not torch.isfinite(image).all():
        raise ValueError("image must be a finite floating-point tensor")
    return image


def prepare_baseline(
    image: torch.Tensor, baseline: Optional[torch.Tensor]
) -> torch.Tensor:
    if baseline is None:
        baseline = normalized_black_baseline(image)
    elif baseline.ndim == 3:
        baseline = baseline.unsqueeze(0)
    if baseline.ndim != 4 or baseline.shape not in (
        image.shape,
        (1, image.shape[1], 1, 1),
    ):
        raise ValueError("baseline must broadcast as (1,C,H,W) or (1,C,1,1)")
    baseline = baseline.to(device=image.device, dtype=image.dtype)
    if not torch.isfinite(baseline).all():
        raise ValueError("baseline must contain only finite values")
    return baseline


def validate_target(
    model: torch.nn.Module, image: torch.Tensor, target: int
) -> None:
    if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
        raise ValueError("target must be a non-negative integer")
    with torch.inference_mode():
        logits = model(image)
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("model must return logits shaped (1, classes)")
    if target >= logits.shape[1]:
        raise ValueError(
            f"target {target} is incompatible with model output {tuple(logits.shape)}"
        )


def normalize_signed_map(attribution: torch.Tensor) -> torch.Tensor:
    """Map a signed 2-D attribution to ``[0,1]`` without changing its rank."""

    if attribution.ndim != 2 or not torch.isfinite(attribution).all():
        raise ValueError("attribution map must be a finite 2-D tensor")
    minimum, maximum = attribution.min(), attribution.max()
    if (maximum - minimum).item() <= 1e-12:
        return torch.zeros_like(attribution)
    return (attribution - minimum) / (maximum - minimum)


def normalize_nonnegative_map(attribution: torch.Tensor) -> torch.Tensor:
    """Scale a non-negative 2-D map while preserving Grad-CAM zero semantics."""

    if attribution.ndim != 2 or not torch.isfinite(attribution).all():
        raise ValueError("attribution map must be a finite 2-D tensor")
    if attribution.min().item() < -1e-7:
        raise ValueError("expected a non-negative attribution map")
    attribution = attribution.clamp_min(0.0)
    maximum = attribution.max()
    if maximum.item() <= 1e-12:
        return torch.zeros_like(attribution)
    return attribution / maximum
