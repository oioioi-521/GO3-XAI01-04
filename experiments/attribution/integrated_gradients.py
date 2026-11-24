"""Integrated Gradients adapter for the project's attribution contract."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import torch

from experiments.attribution.gradient_common import (
    normalize_signed_map,
    prepare_baseline,
    single_rgb_image,
    validate_target,
)


_APPROXIMATION_METHODS = {
    "gausslegendre",
    "riemann_left",
    "riemann_middle",
    "riemann_right",
    "riemann_trapezoid",
}


@dataclass(frozen=True)
class IntegratedGradientsConfig:
    n_steps: int = 50
    internal_batch_size: int | None = 10
    approximation_method: str = "gausslegendre"


class IntegratedGradients:
    """Produce a normalized 2-D signed Integrated Gradients heatmap.

    Captum computes per-channel attributions from the normalized black image
    to the input. RGB values are reduced by a signed arithmetic mean, then
    min-max normalized. The transform preserves the descending spatial rank
    consumed by the current MoRF metric; returned zero denotes the per-image
    minimum signed attribution rather than necessarily zero raw contribution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 50,
        internal_batch_size: int | None = 10,
        approximation_method: str = "gausslegendre",
    ) -> None:
        if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 2:
            raise ValueError("n_steps must be an integer of at least 2")
        if internal_batch_size is not None and (
            isinstance(internal_batch_size, bool)
            or not isinstance(internal_batch_size, Integral)
            or internal_batch_size <= 0
        ):
            raise ValueError("internal_batch_size must be null or a positive integer")
        if approximation_method not in _APPROXIMATION_METHODS:
            raise ValueError(
                "approximation_method must be one of "
                f"{sorted(_APPROXIMATION_METHODS)}"
            )
        self.model = model
        self.config = IntegratedGradientsConfig(
            n_steps=int(n_steps),
            internal_batch_size=(
                int(internal_batch_size) if internal_batch_size is not None else None
            ),
            approximation_method=approximation_method,
        )

    @staticmethod
    def _captum_class():
        try:
            from captum.attr import IntegratedGradients as CaptumIntegratedGradients
        except ImportError as error:  # pragma: no cover - environment setup failure
            raise RuntimeError(
                "Captum is required for Integrated Gradients. Install requirements.txt first."
            ) from error
        return CaptumIntegratedGradients

    def attribute(
        self,
        image: torch.Tensor,
        target: int,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image = single_rgb_image(image, "Integrated Gradients")
        baseline = prepare_baseline(image, baseline).expand_as(image)
        validate_target(self.model, image, target)

        raw = self._captum_class()(self.model).attribute(
            image,
            baselines=baseline,
            target=int(target),
            n_steps=self.config.n_steps,
            method=self.config.approximation_method,
            internal_batch_size=self.config.internal_batch_size,
        )
        if not isinstance(raw, torch.Tensor) or raw.shape != image.shape:
            raise ValueError("Captum Integrated Gradients returned an unexpected shape")
        if not torch.isfinite(raw).all():
            raise ValueError("Captum Integrated Gradients returned non-finite attributions")

        signed_spatial = raw.mean(dim=1).squeeze(0)
        heatmap = normalize_signed_map(signed_spatial)
        if heatmap.shape != image.shape[-2:]:
            raise ValueError("failed to produce a spatial Integrated Gradients heatmap")
        return heatmap.clamp(0.0, 1.0).detach().cpu()
