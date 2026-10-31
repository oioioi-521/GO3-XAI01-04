"""Grad-CAM adapter with explicit, auditable target-layer selection."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from experiments.attribution.gradient_common import (
    normalize_nonnegative_map,
    prepare_baseline,
    single_rgb_image,
    validate_target,
)


def resolve_target_layer(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Resolve a dotted module path such as ``features.28`` or ``layer4.2``."""

    if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
        raise ValueError("target_layer must be a non-empty dotted module path")
    current = model
    traversed = []
    for part in path.split("."):
        traversed.append(part)
        if part in current._modules:
            current = current._modules[part]
            continue
        candidate = getattr(current, part, None)
        if isinstance(candidate, torch.nn.Module):
            current = candidate
            continue
        raise ValueError(
            f"target_layer {path!r} does not resolve at {'.'.join(traversed)!r}"
        )
    return current


class GradCAM:
    """Produce original-paper Grad-CAM (ReLU) at input image resolution."""

    def __init__(self, model: torch.nn.Module, target_layer: str) -> None:
        self.model = model
        self.target_layer_path = target_layer
        self.target_layer = resolve_target_layer(model, target_layer)

    @staticmethod
    def _captum_classes():
        try:
            from captum.attr import LayerAttribution, LayerGradCam
        except ImportError as error:  # pragma: no cover - environment setup failure
            raise RuntimeError(
                "Captum is required for Grad-CAM. Install requirements.txt first."
            ) from error
        return LayerGradCam, LayerAttribution

    def attribute(
        self,
        image: torch.Tensor,
        target: int,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image = single_rgb_image(image, "Grad-CAM")
        # Grad-CAM does not use a reference input, but validating the shared
        # argument catches a malformed runner configuration consistently.
        prepare_baseline(image, baseline)
        validate_target(self.model, image, target)

        layer_grad_cam, layer_attribution = self._captum_classes()
        raw = layer_grad_cam(self.model, self.target_layer).attribute(
            image,
            target=int(target),
            relu_attributions=True,
            attr_dim_summation=True,
        )
        if not isinstance(raw, torch.Tensor) or raw.ndim != 4:
            raise ValueError("Captum LayerGradCam returned an unexpected attribution shape")
        if raw.shape[:2] != (1, 1) or not torch.isfinite(raw).all():
            raise ValueError("Grad-CAM target layer must return one finite spatial tensor")

        # Use Captum's documented interpolation path. The F.interpolate
        # fallback is retained for compatibility with older supported Captum
        # releases if their helper rejects a layer-specific output shape.
        try:
            upsampled = layer_attribution.interpolate(
                raw, image.shape[-2:], interpolate_mode="bilinear"
            )
        except (TypeError, ValueError):
            upsampled = F.interpolate(
                raw, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        if upsampled.shape != (1, 1, *image.shape[-2:]):
            raise ValueError("failed to upsample Grad-CAM to input resolution")
        heatmap = normalize_nonnegative_map(upsampled.squeeze(0).squeeze(0))
        return heatmap.clamp(0.0, 1.0).detach().cpu()
