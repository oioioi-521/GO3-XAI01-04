"""Captum Occlusion adapter for the project's normalized image contract.

The adapter intentionally follows :class:`RISE`'s public call shape:
``attribute(image, target, baseline) -> Tensor[H, W]``.  Captum returns one
signed attribution per input channel.  We take the arithmetic mean across RGB
channels (never an absolute value), then min-max normalize the signed spatial
map.  Consequently, higher values mean locations whose occlusion caused a
larger increase in the selected target logit; negative contributions remain
below positive contributions before normalization instead of being discarded.
The returned value ``0`` is the per-image *minimum* signed score, not generally
an original zero-contribution score.  This is safe for the current MoRF metric,
which uses only descending spatial rank; min-max normalization preserves that
rank for every non-constant map.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional

import torch


# These are the values used by preprocessing.dataset.  A caller using another
# preprocessing convention must supply its own baseline explicitly.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class OcclusionConfig:
    """Sliding-window parameters passed to Captum Occlusion.

    ``window_height`` and ``window_width`` are measured in input pixels after
    preprocessing (normally 224 x 224).  Strides are validated so every pixel
    is covered by at least one window.
    """

    window_height: int = 15
    window_width: int = 15
    stride_height: int = 8
    stride_width: int = 8
    perturbations_per_eval: int = 1


def normalized_black_baseline(
    image: torch.Tensor,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    """Return a broadcastable tensor representing an RGB black pixel.

    Images in this project are normalized as ``(pixel - mean) / std``.  Thus
    numeric zero is *not* black.  The returned ``(1, C, 1, 1)`` tensor is the
    normalized representation of an all-zero RGB pixel.
    """

    if image.ndim == 3:
        channels = image.shape[0]
    elif image.ndim == 4:
        channels = image.shape[1]
    else:
        raise ValueError("image must have shape (C,H,W) or (1,C,H,W)")
    if channels != len(mean) or len(mean) != len(std):
        raise ValueError("black baseline constants must match the image channel count")
    values = [(-channel_mean / channel_std) for channel_mean, channel_std in zip(mean, std)]
    return torch.tensor(values, device=image.device, dtype=image.dtype).view(1, channels, 1, 1)


class Occlusion:
    """Produce a normalized 2-D signed Occlusion heatmap for one image.

    Captum measures the selected output's change after replacing a window with
    a baseline.  This adapter preserves that sign through RGB reduction by
    using a channel mean.  It then performs min-max normalization solely to
    satisfy the shared ``[0, 1]`` heatmap contract required by the current
    MoRF metric and PNG writer; the returned numbers are therefore relative
    spatial ranks, not signed effect magnitudes.  In particular, returned zero
    denotes the image's lowest signed attribution; it denotes a raw score of
    zero only when the raw map minimum is itself zero.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        window_height: int = 15,
        window_width: int = 15,
        stride_height: int = 8,
        stride_width: int = 8,
        perturbations_per_eval: int = 1,
    ) -> None:
        self.config = OcclusionConfig(
            window_height=window_height,
            window_width=window_width,
            stride_height=stride_height,
            stride_width=stride_width,
            perturbations_per_eval=perturbations_per_eval,
        )
        self._validate_config_values()
        self.model = model

    def _validate_config_values(self) -> None:
        values = (
            self.config.window_height,
            self.config.window_width,
            self.config.stride_height,
            self.config.stride_width,
            self.config.perturbations_per_eval,
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in values):
            raise ValueError("window sizes, strides, and perturbations_per_eval must be positive integers")

    @staticmethod
    def _single_image(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Occlusion expects exactly one image shaped (C,H,W) or (1,C,H,W)")
        if image.shape[1] != 3:
            raise ValueError("Occlusion expects a three-channel RGB image")
        if not torch.isfinite(image).all():
            raise ValueError("image must contain only finite values")
        return image

    def _validate_geometry(self, image: torch.Tensor) -> None:
        _, _, height, width = image.shape
        if self.config.window_height > height or self.config.window_width > width:
            raise ValueError("occlusion window must not exceed image height or width")
        if (
            (self.config.window_height < height and self.config.stride_height > self.config.window_height)
            or (self.config.window_width < width and self.config.stride_width > self.config.window_width)
        ):
            raise ValueError("stride must not exceed the window size when the window does not cover that dimension")

    @staticmethod
    def _prepare_baseline(image: torch.Tensor, baseline: Optional[torch.Tensor]) -> torch.Tensor:
        if baseline is None:
            baseline = normalized_black_baseline(image)
        elif baseline.ndim == 3:
            baseline = baseline.unsqueeze(0)
        if baseline.ndim != 4 or baseline.shape not in (image.shape, (1, image.shape[1], 1, 1)):
            raise ValueError("baseline must broadcast as (1,C,H,W) or (1,C,1,1)")
        baseline = baseline.to(device=image.device, dtype=image.dtype)
        if not torch.isfinite(baseline).all():
            raise ValueError("baseline must contain only finite values")
        return baseline

    @staticmethod
    def _normalize_signed_map(attribution: torch.Tensor) -> torch.Tensor:
        """Preserve signed ordering while mapping a spatial map into ``[0, 1]``."""

        if attribution.ndim != 2 or not torch.isfinite(attribution).all():
            raise ValueError("attribution map must be a finite 2-D tensor")
        minimum, maximum = attribution.min(), attribution.max()
        if (maximum - minimum).item() <= 1e-12:
            return torch.zeros_like(attribution)
        return (attribution - minimum) / (maximum - minimum)

    @staticmethod
    def _captum_occlusion_class():
        try:
            from captum.attr import Occlusion as CaptumOcclusion
        except ImportError as error:  # pragma: no cover - exercised by environment setup
            raise RuntimeError(
                "Captum is required for Occlusion. Install the project test environment before running it."
            ) from error
        return CaptumOcclusion

    def attribute(
        self,
        image: torch.Tensor,
        target: int,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a finite CPU heatmap shaped ``(H, W)`` in ``[0, 1]``.

        ``target`` always selects a true-class logit.  It is checked against an
        actual model forward pass before Captum is invoked, preventing a VOC
        label from being accidentally used with a 1,000-class ImageNet model.
        """

        if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
            raise ValueError("target must be a non-negative integer")
        image = self._single_image(image)
        self._validate_geometry(image)
        baseline = self._prepare_baseline(image, baseline)

        with torch.inference_mode():
            logits = self.model(image)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("model must return logits shaped (1, classes)")
        if target >= logits.shape[1]:
            raise ValueError(
                f"target {target} is incompatible with model output {tuple(logits.shape)}"
            )

        captum_occlusion = self._captum_occlusion_class()(self.model)
        raw = captum_occlusion.attribute(
            image,
            sliding_window_shapes=(image.shape[1], self.config.window_height, self.config.window_width),
            strides=(image.shape[1], self.config.stride_height, self.config.stride_width),
            baselines=baseline,
            target=int(target),
            perturbations_per_eval=self.config.perturbations_per_eval,
        )
        if not isinstance(raw, torch.Tensor) or raw.shape != image.shape:
            raise ValueError("Captum Occlusion returned an unexpected attribution shape")
        if not torch.isfinite(raw).all():
            raise ValueError("Captum Occlusion returned non-finite attributions")

        # A signed arithmetic mean preserves ordering between counter-evidence
        # (negative) and supporting evidence (positive), unlike abs/sum(abs()).
        signed_spatial = raw.mean(dim=1).squeeze(0)
        heatmap = self._normalize_signed_map(signed_spatial)
        if heatmap.shape != image.shape[-2:] or not torch.isfinite(heatmap).all():
            raise ValueError("failed to produce a finite spatial Occlusion heatmap")
        return heatmap.clamp(0.0, 1.0).detach().cpu()
