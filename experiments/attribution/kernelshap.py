"""Captum KernelSHAP adapter for the project's unified attribution interface.

The adapter explains one image at a time and groups pixels into a regular grid
of interpretable features.  Treating all 224x224x3 input values as independent
features would make KernelSHAP impractically expensive and would not match the
project's compute budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

import torch


@dataclass(frozen=True)
class KernelSHAPConfig:
    n_samples: int = 256
    perturbations_per_eval: int = 16
    feature_grid_size: int = 14
    seed: int = 42
    show_progress: bool = False


def _load_captum_kernel_shap() -> Type[Any]:
    try:
        from captum.attr import KernelShap
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "KernelSHAP requires Captum. Install the project environment after "
            "requirements.txt is merged, or run: pip install captum"
        ) from exc
    return KernelShap


class KernelSHAP:
    """Generate a normalized 2-D KernelSHAP attribution map for one image.

    Args:
        model: Model returning logits shaped ``(batch, classes)``.
        n_samples: Number of KernelSHAP perturbation samples.
        perturbations_per_eval: Perturbations evaluated in one model batch.
        feature_grid_size: Number of grid cells along each spatial dimension.
        seed: Random seed for Captum's feature coalition sampling.
        show_progress: Forward Captum's progress output.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: int = 256,
        perturbations_per_eval: int = 16,
        feature_grid_size: int = 14,
        seed: int = 42,
        show_progress: bool = False,
    ) -> None:
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2")
        if perturbations_per_eval <= 0:
            raise ValueError("perturbations_per_eval must be positive")
        if feature_grid_size <= 0:
            raise ValueError("feature_grid_size must be positive")
        self.model = model
        self.config = KernelSHAPConfig(
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            feature_grid_size=feature_grid_size,
            seed=seed,
            show_progress=show_progress,
        )

    def _feature_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Return an integer feature id map shaped ``(1,1,H,W)``."""

        grid = min(self.config.feature_grid_size, height, width)
        y_ids = torch.div(
            torch.arange(height, device=device) * grid,
            height,
            rounding_mode="floor",
        )
        x_ids = torch.div(
            torch.arange(width, device=device) * grid,
            width,
            rounding_mode="floor",
        )
        return (y_ids[:, None] * grid + x_ids[None, :]).view(1, 1, height, width)

    def attribute(
        self,
        image: torch.Tensor,
        target: int,
        baseline: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a signed-channel-mean saliency map shaped ``(H,W)``.

        The output is min-max normalized to ``[0,1]`` and moved to CPU so it
        matches the RISE pipeline contract.  The baseline should represent a
        black image after normalization.
        """

        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                "KernelSHAP expects exactly one image shaped (C,H,W) or (1,C,H,W)"
            )
        if target < 0:
            raise ValueError("target must be a non-negative class index")

        if baseline is None:
            baseline = torch.zeros_like(image)
        elif baseline.ndim == 3:
            baseline = baseline.unsqueeze(0)
        baseline = baseline.to(device=image.device, dtype=image.dtype)
        if baseline.shape not in (image.shape, (1, image.shape[1], 1, 1)):
            raise ValueError("baseline must broadcast as (1,C,H,W) or (1,C,1,1)")
        baseline = baseline.expand_as(image)

        _, _, height, width = image.shape
        feature_mask = self._feature_mask(height, width, image.device)
        kernel_shap = _load_captum_kernel_shap()(self.model)
        cuda_devices: list[int] = []
        if image.device.type == "cuda":
            cuda_devices = [
                image.device.index
                if image.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.config.seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(self.config.seed)
            attribution = kernel_shap.attribute(
                inputs=image,
                baselines=baseline,
                target=target,
                feature_mask=feature_mask,
                n_samples=self.config.n_samples,
                perturbations_per_eval=self.config.perturbations_per_eval,
                show_progress=self.config.show_progress,
            )

        if attribution.shape != image.shape:
            raise RuntimeError(
                "Captum returned an unexpected attribution shape: "
                f"{tuple(attribution.shape)} != {tuple(image.shape)}"
            )
        # Keep the signed ranking used by the other Captum methods in this
        # project. Taking abs() would rank strongly negative evidence as if it
        # were strongly positive evidence in the MoRF deletion metric.
        saliency = attribution.squeeze(0).mean(dim=0)
        minimum, maximum = saliency.min(), saliency.max()
        if (maximum - minimum).item() > 1e-12:
            saliency = (saliency - minimum) / (maximum - minimum)
        else:
            saliency = torch.zeros_like(saliency)
        return saliency.detach().cpu()
