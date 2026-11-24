"""Official image LIME adapter: quickshift, Bernoulli masks and weighted Ridge."""

from __future__ import annotations

from numbers import Integral
from typing import Optional

import numpy as np
import torch

from .gradient_common import prepare_baseline, single_rgb_image, normalize_signed_map


class LIME:
    """Explain the supplied target and project all signed coefficients to pixels.

    A fresh local RNG per call makes warm-up, resume and call order irrelevant.
    Segmentation runs in denormalized RGB; model inference uses normalized RGB.
    The returned min-max map preserves signed coefficient ordering, not magnitude.
    """

    def __init__(
        self, model: torch.nn.Module, num_samples: int = 1000,
        batch_size: int = 32, seed: int = 42, segmentation_seed: int = 42,
        segmentation: str = "quickshift", kernel_size: float = 4.0,
        max_dist: float = 200.0, ratio: float = 0.2, sigma: float = 0.0,
        convert2lab: bool = True, kernel: str = "exponential",
        kernel_width: float = 0.25, distance_metric: str = "cosine",
        ridge_alpha: float = 1.0, ridge_solver: str = "cholesky",
        feature_selection: str = "none", output_activation: str = "softmax",
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ) -> None:
        for name, value, minimum in (("num_samples", num_samples, 2),
                                     ("batch_size", batch_size, 1),
                                     ("seed", seed, 0),
                                     ("segmentation_seed", segmentation_seed, 0)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if seed >= 2**32 or segmentation_seed >= 2**32:
            raise ValueError("seeds must fit uint32")
        for name, value in (("kernel_size", kernel_size), ("max_dist", max_dist),
                            ("kernel_width", kernel_width), ("ridge_alpha", ridge_alpha)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(ratio) or not 0 <= ratio <= 1 or not np.isfinite(sigma) or sigma < 0:
            raise ValueError("ratio must be in [0,1]; sigma must be finite and nonnegative")
        if (segmentation, kernel, distance_metric, feature_selection, ridge_solver) != (
            "quickshift", "exponential", "cosine", "none", "cholesky"
        ):
            raise ValueError("supported settings: quickshift/exponential/cosine/none/cholesky")
        if output_activation not in {"softmax", "sigmoid"}:
            raise ValueError("output_activation must be softmax or sigmoid")
        if (len(mean) != 3 or len(std) != 3 or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(np.asarray(std) <= 0)):
            raise ValueError("mean/std must contain three finite values; std must be positive")
        self.model = model
        self.num_samples, self.batch_size = int(num_samples), int(batch_size)
        self.seed, self.segmentation_seed = int(seed), int(segmentation_seed)
        self.segmentation_params = dict(kernel_size=kernel_size, max_dist=max_dist,
                                        ratio=ratio, sigma=sigma, convert2lab=convert2lab)
        self.kernel_width, self.ridge_alpha = kernel_width, ridge_alpha
        self.output_activation = output_activation
        self.mean, self.std = tuple(mean), tuple(std)
        self.last_diagnostics: dict = {}

    @torch.inference_mode()
    def attribute(self, image: torch.Tensor, target: int,
                  baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Optional dependencies stay lazy so other methods still work without LIME.
        from lime.lime_image import LimeImageExplainer
        from sklearn.linear_model import Ridge
        from skimage.segmentation import quickshift

        image = single_rgb_image(image, "LIME")
        if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
            raise ValueError("target must be a non-negative integer")
        if self.model.training:
            raise ValueError("LIME requires model.eval() for reproducible predictions")
        baseline = prepare_baseline(image, baseline)
        mean = image.new_tensor(self.mean).view(1, 3, 1, 1)
        std = image.new_tensor(self.std).view(1, 3, 1, 1)
        rgb = (image * std + mean).clamp(0, 1)
        rgb_array = rgb[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        segments = quickshift(rgb_array, rng=self.segmentation_seed,
                              channel_axis=-1, **self.segmentation_params)
        _, segments = np.unique(segments, return_inverse=True)
        segments = segments.reshape(image.shape[-2:])
        n_segments = int(segments.max()) + 1
        reference = baseline.expand_as(image)

        def classifier(batch):
            # LIME hides regions with an out-of-range sentinel. This lets us
            # honor even a spatially varying normalized baseline without clipping it.
            array = np.asarray(batch)
            visible = torch.as_tensor(array[..., :1] >= 0, device=image.device).permute(0, 3, 1, 2)
            pixels = torch.as_tensor(array.copy(), device=image.device, dtype=image.dtype).permute(0, 3, 1, 2)
            inputs = torch.where(visible, (pixels - mean) / std, reference)
            logits = self.model(inputs)
            if logits.ndim != 2 or logits.shape[0] != len(batch) or target >= logits.shape[1]:
                raise ValueError("target incompatible with model output shape")
            if not torch.isfinite(logits).all():
                raise ValueError("model returned non-finite logits")
            probabilities = logits.softmax(1) if self.output_activation == "softmax" else logits.sigmoid()
            return probabilities.cpu().numpy().astype(np.float64)

        explainer = LimeImageExplainer(kernel_width=self.kernel_width,
                                       feature_selection="none", random_state=self.seed)
        explanation = explainer.explain_instance(
            rgb_array, classifier, labels=(int(target),), top_labels=None,
            hide_color=-1.0, num_features=n_segments, num_samples=self.num_samples,
            batch_size=self.batch_size, segmentation_fn=lambda _: segments,
            distance_metric="cosine", random_seed=self.segmentation_seed,
            model_regressor=Ridge(alpha=self.ridge_alpha, fit_intercept=True,
                                  solver="cholesky"),
        )
        coefficients = np.zeros(n_segments, dtype=np.float32)
        for segment, weight in explanation.local_exp[int(target)]:
            coefficients[segment] = weight
        raw = torch.from_numpy(coefficients[segments])
        result = normalize_signed_map(raw).to(dtype=torch.float32)
        self.last_diagnostics = {
            "seed": self.seed, "segmentation_seed": self.segmentation_seed,
            "num_samples": self.num_samples, "n_superpixels": n_segments,
            "weighted_r2": float(explanation.score),
            "coefficient_min": float(coefficients.min()),
            "coefficient_max": float(coefficients.max()),
        }
        return result
