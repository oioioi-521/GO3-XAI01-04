"""Experimental device-side LIME neighborhoods; not wired into formal runs.

Keep the official explainer's sampling RNG, distance, kernel and Ridge path.
Only replace its per-image RGB copying with batched superpixel lookups. Deploy
this module outside the frozen release for paired measurements against LIME.
"""
from __future__ import annotations

import hashlib
from numbers import Integral
import time

import numpy as np
import torch

from experiments.attribution.lime import LIME
from experiments.attribution.gradient_common import (
    normalize_signed_map, prepare_baseline, single_rgb_image,
)


class VectorizedLIME(LIME):
    def __init__(self, *args, capture_neighborhood=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_neighborhood = capture_neighborhood
        self.last_neighborhood = None

    @torch.inference_mode()
    def attribute(self, image, target, baseline=None):
        from lime.lime_image import LimeImageExplainer
        from sklearn.linear_model import Ridge
        from skimage.segmentation import quickshift

        started = time.perf_counter()
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
        before_segment = time.perf_counter()
        segments = quickshift(rgb_array, rng=self.segmentation_seed,
                              channel_axis=-1, **self.segmentation_params)
        _, segments = np.unique(segments, return_inverse=True)
        segments = segments.reshape(image.shape[-2:])
        n_segments = int(segments.max()) + 1
        segmentation_ms = (time.perf_counter() - before_segment) * 1000

        # Reproduce the reference's RGB round trip, clipping, and NHWC->NCHW
        # layout, including noise outside the nominal RGB range. Using image
        # directly here would change those inputs and is deliberately avoided.
        pixels = torch.as_tensor(rgb_array[None].copy(), device=image.device,
                                 dtype=image.dtype).permute(0, 3, 1, 2)
        source = (pixels - mean) / std
        reference = baseline.expand_as(image)
        segment_index = torch.as_tensor(segments, device=image.device)
        outer = self
        measured = {}

        class DeviceNeighborhood(LimeImageExplainer):
            def data_labels(self, image, fudged_image, segments, classifier_fn,
                            num_samples, batch_size=10):
                tick = time.perf_counter()
                # Same RNG API, draw size/order/dtype and first row as upstream.
                data = self.random_state.randint(0, 2, num_samples * n_segments).reshape(
                    (num_samples, n_segments))
                data[0, :] = 1
                keep = torch.as_tensor(data.astype(bool), device=source.device)
                predictions = []
                for start in range(0, num_samples, batch_size):
                    visible = keep[start:start + batch_size, segment_index].unsqueeze(1)
                    inputs = torch.where(visible, source, reference).contiguous(
                        memory_format=torch.channels_last)
                    logits = outer.model(inputs)
                    if logits.ndim != 2 or logits.shape[0] != len(inputs) or target >= logits.shape[1]:
                        raise ValueError("target incompatible with model output shape")
                    if not torch.isfinite(logits).all():
                        raise ValueError("model returned non-finite logits")
                    probs = logits.softmax(1) if outer.output_activation == "softmax" else logits.sigmoid()
                    predictions.append(probs.cpu().numpy().astype(np.float64))
                labels = np.concatenate(predictions, axis=0)
                measured.update(neighborhood_ms=(time.perf_counter() - tick) * 1000,
                                masks_sha256=hashlib.sha256(data.tobytes()).hexdigest())
                if outer.capture_neighborhood:
                    outer.last_neighborhood = {"masks": data.copy(), "predictions": labels.copy(),
                                               "segments": segments.copy()}
                return data, labels

        explainer = DeviceNeighborhood(kernel_width=self.kernel_width,
                                       feature_selection="none", random_state=self.seed)
        explanation = explainer.explain_instance(
            rgb_array, None, labels=(int(target),), top_labels=None,
            hide_color=-1.0, num_features=n_segments, num_samples=self.num_samples,
            batch_size=self.batch_size, segmentation_fn=lambda _: segments,
            distance_metric="cosine", random_seed=self.segmentation_seed,
            model_regressor=Ridge(alpha=self.ridge_alpha, fit_intercept=True, solver="cholesky"),
        )
        coefficients = np.zeros(n_segments, dtype=np.float32)
        for segment, weight in explanation.local_exp[int(target)]:
            coefficients[segment] = weight
        result = normalize_signed_map(torch.from_numpy(coefficients[segments])).to(torch.float32)
        self.last_diagnostics = dict(
            seed=self.seed, segmentation_seed=self.segmentation_seed,
            num_samples=self.num_samples, n_superpixels=n_segments,
            weighted_r2=float(explanation.score), coefficient_min=float(coefficients.min()),
            coefficient_max=float(coefficients.max()), segmentation_ms=segmentation_ms,
            **measured, total_internal_ms=(time.perf_counter() - started) * 1000,
            implementation="device_neighborhood_prototype_v1",
        )
        return result
