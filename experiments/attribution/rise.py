"""RISE: Randomized Input Sampling for Explanation.

The implementation follows Petsiuk et al. (BMVC 2018), while generating
masks in batches so a complete mask bank never has to live in GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RISEConfig:
    num_masks: int = 4000
    mask_size: int = 7
    mask_probability: float = 0.5
    batch_size: int = 64
    seed: int = 42


class RISE:
    """Generate a 2-D RISE saliency map for one image at a time.

    Args:
        model: A model returning logits shaped ``(batch, classes)``.
        num_masks: Number of Monte-Carlo masks. Use 128--512 for debugging
            and at least 4000 for formal experiments.
        mask_size: Side length of the low-resolution Bernoulli mask.
        mask_probability: Probability that a low-resolution cell is visible.
        batch_size: Number of masked inputs per forward pass.
        seed: Seed used for deterministic mask generation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_masks: int = 4000,
        mask_size: int = 7,
        mask_probability: float = 0.5,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        if num_masks <= 0 or mask_size <= 0 or batch_size <= 0:
            raise ValueError("num_masks, mask_size and batch_size must be positive")
        if not 0.0 < mask_probability <= 1.0:
            raise ValueError("mask_probability must be in (0, 1]")
        self.model = model
        self.config = RISEConfig(
            num_masks=num_masks,
            mask_size=mask_size,
            mask_probability=mask_probability,
            batch_size=batch_size,
            seed=seed,
        )

    def _mask_batch(
        self,
        count: int,
        height: int,
        width: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample shifted, bilinearly upscaled RISE masks."""
        s = self.config.mask_size
        p = self.config.mask_probability
        cell_h = max(1, (height + s - 1) // s)
        cell_w = max(1, (width + s - 1) // s)
        up_h, up_w = (s + 1) * cell_h, (s + 1) * cell_w

        low = (
            torch.rand((count, 1, s, s), generator=generator, device="cpu") < p
        ).to(dtype=torch.float32, device=device)
        upscaled = F.interpolate(
            low, size=(up_h, up_w), mode="bilinear", align_corners=False
        )
        offsets_y = torch.randint(0, cell_h, (count,), generator=generator)
        offsets_x = torch.randint(0, cell_w, (count,), generator=generator)
        return torch.stack(
            [
                upscaled[i, :, y : y + height, x : x + width]
                for i, (y, x) in enumerate(zip(offsets_y.tolist(), offsets_x.tolist()))
            ],
            dim=0,
        )

    @torch.inference_mode()
    def attribute(
        self,
        image: torch.Tensor,
        target: int,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a normalized saliency map shaped ``(H, W)``.

        ``image`` may be ``(C,H,W)`` or ``(1,C,H,W)``. A baseline is useful
        when the image is normalized: passing the normalized black pixel makes
        hidden regions truly black before normalization.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("RISE expects exactly one image shaped (C,H,W) or (1,C,H,W)")
        if target < 0:
            raise ValueError("target must be a non-negative class index")

        device = image.device
        _, _, height, width = image.shape
        if baseline is None:
            baseline = torch.zeros_like(image)
        elif baseline.ndim == 3:
            baseline = baseline.unsqueeze(0)
        baseline = baseline.to(device=device, dtype=image.dtype)
        if baseline.shape not in (image.shape, (1, image.shape[1], 1, 1)):
            raise ValueError("baseline must broadcast as (1,C,H,W) or (1,C,1,1)")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        weighted_masks = torch.zeros((height, width), device=device)
        generated = 0

        while generated < self.config.num_masks:
            count = min(self.config.batch_size, self.config.num_masks - generated)
            masks = self._mask_batch(count, height, width, generator, device)
            masked = image * masks + baseline * (1.0 - masks)
            logits = self.model(masked)
            if logits.ndim != 2 or target >= logits.shape[1]:
                raise ValueError(
                    f"target {target} is incompatible with model output {tuple(logits.shape)}"
                )
            probabilities = logits.softmax(dim=1)[:, target]
            weighted_masks += (probabilities[:, None, None, None] * masks).sum(0).squeeze(0)
            generated += count

        saliency = weighted_masks / (self.config.num_masks * self.config.mask_probability)
        minimum, maximum = saliency.min(), saliency.max()
        if (maximum - minimum).item() > 1e-12:
            saliency = (saliency - minimum) / (maximum - minimum)
        else:
            saliency = torch.zeros_like(saliency)
        return saliency.detach().cpu()
