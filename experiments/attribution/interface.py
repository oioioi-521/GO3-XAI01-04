"""Shared public contract for attribution methods used by the experiment runner."""

from __future__ import annotations

from typing import Optional, Protocol

import torch


class AttributionMethod(Protocol):
    """One-image attribution interface consumed by ``experiments.run_unit``.

    Implementations accept either ``(C,H,W)`` or ``(1,C,H,W)`` input, explain
    the supplied ground-truth class, and return a finite CPU tensor shaped
    ``(H,W)`` with values in ``[0, 1]``. The optional baseline must be accepted
    even by methods, such as Grad-CAM, that do not use a reference input.
    """

    def attribute(
        self,
        image: torch.Tensor,
        target: int,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one normalized spatial attribution map."""
