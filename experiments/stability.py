"""Frozen input-noise stability protocol shared by pilots and formal runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from scipy.stats import spearmanr

from preprocessing.dataset import MEAN, STD


PROTOCOL_VERSION = "rgb-gaussian-spearman-v1"


@dataclass(frozen=True)
class StabilitySimilarity:
    """One repeat's raw diagnostic values and aggregate-ready score."""

    spearman: float
    score: float
    top_jaccard: float
    status: str


def stable_seed(dataset: str, image_id: str, repeat: int) -> int:
    """Derive a paired seed that is independent of method, model, and order."""

    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise ValueError("repeat must be a non-negative integer")
    payload = f"{str(dataset).lower()}\0{str(image_id)}\0{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def perturb_rgb(
    normalized: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Add deterministic Gaussian noise in RGB ``[0,1]`` and renormalize."""

    if normalized.ndim != 4 or normalized.shape[:2] != (1, 3):
        raise ValueError("normalized input must have shape (1,3,H,W)")
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized input must be finite")
    if not 0.0 < float(sigma) < 1.0:
        raise ValueError("sigma must be in (0,1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    cpu = normalized.detach().cpu()
    mean = torch.tensor(MEAN, dtype=cpu.dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, dtype=cpu.dtype).view(1, 3, 1, 1)
    pixels = (cpu * std + mean).clamp(0.0, 1.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(pixels.shape, generator=generator, dtype=pixels.dtype)
    perturbed_pixels = (pixels + float(sigma) * noise).clamp(0.0, 1.0)
    mae = float((perturbed_pixels - pixels).abs().mean())
    return (perturbed_pixels - mean) / std, mae


def stability_similarity(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    top_fraction: float = 0.1,
    degenerate_score: float = 0.0,
) -> StabilitySimilarity:
    """Compare maps and score every undefined repeat as an explicit zero.

    ``spearman`` and ``top_jaccard`` remain NaN for a degenerate repeat so the
    trace cannot be mistaken for a defined correlation. ``score`` is always
    finite and is the value used by the formal per-image aggregate.
    """

    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("attribution maps must have the same two-dimensional shape")
    if not 0.0 < float(top_fraction) < 1.0:
        raise ValueError("top_fraction must be in (0,1)")
    if not np.isfinite(float(degenerate_score)):
        raise ValueError("degenerate_score must be finite")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        return StabilitySimilarity(
            float("nan"), float(degenerate_score), float("nan"), "nonfinite"
        )

    candidate_constant = float(candidate.max() - candidate.min()) <= 1e-12
    reference_constant = float(reference.max() - reference.min()) <= 1e-12
    if reference_constant and candidate_constant:
        status = "both_constant"
    elif reference_constant:
        status = "reference_constant"
    elif candidate_constant:
        status = "candidate_constant"
    else:
        status = "valid"
    if status != "valid":
        return StabilitySimilarity(
            float("nan"), float(degenerate_score), float("nan"), status
        )

    correlation = float(spearmanr(reference.ravel(), candidate.ravel()).statistic)
    if not np.isfinite(correlation):
        return StabilitySimilarity(
            float("nan"), float(degenerate_score), float("nan"), "undefined_spearman"
        )
    quantile = 1.0 - float(top_fraction)
    reference_top = reference >= float(np.quantile(reference, quantile))
    candidate_top = candidate >= float(np.quantile(candidate, quantile))
    union = int(np.logical_or(reference_top, candidate_top).sum())
    overlap = float(np.logical_and(reference_top, candidate_top).sum() / union)
    return StabilitySimilarity(correlation, correlation, overlap, "valid")
