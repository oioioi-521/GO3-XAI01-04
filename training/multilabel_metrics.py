"""Finite, dependency-free validation metrics for VOC20 multi-label outputs."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def _average_precision(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    order = torch.argsort(probabilities, descending=True)
    sorted_targets = targets[order]
    positives = int(sorted_targets.sum().item())
    if positives == 0:
        return 0.0
    ranks = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
    precision_at_hits = sorted_targets * sorted_targets.cumsum(0) / ranks
    return float(precision_at_hits.sum().item() / positives)


def _f1(predicted: torch.Tensor, targets: torch.Tensor) -> float:
    true_positive = int((predicted & targets).sum().item())
    false_positive = int((predicted & ~targets).sum().item())
    false_negative = int((~predicted & targets).sum().item())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def multilabel_report(
    logits: torch.Tensor,
    targets: torch.Tensor,
    classes: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return VOC mAP plus macro/micro F1 for logits and 0/1 multi-hot targets."""
    if logits.ndim != 2 or targets.shape != logits.shape or logits.shape[1] != len(classes):
        raise ValueError("logits and targets must have matching [N, num_classes] shapes")
    if not 0.0 < threshold < 1.0:
        raise ValueError("sigmoid threshold must be strictly between 0 and 1")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("logits and targets must be finite")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("multi-label targets must contain only 0 or 1")

    probabilities = logits.sigmoid()
    binary_targets = targets.bool()
    predictions = probabilities >= threshold
    ap = [_average_precision(probabilities[:, index], targets[:, index]) for index in range(len(classes))]
    class_f1 = [_f1(predictions[:, index], binary_targets[:, index]) for index in range(len(classes))]
    report = {
        "mAP": sum(ap) / len(ap),
        "macro_f1": sum(class_f1) / len(class_f1),
        "micro_f1": _f1(predictions, binary_targets),
        "threshold": threshold,
        "per_class_ap": dict(zip(classes, ap)),
    }
    if not all(torch.isfinite(torch.tensor(value)) for value in (report["mAP"], report["macro_f1"], report["micro_f1"])):
        raise ValueError("multi-label metrics must be finite")
    return report
