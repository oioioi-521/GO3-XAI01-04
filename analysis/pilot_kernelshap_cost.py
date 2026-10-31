"""Benchmark KernelSHAP sample-count cost and convergence on frozen debug images.

This is a parameter-selection pilot, not a formal experiment runner. It loads
one model once, measures multiple ``n_samples`` values on the same images, and
compares every map with the largest requested sample count.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

from experiments.attribution.kernelshap import KernelSHAP  # noqa: E402
from experiments.run_unit import _black_baseline  # noqa: E402
from models import load_model  # noqa: E402
from preprocessing.dataset import MetadataDataset  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _similarity(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float, bool]:
    """Return Spearman, tie-aware top-decile Jaccard, and degeneracy status."""

    candidate_constant = float(candidate.max() - candidate.min()) <= 1e-12
    reference_constant = float(reference.max() - reference.min()) <= 1e-12
    degenerate = candidate_constant or reference_constant
    if degenerate:
        return float("nan"), float("nan"), True
    correlation = float(spearmanr(candidate.ravel(), reference.ravel()).statistic)
    candidate_threshold = float(np.quantile(candidate, 0.9))
    reference_threshold = float(np.quantile(reference, 0.9))
    candidate_top = candidate >= candidate_threshold
    reference_top = reference >= reference_threshold
    union = np.logical_or(candidate_top, reference_top).sum()
    overlap = float(np.logical_and(candidate_top, reference_top).sum() / union)
    return correlation, overlap, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--dataset", default="imagenet", choices=("imagenet", "voc"))
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--samples", default="64,128,256,512,1024,2048")
    parser.add_argument("--grid", type=int, default=14)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot/kernelshap_cost.csv"),
    )
    args = parser.parse_args()
    if args.images <= 0 or args.grid <= 0 or args.batch <= 0:
        raise ValueError("images, grid, and batch must be positive")

    sample_counts = [int(value) for value in args.samples.split(",")]
    if not sample_counts or len(sample_counts) != len(set(sample_counts)):
        raise ValueError("samples must contain unique integers")
    if any(value < 2 for value in sample_counts):
        raise ValueError("every sample count must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_count = max(sample_counts)
    checkpoint = None
    weights = "default"
    num_classes = 1000
    output_activation = "softmax"
    if args.dataset == "voc":
        weights = "none"
        num_classes = 20
        output_activation = "sigmoid"
        checkpoint = f"models/checkpoints/{args.model}_voc20.pt"

    model = load_model(
        name=args.model,
        device=device,
        weights=weights,
        num_classes=num_classes,
        checkpoint=checkpoint,
        output_activation=output_activation,
    )
    model.eval()
    dataset = MetadataDataset(args.dataset, split="debug", limit=args.images)
    baseline = _black_baseline(device, torch.float32)
    maps: dict[int, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    for n_samples in sample_counts:
        attributor = KernelSHAP(
            model=model,
            n_samples=n_samples,
            perturbations_per_eval=args.batch,
            feature_grid_size=args.grid,
            seed=42,
            show_progress=False,
        )
        first = dataset[0]
        attributor.attribute(
            first["image"].unsqueeze(0).to(device),
            target=int(first["target"]),
            baseline=baseline,
        )
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        maps[n_samples] = {}

        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(device)
            _sync(device)
            started = time.perf_counter()
            attribution = attributor.attribute(
                image,
                target=int(sample["target"]),
                baseline=baseline,
            )
            _sync(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            array = attribution.numpy().astype(np.float32, copy=False)
            if not np.isfinite(array).all():
                raise RuntimeError(f"non-finite map for n_samples={n_samples}")
            maps[n_samples][sample["image_id"]] = array
            rows.append({
                "dataset": args.dataset,
                "model": args.model,
                "image_id": sample["image_id"],
                "target": int(sample["target"]),
                "feature_grid_size": args.grid,
                "n_samples": n_samples,
                "time_ms": elapsed_ms,
                "map_min": float(array.min()),
                "map_max": float(array.max()),
            })

        peak_mib = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else float("nan")
        )
        for row in rows:
            if row["n_samples"] == n_samples:
                row["peak_memory_mib"] = peak_mib

    reference = maps[reference_count]
    for row in rows:
        image_id = str(row["image_id"])
        correlation, overlap, degenerate = _similarity(
            maps[int(row["n_samples"])][image_id], reference[image_id]
        )
        row["reference_n_samples"] = reference_count
        row["spearman_to_reference"] = correlation
        row["top10_jaccard_to_reference"] = overlap
        row["degenerate"] = degenerate

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = frame.groupby("n_samples", sort=True).agg(
        time_ms_mean=("time_ms", "mean"),
        time_ms_std=("time_ms", "std"),
        spearman_mean=("spearman_to_reference", "mean"),
        spearman_min=("spearman_to_reference", "min"),
        top10_jaccard_mean=("top10_jaccard_to_reference", "mean"),
        degenerate_count=("degenerate", "sum"),
        peak_memory_mib=("peak_memory_mib", "max"),
    )
    print(summary.to_string())
    print(f"wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
