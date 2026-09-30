"""Benchmark RISE mask budgets and convergence on the frozen debug split.

Run a one-image timing smoke first, then the 40-image gate.  Each budget uses
the same per-image seed, so the masks from a smaller budget are a prefix of the
largest budget's stream and convergence comparisons are paired.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.attribution import RISE  # noqa: E402
from experiments.run_unit import (  # noqa: E402
    _black_baseline,
    _load_config,
    _resolve,
    _select_device,
    _synchronize,
)
from models import load_model  # noqa: E402
from preprocessing.dataset import MetadataDataset  # noqa: E402


COLUMNS = [
    "image_id",
    "dataset",
    "model",
    "num_masks",
    "time_ms",
    "reference_num_masks",
    "pearson_to_reference",
    "mae_to_reference",
]


def _map_similarity(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("RISE maps must have the same two-dimensional shape")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise ValueError("RISE maps must be finite")
    mae = float(np.abs(candidate - reference).mean())
    candidate_centered = candidate.ravel() - float(candidate.mean())
    reference_centered = reference.ravel() - float(reference.mean())
    denominator = float(
        np.linalg.norm(candidate_centered) * np.linalg.norm(reference_centered)
    )
    correlation = (
        float(np.dot(candidate_centered, reference_centered) / denominator)
        if denominator > 1e-12
        else float("nan")
    )
    return correlation, mae


def run(
    config_path: Path,
    budgets: list[int],
    max_images: int,
    output_path: Path,
) -> None:
    config = _load_config(config_path)
    if str(config["method"]).lower() != "rise":
        raise ValueError("sampling gate requires a RISE config")
    if max_images not in {1, 40}:
        raise ValueError("max_images must be 1 for smoke or 40 for the frozen gate")
    if len(set(budgets)) < 3 or any(budget <= 0 for budget in budgets):
        raise ValueError("provide at least three distinct positive mask budgets")
    budgets = sorted(set(budgets))

    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
    device = _select_device(str(config["runtime"].get("device", "auto")))
    if device.type != "cuda":
        raise RuntimeError("sampling-budget timing must run on the target CUDA GPU")

    dataset_name = str(config["dataset"]["name"]).lower()
    model_config = config["model"]
    model_name = str(model_config["name"]).lower()
    checkpoint = model_config.get("checkpoint")
    checkpoint_path = _resolve(str(checkpoint)) if checkpoint else None
    model = load_model(
        name=model_name,
        device=device,
        weights=str(model_config.get("weights", "default")),
        num_classes=int(model_config.get("num_classes", 1000)),
        checkpoint=str(checkpoint_path) if checkpoint_path else None,
        output_activation=str(model_config.get("output_activation", "softmax")),
    )
    dataset = MetadataDataset(dataset_name, split="debug", limit=max_images)
    baseline = _black_baseline(device, torch.float32)
    base_attribution = dict(config["attribution"])

    # One discarded warm-up per budget keeps compilation/cache effects out of
    # the timing while preserving the exact configured attribution path.
    warmup = dataset[0]
    warmup_image = warmup["image"].unsqueeze(0).to(device)
    for budget in budgets:
        params = {**base_attribution, "num_masks": budget}
        RISE(model=model, **params).attribute(
            warmup_image, target=int(warmup["target"]), baseline=baseline
        )
        _synchronize(device)

    rows: list[dict[str, object]] = []
    reference_budget = budgets[-1]
    for sample in dataset:
        image = sample["image"].unsqueeze(0).to(device)
        target = int(sample["target"])
        maps: dict[int, np.ndarray] = {}
        times: dict[int, float] = {}
        for budget in budgets:
            params = {**base_attribution, "num_masks": budget}
            attributor = RISE(model=model, **params)
            _synchronize(device)
            started = time.perf_counter()
            maps[budget] = attributor.attribute(
                image, target=target, baseline=baseline
            ).numpy()
            _synchronize(device)
            times[budget] = (time.perf_counter() - started) * 1000.0

        reference = maps[reference_budget]
        for budget in budgets:
            correlation, mae = _map_similarity(maps[budget], reference)
            rows.append({
                "image_id": sample["image_id"],
                "dataset": dataset_name,
                "model": model_name,
                "num_masks": budget,
                "time_ms": times[budget],
                "reference_num_masks": reference_budget,
                "pearson_to_reference": correlation,
                "mae_to_reference": mae,
            })
        print(f"completed {sample['image_id']} ({len(rows) // len(budgets)}/{len(dataset)})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--max-images", type=int, choices=[1, 40], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    run(config_path.resolve(), args.budgets, args.max_images, output_path.resolve())


if __name__ == "__main__":
    main()
