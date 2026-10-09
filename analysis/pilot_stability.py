"""Evaluate candidate input-noise stability protocols on frozen debug images.

This pilot deliberately does not write the project's formal ``per_image.csv``
schema. Its purpose is to select perturbation strength and degeneracy rules
before the group freezes a stability metric in issue #8.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))

from experiments.run_unit import (  # noqa: E402
    _black_baseline,
    _build_attributor,
    _load_config,
    _resolve,
)
from experiments.stability import (  # noqa: E402
    perturb_rgb,
    stability_similarity,
    stable_seed,
)
from models import load_model  # noqa: E402
from preprocessing.dataset import MetadataDataset  # noqa: E402


METHODS = ("ig", "gradcam")
MODELS = ("vgg16", "resnet50", "densenet121")
DATASETS = ("imagenet", "voc")


def _scores(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_activation: str,
) -> tuple[torch.Tensor, int]:
    with torch.inference_mode():
        logits = model(image)
    if logits.ndim != 2 or logits.shape[0] != 1 or not torch.isfinite(logits).all():
        raise ValueError("model must return one finite logits row")
    if output_activation == "softmax":
        scores = logits.softmax(dim=1)
    elif output_activation == "sigmoid":
        scores = logits.sigmoid()
    else:
        raise ValueError(f"unsupported output activation: {output_activation}")
    return scores, int(scores.argmax(dim=1).item())


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_csv(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        raise ValueError(f"invalid {name}: {unknown or value}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--images", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sigmas", default="0.01,0.02,0.05")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot/stability_candidate.csv"),
    )
    args = parser.parse_args()
    if args.images <= 0 or args.repeats <= 0:
        raise ValueError("images and repeats must be positive")

    methods = _parse_csv(args.methods, METHODS, "methods")
    models = _parse_csv(args.models, MODELS, "models")
    datasets = _parse_csv(args.datasets, DATASETS, "datasets")
    sigmas = [float(value) for value in args.sigmas.split(",")]
    if not sigmas or len(sigmas) != len(set(sigmas)) or any(not 0 < value < 1 for value in sigmas):
        raise ValueError("sigmas must contain unique values in (0,1)")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    rows: list[dict[str, object]] = []

    for dataset_name in datasets:
        dataset = MetadataDataset(dataset_name, split="debug", limit=args.images)
        for model_name in models:
            reference_config = _load_config(
                PROJECT_ROOT / "configs" / f"{methods[0]}_{model_name}_{dataset_name}.yaml"
            )
            model_config = reference_config["model"]
            checkpoint = model_config.get("checkpoint")
            checkpoint_path = _resolve(str(checkpoint)) if checkpoint else None
            output_activation = str(model_config["output_activation"])
            model = load_model(
                name=model_name,
                device=device,
                weights=str(model_config.get("weights", "default")),
                num_classes=int(model_config.get("num_classes", 1000)),
                checkpoint=str(checkpoint_path) if checkpoint_path else None,
                output_activation=output_activation,
            )
            model.eval()
            baseline = _black_baseline(device, torch.float32)

            for method_name in methods:
                config = _load_config(
                    PROJECT_ROOT / "configs" / f"{method_name}_{model_name}_{dataset_name}.yaml"
                )
                attributor = _build_attributor(
                    method_name, model=model, attribution=config["attribution"]
                )
                first = dataset[0]
                attributor.attribute(
                    first["image"].unsqueeze(0).to(device),
                    target=int(first["target"]),
                    baseline=baseline,
                )
                _sync(device)

                for sample in dataset:
                    image_id = sample["image_id"]
                    target = int(sample["target"])
                    image = sample["image"].unsqueeze(0).to(device)
                    original_scores, original_pred = _scores(
                        model, image, output_activation
                    )
                    original_map = attributor.attribute(
                        image, target=target, baseline=baseline
                    ).numpy().astype(np.float32, copy=False)

                    for sigma in sigmas:
                        for repeat in range(args.repeats):
                            seed = stable_seed(dataset_name, image_id, repeat)
                            perturbed_cpu, perturbation_mae = perturb_rgb(
                                sample["image"].unsqueeze(0), sigma=sigma, seed=seed
                            )
                            perturbed = perturbed_cpu.to(device)
                            perturbed_scores, perturbed_pred = _scores(
                                model, perturbed, output_activation
                            )
                            _sync(device)
                            started = time.perf_counter()
                            candidate_map = attributor.attribute(
                                perturbed, target=target, baseline=baseline
                            ).numpy().astype(np.float32, copy=False)
                            _sync(device)
                            elapsed_ms = (time.perf_counter() - started) * 1000.0
                            similarity = stability_similarity(candidate_map, original_map)
                            original_target = float(original_scores[0, target].item())
                            perturbed_target = float(perturbed_scores[0, target].item())
                            rows.append({
                                "dataset": dataset_name,
                                "model": model_name,
                                "method": method_name,
                                "image_id": image_id,
                                "target": target,
                                "sigma": sigma,
                                "repeat": repeat,
                                "seed": seed,
                                "status": similarity.status,
                                "spearman": similarity.spearman,
                                "top10_jaccard": similarity.top_jaccard,
                                "original_pred": original_pred,
                                "perturbed_pred": perturbed_pred,
                                "prediction_preserved": original_pred == perturbed_pred,
                                "original_target_score": original_target,
                                "perturbed_target_score": perturbed_target,
                                "target_score_abs_delta": abs(perturbed_target - original_target),
                                "perturbation_mae_pixel": perturbation_mae,
                                "attribution_time_ms": elapsed_ms,
                            })

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    frame["valid"] = frame["status"].eq("valid")
    summary = frame.groupby(
        ["dataset", "model", "method", "sigma"], sort=True
    ).agg(
        repeats_total=("status", "size"),
        valid_repeats=("spearman", "count"),
        degenerate_repeats=("valid", lambda values: int((~values).sum())),
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
        spearman_min=("spearman", "min"),
        top10_jaccard_mean=("top10_jaccard", "mean"),
        prediction_preserved_rate=("prediction_preserved", "mean"),
        target_score_abs_delta_mean=("target_score_abs_delta", "mean"),
        target_score_abs_delta_max=("target_score_abs_delta", "max"),
        perturbation_mae_pixel_mean=("perturbation_mae_pixel", "mean"),
        attribution_time_ms_mean=("attribution_time_ms", "mean"),
    ).reset_index()
    summary_path = args.output.with_name(f"{args.output.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {len(frame)} repeat rows to {args.output}")
    print(f"wrote {len(summary)} summary rows to {summary_path}")


if __name__ == "__main__":
    main()
