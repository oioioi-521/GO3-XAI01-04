"""Generate or resume a method-independent prediction table from a YAML config."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.predictions import PredictionStore, build_prediction_context  # noqa: E402
from experiments.run_unit import _config_hash, _effective_config, _load_config, _resolve, _select_device  # noqa: E402
from models import load_model, predict  # noqa: E402
from preprocessing.dataset import MetadataDataset  # noqa: E402


def run(config_path: Path, max_images: int | None = None) -> None:
    config = _effective_config(_load_config(config_path), max_images)
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
    dataset_name = str(config["dataset"]["name"]).lower()
    split = str(config["dataset"].get("split", "debug")).lower()
    model_config = config["model"]
    model_name = str(model_config["name"]).lower()
    output_activation = str(model_config.get("output_activation", "sigmoid" if dataset_name == "voc" else "softmax"))
    runtime = config["runtime"]
    limit = runtime.get("max_images")
    device = _select_device(str(runtime.get("device", "auto")))

    checkpoint = model_config.get("checkpoint")
    checkpoint_path = _resolve(str(checkpoint)) if checkpoint else None
    model = load_model(
        name=model_name,
        device=device,
        weights=str(model_config.get("weights", "default")),
        num_classes=int(model_config.get("num_classes", 1000)),
        checkpoint=str(checkpoint_path) if checkpoint_path else None,
    )
    dataset = MetadataDataset(dataset_name, split=split, limit=limit)
    prediction_context = build_prediction_context(
        dataset=dataset_name,
        split=split,
        model_config=model_config,
        checkpoint=checkpoint_path,
    )
    prediction_path = _resolve(str(config["output"].get("predictions_csv", "results/predictions.csv")))
    store = PredictionStore(prediction_path, prediction_context)
    execution_config_hash = _config_hash(config)

    snapshot_dir = prediction_path.parent / "configs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / (
        f"prediction_{prediction_context['config_hash']}_{execution_config_hash}_{config_path.name}"
    )
    temporary_snapshot = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    with temporary_snapshot.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    temporary_snapshot.replace(snapshot_path)

    processed = 0
    skipped = 0
    expected_classes = int(model_config.get("num_classes", 1000))
    for sample in tqdm(dataset, desc=f"PREDICT {model_name} {dataset_name}"):
        image_id = sample["image_id"]
        target = int(sample["target"])
        if store.get(image_id, target) is not None:
            skipped += 1
            continue
        image = sample["image"].unsqueeze(0).to(device)
        predicted, confidence, logits = predict(model, image, output_activation=output_activation)
        if logits.shape != (1, expected_classes) or not torch.isfinite(logits).all():
            raise ValueError(f"model output has unexpected values or shape {tuple(logits.shape)}")
        store.record(
            image_id=image_id,
            target_class_id=target,
            target_class_name=str(sample["class_name"]),
            predicted_class_id=predicted,
            confidence=confidence,
        )
        processed += 1

    correct, total, accuracy = store.accuracy()
    log_path = prediction_path.parent / "prediction_log.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            **prediction_context,
            "execution_config_hash": execution_config_hash,
            "device": str(device),
            "processed": processed,
            "skipped": skipped,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, ensure_ascii=False) + "\n")
    print(
        f"prediction_hash={prediction_context['config_hash']} execution_hash={execution_config_hash} processed={processed} "
        f"skipped={skipped} accuracy={correct}/{total}={accuracy:.6f} output={prediction_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-images", type=int, help="override runtime.max_images and snapshot it")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    run(config_path.resolve(), max_images=args.max_images)


if __name__ == "__main__":
    main()
