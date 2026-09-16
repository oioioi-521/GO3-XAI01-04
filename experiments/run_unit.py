"""Run one configured (method, model, dataset) experiment unit.

Example:
    python experiments/run_unit.py --config configs/rise_resnet50_imagenet.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.attribution import RISE  # noqa: E402
from experiments.metrics import faithfulness_morf_auc  # noqa: E402
from models import load_model, predict  # noqa: E402
from preprocessing.dataset import MEAN, STD, MetadataDataset  # noqa: E402

PER_IMAGE_COLUMNS = ["image_id", "dataset", "model", "method", "metric", "value", "time_ms"]
UNIT_COLUMNS = ["method", "model", "dataset", "metric", "mean", "std", "n", "config_hash"]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    required = ("method", "dataset", "model", "attribution", "metrics", "runtime", "output")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"config is missing keys: {missing}")
    if str(config["method"]).lower() != "rise":
        raise ValueError("this W2 runner currently supports method: rise")
    return config


def _config_hash(config: Dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _black_baseline(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [(-mean / std) for mean, std in zip(MEAN, STD)]
    return torch.tensor(values, device=device, dtype=dtype).view(1, 3, 1, 1)


def _read_completed(
    output_path: Path, unit: Dict[str, str], metrics: Iterable[str]
) -> Set[str]:
    if not output_path.is_file():
        return set()
    frame = pd.read_csv(output_path)
    if frame.empty or not set(PER_IMAGE_COLUMNS).issubset(frame.columns):
        return set()
    selected = frame[
        (frame["dataset"] == unit["dataset"])
        & (frame["model"] == unit["model"])
        & (frame["method"] == unit["method"])
        & (frame["metric"].isin(list(metrics)))
    ]
    needed = len(set(metrics))
    counts = selected.groupby("image_id")["metric"].nunique()
    return set(counts[counts >= needed].index.astype(str))


def _append_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def _remove_unit_rows(path: Path, unit: Dict[str, str]) -> None:
    """Remove an existing unit before an explicit forced rerun."""
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    keep = ~(
        (frame["dataset"] == unit["dataset"])
        & (frame["model"] == unit["model"])
        & (frame["method"] == unit["method"])
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame[keep].to_csv(temporary, index=False)
    temporary.replace(path)


def _upsert_unit_summary(
    per_image_path: Path, units_path: Path, unit: Dict[str, str], config_hash: str
) -> None:
    frame = pd.read_csv(per_image_path)
    selected = frame[
        (frame["dataset"] == unit["dataset"])
        & (frame["model"] == unit["model"])
        & (frame["method"] == unit["method"])
    ].copy()
    summary = selected.groupby("metric")["value"].agg(
        mean="mean", std="std", n="count"
    ).reset_index()
    summary.insert(0, "dataset", unit["dataset"])
    summary.insert(0, "model", unit["model"])
    summary.insert(0, "method", unit["method"])
    summary["std"] = summary["std"].fillna(0.0)
    summary["config_hash"] = config_hash
    summary = summary[UNIT_COLUMNS]

    if units_path.is_file():
        old = pd.read_csv(units_path)
        keep = ~(
            (old["dataset"] == unit["dataset"])
            & (old["model"] == unit["model"])
            & (old["method"] == unit["method"])
        )
        summary = pd.concat([old[keep], summary], ignore_index=True)
    units_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = units_path.with_suffix(units_path.suffix + ".tmp")
    summary.to_csv(temporary, index=False)
    temporary.replace(units_path)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(config_path: Path, max_images: int | None = None, force: bool = False) -> None:
    config = _load_config(config_path)
    # Keep downloaded torchvision weights inside the project. This works in
    # restricted lab accounts that cannot write to the default user cache.
    os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
    digest = _config_hash(config)
    dataset_name = str(config["dataset"]["name"]).lower()
    split = str(config["dataset"].get("split", "debug")).lower()
    model_config = config["model"]
    model_name = str(model_config["name"]).lower()
    unit = {"dataset": dataset_name, "model": model_name, "method": "rise"}
    metrics = [str(metric) for metric in config["metrics"]["names"]]
    supported_metrics = {"efficiency_time_ms", "faithfulness_morf_auc"}
    unknown = set(metrics) - supported_metrics
    if unknown:
        raise ValueError(f"unsupported metrics: {sorted(unknown)}")

    runtime = config["runtime"]
    limit = max_images if max_images is not None else runtime.get("max_images")
    device = _select_device(str(runtime.get("device", "auto")))
    seed = int(runtime.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    checkpoint = model_config.get("checkpoint")
    if checkpoint:
        checkpoint = str(_resolve(str(checkpoint)))
    model = load_model(
        name=model_name,
        device=device,
        weights=str(model_config.get("weights", "default")),
        num_classes=int(model_config.get("num_classes", 1000)),
        checkpoint=checkpoint,
    )
    dataset = MetadataDataset(dataset_name, split=split, limit=limit)
    baseline = _black_baseline(device, torch.float32)
    rise = RISE(model=model, **config["attribution"])

    per_image_path = _resolve(str(config["output"]["per_image_csv"]))
    units_path = _resolve(str(config["output"]["units_csv"]))
    if force:
        _remove_unit_rows(per_image_path, unit)
    completed = set() if force or not runtime.get("resume", True) else _read_completed(
        per_image_path, unit, metrics
    )

    snapshot_dir = _resolve(str(config["output"].get("config_dir", "results/configs")))
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, snapshot_dir / f"{digest}_{config_path.name}")
    print(
        f"unit={unit['method']}/{model_name}/{dataset_name}:{split} "
        f"device={device} images={len(dataset)} config_hash={digest}"
    )

    processed = 0
    for sample in tqdm(dataset, desc=f"RISE {model_name} {dataset_name}"):
        image_id = sample["image_id"]
        if image_id in completed:
            continue
        image = sample["image"].unsqueeze(0).to(device)
        target = int(sample["target"])
        predicted, confidence, _ = predict(model, image)

        _synchronize(device)
        started = time.perf_counter()
        attribution = rise.attribute(image, target=target, baseline=baseline)
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if config["output"].get("save_maps", True):
            maps_root = _resolve(str(config["output"].get("maps_dir", "results/maps")))
            unit_maps = maps_root / f"rise_{model_name}_{dataset_name}"
            unit_maps.mkdir(parents=True, exist_ok=True)
            safe_image_id = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in image_id
            )
            Image.fromarray(attribution.mul(255).byte().numpy(), mode="L").save(
                unit_maps / f"{safe_image_id}.png"
            )

        rows: List[Dict[str, Any]] = []
        common = {"image_id": image_id, **unit}
        if "efficiency_time_ms" in metrics:
            rows.append({**common, "metric": "efficiency_time_ms", "value": elapsed_ms, "time_ms": elapsed_ms})
        if "faithfulness_morf_auc" in metrics:
            score = faithfulness_morf_auc(
                model=model,
                image=image,
                target=target,
                attribution=attribution,
                baseline=baseline,
                fractions=config["metrics"].get("deletion_fractions", [0, 0.25, 0.5, 0.75, 1]),
            )
            rows.append({**common, "metric": "faithfulness_morf_auc", "value": score, "time_ms": ""})
        _append_rows(per_image_path, rows)
        processed += 1
        tqdm.write(
            f"{image_id}: target={target} pred={predicted} conf={confidence:.4f} "
            f"rise={elapsed_ms:.1f}ms"
        )

    if per_image_path.is_file():
        _upsert_unit_summary(per_image_path, units_path, unit, digest)
    log_path = _resolve(str(config["output"].get("run_log", "results/run_log.jsonl")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            **unit,
            "split": split,
            "config_hash": digest,
            "device": str(device),
            "processed": processed,
            "skipped": len(completed),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, ensure_ascii=False) + "\n")
    print(f"done: processed={processed}, skipped={len(completed)}, output={per_image_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-images", type=int, help="override runtime.max_images")
    parser.add_argument("--force", action="store_true", help="ignore resume state")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    run(config_path.resolve(), max_images=args.max_images, force=args.force)


if __name__ == "__main__":
    main()
