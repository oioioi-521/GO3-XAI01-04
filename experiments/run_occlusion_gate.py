"""Run an isolated 1- or 40-image GPU Occlusion engineering gate.

This is not an eval or stability-ranking runner. It delegates all predictions,
attribution, MoRF, snapshots, and resume handling to ``run_unit`` and checks
the resulting artifacts without changing their public CSV schemas.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch
import yaml
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import run_unit  # noqa: E402
from experiments.predictions import PREDICTION_COLUMNS, sha256_file  # noqa: E402
from preprocessing.dataset import get_split  # noqa: E402

GRID_21 = [index / 20 for index in range(21)]
METRICS = {"efficiency_time_ms", "faithfulness_morf_auc_raw"}
MODEL_WEIGHTS = {
    "vgg16": models.VGG16_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "densenet121": models.DenseNet121_Weights.DEFAULT,
}
OUTPUT_FILES = {
    "per_image_csv": "per_image.csv",
    "units_csv": "units.csv",
    "predictions_csv": "predictions.csv",
    "config_dir": "configs",
    "run_log": "run_log.jsonl",
    "maps_dir": "maps",
    "float_maps_dir": "maps_float",
}


def _root_for(stage: str, model: str, dataset: str) -> Path:
    return Path("results") / "occlusion_gates" / stage / f"{model}_{dataset}"


def _outputs_for(stage: str, model: str, dataset: str) -> dict[str, Any]:
    root = _root_for(stage, model, dataset)
    return {
        **{key: (root / leaf).as_posix() for key, leaf in OUTPUT_FILES.items()},
        "save_maps": True,
        "save_float_maps": True,
    }


def validate_gate_config(config: dict[str, Any]) -> tuple[str, str]:
    """Reject a formal, legacy-ratio, mixed-output, or non-GPU run."""
    model = str(config["model"]["name"]).lower()
    dataset = str(config["dataset"]["name"]).lower()
    if config["method"] != "occlusion" or model not in MODEL_WEIGHTS or dataset not in {"imagenet", "voc"}:
        raise ValueError("gate requires Occlusion and one of the six supported model/dataset units")
    if config["dataset"].get("split") != "debug" or config["runtime"].get("max_images") != 40:
        raise ValueError("source gate config must be limited to 40 frozen debug images")
    if config["runtime"].get("device") != "cuda" or config["runtime"].get("warmup_runs", 0) < 1:
        raise ValueError("gate requires CUDA and at least one unmeasured warm-up")
    if config["runtime"].get("seed") != 42:
        raise ValueError("gate requires the fixed seed 42")
    if config["runtime"].get("resume") is not True:
        raise ValueError("gate requires resumable runs")
    if set(config["metrics"]["names"]) != METRICS or len(config["metrics"]["names"]) != 2:
        raise ValueError("gate must emit efficiency and only canonical raw MoRF")
    if len(config["metrics"]["deletion_fractions"]) != 21 or any(
        not math.isclose(float(actual), expected, abs_tol=1e-9)
        for actual, expected in zip(config["metrics"]["deletion_fractions"], GRID_21)
    ):
        raise ValueError("gate requires the 21-point deletion grid")
    for key in ("window_height", "window_width", "stride_height", "stride_width", "perturbations_per_eval"):
        value = config["attribution"].get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"attribution.{key} must be a positive integer")
    expected_model = (
        {"weights": "default", "num_classes": 1000, "task_type": "multiclass", "output_activation": "softmax", "checkpoint": None}
        if dataset == "imagenet" else
        {"weights": "none", "num_classes": 20, "task_type": "multilabel", "output_activation": "sigmoid", "checkpoint": f"models/checkpoints/{model}_voc20.pt"}
    )
    if any(config["model"].get(key) != value for key, value in expected_model.items()):
        raise ValueError("model output semantics or checkpoint path differs from the frozen gate contract")
    if config["output"] != _outputs_for("debug40", model, dataset):
        raise ValueError("gate output must be isolated under results/occlusion_gates/debug40")
    return model, dataset


def effective_gate_config(config_path: Path, images: int) -> tuple[dict[str, Any], Path]:
    if images not in (1, 40):
        raise ValueError("gate size must be one image or the full 40-image debug split")
    config = run_unit._load_config(config_path)
    model, dataset = validate_gate_config(config)
    if images == 40:
        return config, config_path
    single = copy.deepcopy(config)
    single["runtime"]["max_images"] = 1
    single["output"] = _outputs_for("single", model, dataset)
    generated_path = PROJECT_ROOT / _root_for("single", model, dataset) / "effective_config.yaml"
    return single, generated_path


def _write_effective_config(config: dict[str, Any], path: Path) -> None:
    """Write only under ignored gate results; never replace differing user data."""
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(f"different effective config already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _verify_frozen_data(dataset: str, images: int) -> dict[str, Any]:
    data_root = PROJECT_ROOT / "data"
    version = json.loads((data_root / "DATA_VERSION.json").read_text(encoding="utf-8"))
    modes = {}
    for name, expected in version["sha256"].items():
        raw = (data_root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() == expected:
            modes[name] = "raw"
        elif b"\r\n" in raw and hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == expected:
            modes[name] = "CRLF-to-LF only"
        else:
            raise ValueError(f"frozen data SHA-256 mismatch beyond line endings: {name}")
    for name in ("imagenet", "voc"):
        raw_dir = data_root / name / "raw"
        count = sum(path.is_file() for path in raw_dir.iterdir())
        if count != version["datasets"][name]["raw_images"]:
            raise ValueError(f"frozen {name} image count is {count}, not {version['datasets'][name]['raw_images']}")
    rows = get_split(dataset, "debug")
    if len(rows) != 40:
        raise ValueError(f"frozen {dataset} debug split has {len(rows)} rather than 40 images")
    for row in rows.iloc[:images].itertuples():
        relative = Path(str(row.image_path).replace("\\", "/"))
        if not (data_root / relative).is_file():
            raise FileNotFoundError(data_root / relative)
    return {"version": version["version"], "hash_modes": modes, "debug_images": len(rows)}


def _verify_weight(model: str, dataset: str) -> dict[str, Any]:
    if dataset == "voc":
        manifest = json.loads((PROJECT_ROOT / "docs/VOC20_CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))
        entry = next((entry for entry in manifest["checkpoints"] if entry["model"] == model), None)
        if entry is None:
            raise ValueError(f"VOC checkpoint not listed in manifest: {model}")
        path = PROJECT_ROOT / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"VOC checkpoint size/SHA-256 mismatch: {path}")
        return {"source": "VOC20_CHECKPOINT_MANIFEST.json", "path": entry["path"], "sha256": entry["sha256"]}
    os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
    weights = MODEL_WEIGHTS[model]
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights.url).path).name
    if not path.is_file():
        raise FileNotFoundError(f"official torchvision weight is not cached; gate will not download: {path}")
    return {"source": weights.url, "path": str(path), "sha256": sha256_file(path)}


def _require_idle_gpu() -> None:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    )
    if query.stdout.strip():
        raise RuntimeError(f"another GPU compute process is active; no gate started: {query.stdout.strip()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback is permitted for this gate")


def _read_csv(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f"unexpected CSV schema: {path}")
        return list(reader)


def inspect_gate_outputs(config: dict[str, Any], config_path: Path, images: int) -> dict[str, Any]:
    model = config["model"]["name"]
    dataset = config["dataset"]["name"]
    digest = run_unit._config_hash(config)
    expected_ids = set(get_split(dataset, "debug").iloc[:images]["image_id"].astype(str))
    output = config["output"]
    per_image = _read_csv(PROJECT_ROOT / output["per_image_csv"], run_unit.PER_IMAGE_COLUMNS)
    keys = Counter((row["image_id"], row["metric"]) for row in per_image)
    expected_keys = {(image_id, metric) for image_id in expected_ids for metric in METRICS}
    if set(keys) != expected_keys or any(count != 1 for count in keys.values()):
        raise ValueError("missing, duplicate, or extra per-image gate metric rows")
    if any(row["method"] != "occlusion" or row["model"] != model or row["dataset"] != dataset for row in per_image):
        raise ValueError("per-image unit identity mismatch")
    raw_values = [float(row["value"]) for row in per_image if row["metric"] == "faithfulness_morf_auc_raw"]
    times_ms = [float(row["value"]) for row in per_image if row["metric"] == "efficiency_time_ms"]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in raw_values):
        raise ValueError("raw MoRF is not finite and bounded")
    if any(not math.isfinite(value) or value <= 0 for value in times_ms):
        raise ValueError("attribution timing is not finite and positive")
    summary = _read_csv(PROJECT_ROOT / output["units_csv"], run_unit.UNIT_COLUMNS)
    if len(summary) != 2 or {row["metric"] for row in summary} != METRICS or any(row["config_hash"] != digest or int(row["n"]) != images for row in summary):
        raise ValueError("unit summary metrics, n, or config hash mismatch")
    snapshot = PROJECT_ROOT / output["config_dir"] / f"{digest}_{config_path.name}"
    if run_unit._config_hash(yaml.safe_load(snapshot.read_text(encoding="utf-8"))) != digest:
        raise ValueError("effective configuration snapshot hash mismatch")
    predictions = _read_csv(PROJECT_ROOT / output["predictions_csv"], PREDICTION_COLUMNS)
    if len(predictions) != images or {row["image_id"] for row in predictions} != expected_ids:
        raise ValueError("prediction table does not match gate images")
    classes = int(config["model"]["num_classes"])
    if any(
        row["dataset"] != dataset or row["split"] != "debug" or row["model"] != model
        or not 0 <= int(row["target_class_id"]) < classes
        or not 0 <= int(row["predicted_class_id"]) < classes
        or not math.isfinite(float(row["confidence"]))
        or not 0 <= float(row["confidence"]) <= 1
        for row in predictions
    ):
        raise ValueError("prediction identity, class, or confidence is invalid")
    maps_dir = PROJECT_ROOT / output["float_maps_dir"] / f"occlusion_{model}_{dataset}"
    constant = []
    for image_id in expected_ids:
        safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in image_id)
        heatmap = np.load(maps_dir / f"{safe_id}.npy", allow_pickle=False)
        if heatmap.dtype != np.float32 or heatmap.shape != (224, 224) or not np.isfinite(heatmap).all():
            raise ValueError(f"invalid float32 attribution map for {image_id}")
        if heatmap.min() < 0 or heatmap.max() > 1:
            raise ValueError(f"attribution map outside [0,1] for {image_id}")
        if float(heatmap.max() - heatmap.min()) <= 1e-12:
            constant.append(image_id)
    return {
        "images": images,
        "failed_images": 0,
        "constant_maps": sorted(constant),
        "constant_map_count": len(constant),
        "attribution_time_ms_total": sum(times_ms),
        "attribution_time_ms_mean": sum(times_ms) / images,
        "attribution_time_ms_max": max(times_ms),
        "raw_morf_mean": sum(raw_values) / images,
        "raw_morf_min": min(raw_values),
        "raw_morf_max": max(raw_values),
        "config_hash": digest,
        "float_map_dir": output["float_maps_dir"],
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    with (path.parent / "gate_runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False) + "\n")


def run_gate(config_path: Path, images: int) -> dict[str, Any]:
    config, effective_path = effective_gate_config(config_path, images)
    model, dataset = config["model"]["name"], config["dataset"]["name"]
    data_provenance = _verify_frozen_data(dataset, images)
    weight_provenance = _verify_weight(model, dataset)
    _require_idle_gpu()
    if images == 1:
        _write_effective_config(config, effective_path)
    output_dir = PROJECT_ROOT / _root_for("single" if images == 1 else "debug40", model, dataset)
    report_path = output_dir / "gate_report.json"
    report: dict[str, Any] = {
        "model": model, "dataset": dataset, "split": "debug", "stage": "single" if images == 1 else "debug40",
        "data_provenance": data_provenance, "weight_provenance": weight_provenance,
        "attribution_config": config["attribution"], "deletion_grid_points": 21,
        "warmup_runs": config["runtime"]["warmup_runs"],
    }
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        run_unit.run(effective_path)
        report.update(inspect_gate_outputs(config, effective_path, images))
        report["status"] = "passed"
    except BaseException as error:
        report.update({"status": "failed", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        report["wall_seconds_this_process"] = time.perf_counter() - start
        try:
            report["torch_cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1048576
            report["torch_cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1048576
        except RuntimeError as error:
            report["cuda_peak_error"] = str(error)
        _write_report(report_path, report)
        print(f"gate report: {report_path} status={report['status']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=int, choices=(1, 40), required=True)
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    run_gate(path.resolve(), args.images)


if __name__ == "__main__":
    main()
