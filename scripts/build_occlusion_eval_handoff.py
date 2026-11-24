"""Verify and inventory B's six immutable Occlusion eval result directories.

This is a handoff-time inventory, not a claim that artifact hashes were recorded
by the experiment runner. It never runs a model or edits experiment results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from urllib.parse import urlparse

import numpy as np
import yaml
from torchvision.models import DenseNet121_Weights, ResNet50_Weights, VGG16_Weights


ROOT = Path(__file__).resolve().parents[1]
RUN_COMMIT = "6d5645fecfcfdc0af1665435186a81f775faf8e2"
MODELS = ("resnet50", "densenet121", "vgg16")
DATASETS = ("imagenet", "voc")
WEIGHTS = {
    "resnet50": ResNet50_Weights.DEFAULT,
    "densenet121": DenseNet121_Weights.DEFAULT,
    "vgg16": VGG16_Weights.DEFAULT,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def entry(path: Path, kind: str) -> dict:
    return {"path": relative(path), "kind": kind, "bytes": path.stat().st_size, "sha256": sha256(path)}


def rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def image_ids(dataset: str, split: str) -> set[str]:
    return {r["image_id"] for r in rows(ROOT / "data/metadata.csv") if r["dataset"] == dataset and r["split"] == split}


def weight_id(model: str, dataset: str, voc_manifest: dict, cache: Path) -> dict:
    if dataset == "voc":
        match = next(x for x in voc_manifest["checkpoints"] if x["model"] == model)
        path = ROOT / match["path"]
        actual = entry(path, "voc_checkpoint")
        require(actual["bytes"] == match["bytes"] and actual["sha256"] == match["sha256"], f"VOC weight mismatch: {path}")
        return {"source": "VOC20_CHECKPOINT_MANIFEST.json", **actual}
    enum = WEIGHTS[model]
    filename = Path(urlparse(enum.url).path).name
    path = cache / filename
    require(path.is_file(), f"Missing cached ImageNet weight: {path}")
    # The path below is a cache identity, not a repo-relative transferable artifact.
    return {"source": "torchvision", "enum": enum.name, "url": enum.url,
            "cache_filename": filename, "bytes": path.stat().st_size, "sha256": sha256(path)}


def inspect_unit(model: str, dataset: str, weights: dict) -> dict:
    unit = f"{model}_{dataset}"
    result = ROOT / "results/occlusion_eval" / unit
    source_config = ROOT / "configs" / f"occlusion_{unit}.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    report_path = result / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config_hash = report["config_hash"]
    snapshot = result / "configs" / f"{config_hash}_occlusion_{unit}.yaml"
    snapshot_data = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
    require(config == snapshot_data, f"Config snapshot differs: {unit}")
    require(config["dataset"] == {"name": dataset, "split": "eval"}, f"Bad dataset: {unit}")
    require(config["runtime"]["max_images"] is None and config["runtime"]["resume"] is True, f"Bad runtime: {unit}")
    require(config["metrics"]["names"] == ["efficiency_time_ms", "faithfulness_morf_auc_raw"], f"Bad metric names: {unit}")
    require(len(config["metrics"]["deletion_fractions"]) == 21, f"Bad deletion grid: {unit}")
    require(config["model"]["output_activation"] == ("softmax" if dataset == "imagenet" else "sigmoid"), f"Bad activation: {unit}")
    require(report["status"] == "pending_group_acceptance" and report["processed"] == 460 and report["skipped"] == 0 and report["failed"] == 0, f"Bad run status: {unit}")

    metric = rows(result / "per_image.csv")
    predictions = rows(result / "predictions.csv")
    expected = image_ids(dataset, "eval")
    debug = image_ids(dataset, "debug")
    require(len(expected) == 460 and not expected & debug, f"Frozen split problem: {unit}")
    require(len(predictions) == 460 and {r["image_id"] for r in predictions} == expected, f"Prediction IDs: {unit}")
    require(all(r["dataset"] == dataset and r["model"] == model and r["split"] == "eval" and r["checkpoint_sha256"] == weights["sha256"] for r in predictions), f"Prediction identity: {unit}")
    require(len({r["config_hash"] for r in predictions}) == 1, f"Prediction context mismatch: {unit}")
    require(len(metric) == 920 and {r["image_id"] for r in metric} == expected, f"Metric IDs: {unit}")
    require(all(r["dataset"] == dataset and r["model"] == model and r["method"] == "occlusion" for r in metric), f"Metric identity: {unit}")
    require({r["metric"] for r in metric} == {"efficiency_time_ms", "faithfulness_morf_auc_raw"}, f"Metric names: {unit}")
    require(len({(r["image_id"], r["metric"]) for r in metric}) == 920, f"Duplicate metric: {unit}")
    raw = [float(r["value"]) for r in metric if r["metric"] == "faithfulness_morf_auc_raw"]
    time_ms = [float(r["value"]) for r in metric if r["metric"] == "efficiency_time_ms"]
    require(all(math.isfinite(x) and 0 <= x <= 1 for x in raw), f"Invalid raw AUC: {unit}")
    require(all(math.isfinite(x) and x > 0 for x in time_ms), f"Invalid attribution time: {unit}")
    require(math.isclose(mean(raw), report["raw_mean"], abs_tol=1e-9) and math.isclose(stdev(raw), report["raw_std"], abs_tol=1e-9), f"Report/stat mismatch: {unit}")
    require(math.isclose(mean(time_ms), report["attribution_ms_mean"], abs_tol=1e-9), f"Report/time mismatch: {unit}")

    maps = sorted((result / "maps_float").rglob("*.npy"))
    require(len(maps) == 460, f"Float map count: {unit}")
    map_ids = set()
    files = []
    for path in maps:
        map_ids.add(path.stem)
        array = np.load(path, allow_pickle=False)
        require(array.dtype == np.float32 and array.shape == (224, 224) and bool(np.isfinite(array).all()), f"Invalid float map: {path}")
        require(float(array.min()) >= 0 and float(array.max()) <= 1, f"Out-of-range map: {path}")
        files.append(entry(path, "float32_map"))
    require(map_ids == expected, f"Float map IDs: {unit}")
    require(report["constant_map_count"] == 0 and report["float_maps"] == 460 and report["predictions"] == 460 and report["metric_rows"] == 920, f"Validation count mismatch: {unit}")
    for name, kind in (("per_image.csv", "per_image"), ("predictions.csv", "predictions"), ("units.csv", "summary"), ("validation_report.json", "validation_report"), ("run_log.jsonl", "run_log"), ("console.log", "console_log")):
        files.append(entry(result / name, kind))
    files.extend((entry(snapshot, "config_snapshot"), entry(source_config, "source_config")))
    states = sorted((result / "run_state").glob("*.json"))
    require(len(states) == 1, f"Run state count: {unit}")
    files.append(entry(states[0], "run_state"))
    return {"unit": unit, "dataset": dataset, "model": model,
            "config_hash": config_hash, "prediction_context_hash": predictions[0]["config_hash"],
            "weight": weights, "processed": 460, "skipped": 0, "failed": 0,
            "constant_maps": 0, "raw_mean": mean(raw), "raw_std": stdev(raw),
            "attribution_ms_mean": mean(time_ms),
            "console_wall_span_seconds": report["console_wall_span_seconds"],
            "result_directory": relative(result), "files": files}


def build(cache: Path) -> tuple[dict, dict]:
    voc = json.loads((ROOT / "docs/VOC20_CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))
    units = [inspect_unit(model, dataset, weight_id(model, dataset, voc, cache))
             for model in MODELS for dataset in DATASETS]
    version = json.loads((ROOT / "data/DATA_VERSION.json").read_text(encoding="utf-8"))
    detailed = {
        "schema_version": 1, "manifest_kind": "handoff_time_inventory",
        "experiment_commit": RUN_COMMIT,
        "provenance": {
            "runtime_recorded": "config_hash in run_log/state/snapshot; prediction context and checkpoint SHA-256 in predictions.csv",
            "handoff_recalculated": "Every file size and SHA-256 in this inventory, including cached ImageNet and VOC weights",
            "limitation": "The base runner did not cryptographically bind each saved float32 map to a weight hash at write time. Current matching paths/config and post-run hashes do not prove that binding independently."
        },
        "data_version": {"version": version["version"], **entry(ROOT / "data/DATA_VERSION.json", "data_version")},
        "class_order": {"voc20": voc["classes"], "imagenet": "torchvision ImageNet-1K 0..999; see class-index file"},
        "class_index": entry(ROOT / "data/imagenet_class_index.json", "imagenet_class_index"),
        "preprocessing": {"path": "preprocessing/dataset.py", "sha256": sha256(ROOT / "preprocessing/dataset.py"),
                          "resize": [224, 224], "tensor_scale": "RGB [0,1]", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "voc_checkpoint_manifest": entry(ROOT / "docs/VOC20_CHECKPOINT_MANIFEST.json", "voc_checkpoint_manifest"),
        "units": units,
    }
    brief = {key: detailed[key] for key in ("schema_version", "manifest_kind", "experiment_commit", "provenance", "data_version", "class_order", "class_index", "preprocessing", "voc_checkpoint_manifest")}
    brief["status"] = "pending_group_acceptance"
    brief["detailed_manifest"] = {"path": "results/occlusion_handoff/B_OCCLUSION_EVAL_FILES.json"}
    brief["units"] = [{key: u[key] for key in ("unit", "config_hash", "prediction_context_hash", "processed", "skipped", "failed", "constant_maps", "raw_mean", "raw_std", "attribution_ms_mean", "console_wall_span_seconds", "result_directory", "weight")}
                      for u in units]
    return detailed, brief


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-cache", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="write ignored detailed and tracked brief manifests")
    args = parser.parse_args()
    detailed, brief = build(args.torch_cache)
    detailed_path = ROOT / brief["detailed_manifest"]["path"]
    data = (json.dumps(detailed, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    brief["detailed_manifest"].update({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    if args.write:
        detailed_path.parent.mkdir(parents=True, exist_ok=True)
        detailed_path.write_bytes(data)
        (ROOT / "docs/B_OCCLUSION_EVAL_MANIFEST.json").write_text(json.dumps(brief, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        require(detailed_path.read_bytes() == data, "Detailed manifest has drifted")
        saved = json.loads((ROOT / "docs/B_OCCLUSION_EVAL_MANIFEST.json").read_text(encoding="utf-8"))
        require(saved == brief, "Brief manifest has drifted")
    print(json.dumps({"units": len(detailed["units"]), "maps": sum(sum(f["kind"] == "float32_map" for f in u["files"]) for u in detailed["units"]),
                      "detailed_bytes": len(data), "detailed_sha256": brief["detailed_manifest"]["sha256"]}))


if __name__ == "__main__":
    main()
