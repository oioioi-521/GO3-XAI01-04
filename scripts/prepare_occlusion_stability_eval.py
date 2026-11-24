"""Preflight and back up six immutable Occlusion base-eval units.

Use the original base YAML paths so stability attaches to the correct config
identity. Never overwrites a backup or an existing stability output directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import yaml

from experiments.run_unit import _config_hash
from preprocessing.dataset import MetadataDataset


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/occlusion_stability_eval"
UNITS = tuple(f"{model}_{dataset}" for model in ("resnet50", "densenet121", "vgg16")
              for dataset in ("imagenet", "voc"))
EXPECTED_PROTOCOL_SHA = "324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def preflight() -> tuple[list[dict], dict]:
    blob = subprocess.check_output(["git", "show", "HEAD:configs/stability_protocol_v1.yaml"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != EXPECTED_PROTOCOL_SHA:
        raise ValueError("frozen stability protocol Git blob SHA mismatch")
    protocol = yaml.safe_load(blob)
    assert protocol["repeats"] == 5 and protocol["sigma"] == 0.005
    assert protocol["target"] == "metadata" and protocol["similarity"] == "spearman"
    assert protocol["degenerate_score"] == 0.0
    base_manifest = json.loads((ROOT / "docs/B_OCCLUSION_EVAL_MANIFEST.json").read_text(encoding="utf-8"))
    voc_manifest = json.loads((ROOT / "docs/VOC20_CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))
    expected_voc = {entry["model"]: entry for entry in voc_manifest["checkpoints"]}
    by_unit = {entry["unit"]: entry for entry in base_manifest["units"]}
    assert set(by_unit) == set(UNITS)
    home_cache = Path.home() / ".cache/torch/hub/checkpoints"
    reports: list[dict] = []
    for unit in UNITS:
        model, dataset_name = unit.rsplit("_", 1)
        config_path = ROOT / "configs" / f"occlusion_{unit}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        source = ROOT / "results/occlusion_eval" / unit
        original = by_unit[unit]
        assert config["method"] == "occlusion"
        assert config["dataset"] == {"name": dataset_name, "split": "eval"}
        assert config["model"]["name"] == model
        assert config["model"]["output_activation"] == ("softmax" if dataset_name == "imagenet" else "sigmoid")
        assert config["runtime"]["max_images"] is None and config["runtime"]["resume"] is True
        assert config["attribution"] == {"window_height": 32, "window_width": 32,
                                           "stride_height": 16, "stride_width": 16,
                                           "perturbations_per_eval": 16}
        assert config["metrics"]["names"] == ["efficiency_time_ms", "faithfulness_morf_auc_raw"]
        assert len(config["metrics"]["deletion_fractions"]) == 21
        assert _config_hash(config) == original["config_hash"]
        assert (ROOT / config["output"]["per_image_csv"]).resolve() == (source / "per_image.csv").resolve()
        assert (ROOT / config["output"]["units_csv"]).resolve() == (source / "units.csv").resolve()
        assert (ROOT / config["output"]["predictions_csv"]).resolve() == (source / "predictions.csv").resolve()
        weight = original["weight"]
        if dataset_name == "voc":
            assert weight["sha256"] == expected_voc[model]["sha256"]
            weight_path = ROOT / expected_voc[model]["path"]
        else:
            assert weight["source"] == "torchvision"
            weight_path = home_cache / weight["cache_filename"]
        assert weight_path.stat().st_size == weight["bytes"]
        assert sha256(weight_path) == weight["sha256"]
        predictions = rows(source / "predictions.csv")
        metric_rows = rows(source / "per_image.csv")
        summary = rows(source / "units.csv")
        frozen = MetadataDataset(dataset_name, split="eval")
        expected = {str(frozen[i]["image_id"]): int(frozen[i]["target"]) for i in range(len(frozen))}
        assert len(expected) == 460 and len(predictions) == 460 and len(metric_rows) == 920
        assert {p["image_id"] for p in predictions} == set(expected)
        assert all(p["dataset"] == dataset_name and p["split"] == "eval" and p["model"] == model
                   and int(p["target_class_id"]) == expected[p["image_id"]]
                   and p["checkpoint_sha256"] == weight["sha256"]
                   and p["config_hash"] == original["prediction_context_hash"] for p in predictions)
        assert len(summary) == 2 and all(r["config_hash"] == original["config_hash"]
                                         and int(r["n"]) == 460 for r in summary)
        assert {(r["image_id"], r["metric"]) for r in metric_rows} == {
            (image_id, metric) for image_id in expected
            for metric in ("efficiency_time_ms", "faithfulness_morf_auc_raw")}
        state = json.loads((source / "run_state" / f"occlusion_{unit}.json").read_text(encoding="utf-8"))
        assert state["config_hash"] == original["config_hash"]
        maps = source / "maps_float" / f"occlusion_{unit}"
        assert {p.stem for p in maps.glob("*.npy")} == set(expected)
        assert not list(maps.glob("*.provenance.json"))
        for path in maps.glob("*.npy"):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            assert array.dtype == np.float32 and array.shape == (224, 224)
        report = json.loads((source / "validation_report.json").read_text(encoding="utf-8"))
        assert report["total"] == 460 and report["failed"] == 0
        reports.append({"unit": unit, "base_config": config_path.relative_to(ROOT).as_posix(),
                        "base_config_sha256": sha256(config_path),
                        "base_config_hash": original["config_hash"],
                        "prediction_context_hash": original["prediction_context_hash"],
                        "weight": {"path": str(weight_path.relative_to(ROOT)) if weight_path.is_relative_to(ROOT)
                                            else weight["cache_filename"],
                                   "bytes": weight["bytes"], "sha256": weight["sha256"]},
                        "image_count": 460})
    return reports, protocol


def prepare() -> None:
    reports, protocol = preflight()
    if DEST.exists():
        raise FileExistsError(f"refusing to overwrite stability output/backup: {DEST}")
    backup = DEST / "backup"
    backup.mkdir(parents=True)
    inventory = []
    for report in reports:
        unit = report["unit"]
        source = ROOT / "results/occlusion_eval" / unit
        for original in sorted(source.rglob("*")):
            if not original.is_file():
                continue
            relative = original.relative_to(source)
            copied = backup / unit / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            if copied.exists():
                raise FileExistsError(copied)
            shutil.copy2(original, copied)
            digest = sha256(original)
            if copied.stat().st_size != original.stat().st_size or sha256(copied) != digest:
                raise ValueError(f"backup mismatch: {original}")
            inventory.append({"source": original.relative_to(ROOT).as_posix(),
                              "backup": copied.relative_to(ROOT).as_posix(),
                              "bytes": original.stat().st_size, "sha256": digest})
        output = DEST / unit
        output.mkdir()
        unit_protocol = dict(protocol)
        unit_protocol["runtime"] = {**protocol["runtime"], "device": "cuda"}
        unit_protocol["output"] = {
            "trace_csv": f"results/occlusion_stability_eval/{unit}/stability_trace.csv",
            "state_dir": f"results/occlusion_stability_eval/{unit}/state",
            "config_dir": f"results/occlusion_stability_eval/{unit}/snapshots",
            "run_log": f"results/occlusion_stability_eval/{unit}/run_log.jsonl",
        }
        protocol_path = output / "protocol.yaml"
        protocol_path.write_text(yaml.safe_dump(unit_protocol, sort_keys=False), encoding="utf-8")
        report["protocol_path"] = protocol_path.relative_to(ROOT).as_posix()
        report["protocol_sha256"] = sha256(protocol_path)
    manifest = {"schema": 1, "frozen_protocol_git_blob_sha256": EXPECTED_PROTOCOL_SHA,
                "base_eval_code_commit": "6d5645fecfcfdc0af1665435186a81f775faf8e2",
                "units": reports, "source_file_count": len(inventory),
                "source_files": inventory}
    manifest_path = backup / "backup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "units": len(reports), "files": len(inventory),
                      "bytes": sum(row["bytes"] for row in inventory),
                      "backup_manifest_sha256": sha256(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    prepare()
