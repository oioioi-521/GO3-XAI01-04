"""Prepare isolated Occlusion stability gates from immutable debug40 artifacts.

Never overwrites a destination or writes into the source gate directory.
The copied legacy maps are verified by run_stability before sidecars are added.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
UNITS = tuple(f"{model}_{dataset}" for model in ("resnet50", "densenet121", "vgg16")
              for dataset in ("imagenet", "voc"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path, ledger: list[dict]) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = sha256(source)
    if sha256(destination) != digest or destination.stat().st_size != source.stat().st_size:
        raise ValueError(f"copy mismatch: {source} -> {destination}")
    ledger.append({"source": source.relative_to(ROOT).as_posix(),
                   "copy": destination.relative_to(ROOT).as_posix(),
                   "bytes": source.stat().st_size, "sha256": digest})


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_subset(source: Path, destination: Path, ids: set[str], ledger: list[dict]) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        assert columns is not None
        rows = [row for row in reader if row["image_id"] in ids]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    ledger.append({"source": source.relative_to(ROOT).as_posix(),
                   "source_sha256": sha256(source), "derived_subset": destination.relative_to(ROOT).as_posix(),
                   "image_ids": sorted(ids), "rows": len(rows), "bytes": destination.stat().st_size,
                   "sha256": sha256(destination)})


def prepare(stage: str, unit: str) -> None:
    count = 1 if stage in ("single", "probe") else 40
    destination = ROOT / "results/occlusion_stability_gates" / stage / unit
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing gate: {destination}")
    source = ROOT / "results/occlusion_gates/debug40" / unit
    base_path = ROOT / "configs" / f"occlusion_gate_{unit}.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    protocol_path = ROOT / "configs/stability_protocol_v1.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    model, dataset = unit.rsplit("_", 1)
    if (base["model"]["name"], base["dataset"]) != (model, {"name": dataset, "split": "debug"}):
        raise ValueError(f"source config mismatch: {unit}")
    if base["attribution"] != {"window_height": 32, "window_width": 32,
                               "stride_height": 16, "stride_width": 16,
                               "perturbations_per_eval": 16}:
        raise ValueError(f"source attribution mismatch: {unit}")
    if base["metrics"]["names"] != ["efficiency_time_ms", "faithfulness_morf_auc_raw"]:
        raise ValueError(f"source metric mismatch: {unit}")
    if base["runtime"]["max_images"] != 40:
        raise ValueError(f"source was not a 40-image gate: {unit}")
    source_predictions = csv_rows(source / "predictions.csv")
    if len(source_predictions) != 40 or len({row["image_id"] for row in source_predictions}) != 40:
        raise ValueError(f"source predictions are not 40 unique images: {unit}")
    # Dataset order, not CSV order, determines MetadataDataset's first image.
    metadata = csv_rows(ROOT / "data/metadata.csv")
    expected = [row["image_id"] for row in metadata
                if row["dataset"] == dataset and row["split"] == "debug"][:count]
    if len(expected) != count or not set(expected).issubset({row["image_id"] for row in source_predictions}):
        raise ValueError(f"source image set differs from frozen debug metadata: {unit}")
    selected = set(expected)
    ledger: list[dict] = []
    destination.mkdir(parents=True)
    if stage in ("single", "probe"):
        write_subset(source / "per_image.csv", destination / "per_image.csv", selected, ledger)
        write_subset(source / "predictions.csv", destination / "predictions.csv", selected, ledger)
        # A single-image summary must not pretend to be the source 40-image summary.
        metrics = csv_rows(destination / "per_image.csv")
        original_summary = csv_rows(source / "units.csv")
        summary = [{**row, "mean": next(item["value"] for item in metrics if item["metric"] == row["metric"]),
                    "std": "0", "n": "1"} for row in original_summary]
        with (destination / "units.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(original_summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        ledger.append({"source": (source / "units.csv").relative_to(ROOT).as_posix(),
                       "source_sha256": sha256(source / "units.csv"),
                       "derived_subset": (destination / "units.csv").relative_to(ROOT).as_posix(),
                       "bytes": (destination / "units.csv").stat().st_size,
                       "sha256": sha256(destination / "units.csv")})
    else:
        for filename in ("per_image.csv", "predictions.csv", "units.csv"):
            copy_verified(source / filename, destination / filename, ledger)
    source_maps = source / "maps_float" / f"occlusion_{unit}"
    for image_id in expected:
        copy_verified(source_maps / f"{image_id}.npy",
                      destination / "maps_float" / f"occlusion_{unit}" / f"{image_id}.npy", ledger)

    prefix = destination.relative_to(ROOT).as_posix()
    base["runtime"]["max_images"] = count
    base["output"].update({
        "per_image_csv": f"{prefix}/per_image.csv", "units_csv": f"{prefix}/units.csv",
        "predictions_csv": f"{prefix}/predictions.csv", "config_dir": f"{prefix}/base_configs",
        "run_log": f"{prefix}/base_run_log.jsonl", "state_dir": f"{prefix}/base_run_state",
        "maps_dir": f"{prefix}/maps", "float_maps_dir": f"{prefix}/maps_float",
    })
    protocol["runtime"]["device"] = "cuda"
    protocol["output"] = {
        "trace_csv": f"{prefix}/stability_trace.csv", "state_dir": f"{prefix}/stability_state",
        "config_dir": f"{prefix}/stability_configs", "run_log": f"{prefix}/stability_run_log.jsonl",
    }
    (destination / "base.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    (destination / "protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    source_report = source / "gate_report.json"
    report = json.loads(source_report.read_text(encoding="utf-8"))
    if report["status"] != "passed" or report["images"] != 40:
        raise ValueError(f"source gate report failed: {unit}")
    (destination / "input_manifest.json").write_text(json.dumps({
        "stage": stage, "unit": unit, "split": "debug", "selected_image_ids": expected,
        "source_config": base_path.relative_to(ROOT).as_posix(),
        "source_config_sha256": sha256(base_path), "source_gate_report_sha256": sha256(source_report),
        "source_gate_config_hash": report["config_hash"],
        "source_prediction_checkpoint_sha256": sorted({r["checkpoint_sha256"] for r in source_predictions}),
        "generated_base_config_sha256": sha256(destination / "base.yaml"),
        "generated_protocol_sha256": sha256(destination / "protocol.yaml"),
        "files": ledger,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": stage, "unit": unit, "images": count, "copied_maps": count,
                      "destination": prefix}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("single", "probe", "debug40"), required=True)
    parser.add_argument("--unit", choices=UNITS, required=True)
    args = parser.parse_args()
    prepare(args.stage, args.unit)


if __name__ == "__main__":
    main()
