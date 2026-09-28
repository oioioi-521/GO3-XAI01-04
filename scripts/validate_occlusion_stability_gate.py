"""Validate a copied debug Occlusion stability gate without changing source results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.stability import stable_seed


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def validate(gate: Path) -> dict:
    gate = gate.resolve()
    if not gate.is_relative_to((ROOT / "results/occlusion_stability_gates").resolve()):
        raise ValueError("not an isolated gate")
    manifest = json.loads((gate / "input_manifest.json").read_text(encoding="utf-8"))
    unit = manifest["unit"]
    model, dataset = unit.rsplit("_", 1)
    expected = set(manifest["selected_image_ids"])
    for entry in manifest["files"]:
        original = ROOT / entry["source"]
        if "copy" in entry:
            copied = ROOT / entry["copy"]
            assert digest(original) == entry["sha256"]
            if copied.suffix == ".npy" or copied.name == "predictions.csv":
                assert digest(copied) == entry["sha256"]
        else:
            assert digest(original) == entry["source_sha256"]
    prediction_rows = rows(gate / "predictions.csv")
    assert len(prediction_rows) == len(expected)
    assert {row["image_id"] for row in prediction_rows} == expected
    metric_rows = rows(gate / "per_image.csv")
    assert len(metric_rows) == 4 * len(expected)
    assert {(r["image_id"], r["metric"]) for r in metric_rows} == {
        (image_id, metric) for image_id in expected for metric in (
            "efficiency_time_ms", "faithfulness_morf_auc_raw", "stability_spearman",
            "stability_valid_rate")}
    trace_files = list(gate.glob("stability_trace_*.csv"))
    assert len(trace_files) == 1
    traces = rows(trace_files[0])
    assert len(traces) == 5 * len(expected)
    assert {(r["image_id"], int(r["repeat"])) for r in traces} == {
        (image_id, repeat) for image_id in expected for repeat in range(5)}
    run_lines = [json.loads(line) for line in (gate / "stability_run_log.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert run_lines[0]["processed"] == len(expected) and run_lines[0]["skipped"] == 0
    assert run_lines[-1]["processed"] == 0 and run_lines[-1]["skipped"] == len(expected)
    config_hash = run_lines[0]["config_hash"]
    assert all(line["config_hash"] == config_hash and line["split"] == "debug" for line in run_lines)
    assert all(r["config_hash"] == config_hash for r in traces)
    assert all(r["model"] == model and r["dataset"] == dataset and r["method"] == "occlusion"
               for r in traces + metric_rows)
    status_counts: dict[str, int] = {}
    for trace in traces:
        assert int(trace["seed"]) == stable_seed(dataset, trace["image_id"], int(trace["repeat"]))
        assert int(trace["target"]) in range(1000 if dataset == "imagenet" else 20)
        assert float(trace["sigma"]) == 0.005
        score = float(trace["score"])
        assert math.isfinite(score) and 0 <= score <= 1
        assert math.isfinite(float(trace["attribution_time_ms"]))
        status_counts[trace["status"]] = status_counts.get(trace["status"], 0) + 1
    by_metric = {(r["image_id"], r["metric"]): float(r["value"]) for r in metric_rows}
    for image_id in expected:
        image_traces = [r for r in traces if r["image_id"] == image_id]
        assert math.isclose(by_metric[image_id, "stability_spearman"],
                            sum(float(r["score"]) for r in image_traces) / 5,
                            abs_tol=1e-12)
        assert math.isclose(by_metric[image_id, "stability_valid_rate"],
                            sum(r["status"] == "valid" for r in image_traces) / 5,
                            abs_tol=1e-12)
        for metric in ("faithfulness_morf_auc_raw", "stability_spearman", "stability_valid_rate"):
            assert 0 <= by_metric[image_id, metric] <= 1
        image_map = gate / "maps_float" / f"occlusion_{unit}" / f"{image_id}.npy"
        map_array = np.load(image_map, allow_pickle=False)
        assert map_array.shape == (224, 224) and map_array.dtype == np.float32
        assert np.isfinite(map_array).all()
        provenance = json.loads(Path(str(image_map) + ".provenance.json").read_text(encoding="utf-8"))
        assert provenance["binding"] == "verified_recomputation"
        assert provenance["map_sha256"] == digest(image_map)
        assert provenance["base_config_hash"] == run_lines[0]["base_config_hash"]
        assert provenance["checkpoint_sha256"] == run_lines[0]["checkpoint_sha256"]
        assert provenance["image_id"] == image_id and provenance["split"] == "debug"
    summaries = rows(gate / "units.csv")
    assert len(summaries) == 4
    assert {r["metric"] for r in summaries} == {r["metric"] for r in metric_rows}
    for row in summaries:
        assert int(row["n"]) == len(expected)
        if row["metric"].startswith("stability_"):
            assert row["config_hash"] == config_hash
        assert math.isfinite(float(row["mean"])) and math.isfinite(float(row["std"]))
    process_file = gate / "gate_process_runs.jsonl"
    processes = [json.loads(line) for line in process_file.read_text(encoding="utf-8").splitlines()] if process_file.exists() else []
    assert all(record["status"] == "success" for record in processes)
    report = {
        "status": "passed", "stage": manifest["stage"], "unit": unit,
        "images": len(expected), "trace_rows": len(traces), "stability_metric_rows": 2 * len(expected),
        "config_hash": config_hash, "reference_binding": "verified_recomputation",
        "status_counts": status_counts,
        "stability_spearman_mean": next(float(r["mean"]) for r in summaries if r["metric"] == "stability_spearman"),
        "stability_valid_rate_mean": next(float(r["mean"]) for r in summaries if r["metric"] == "stability_valid_rate"),
        "attribution_time_ms_mean": sum(float(r["attribution_time_ms"]) for r in traces) / len(traces),
        "process_wall_seconds": processes[0]["wall_seconds_this_process"] if processes else None,
        "cuda_peak_allocated_mib": processes[0]["cuda_peak_allocated_mib"] if processes else None,
        "cuda_peak_reserved_mib": processes[0]["cuda_peak_reserved_mib"] if processes else None,
        "failed_runs": sum(record["status"] != "success" for record in processes),
    }
    (gate / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.gate_dir), ensure_ascii=False))
