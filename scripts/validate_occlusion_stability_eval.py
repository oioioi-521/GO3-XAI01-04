"""Independently check a complete 460-image stability unit and base preservation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

import yaml

from experiments.run_unit import _config_hash
from experiments.stability import stable_seed
from scripts.prepare_occlusion_stability_eval import DEST, ROOT, UNITS, rows, sha256


BASE_METRICS = {"efficiency_time_ms", "faithfulness_morf_auc_raw"}
STABILITY_METRICS = {"stability_spearman", "stability_valid_rate"}


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)


def validate(unit: str, require_resume: bool = True) -> dict:
    if unit not in UNITS:
        raise ValueError(f"unknown unit: {unit}")
    model, dataset = unit.rsplit("_", 1)
    result = ROOT / "results/occlusion_eval" / unit
    output = DEST / unit
    backup = DEST / "backup" / unit
    manifest = json.loads((DEST / "backup/backup_manifest.json").read_text(encoding="utf-8"))
    identity = next(item for item in manifest["units"] if item["unit"] == unit)
    base_config = yaml.safe_load((ROOT / identity["base_config"]).read_text(encoding="utf-8"))
    protocol = yaml.safe_load((output / "protocol.yaml").read_text(encoding="utf-8"))
    assert _config_hash(base_config) == identity["base_config_hash"]
    assert sha256(output / "protocol.yaml") == identity["protocol_sha256"]
    assert protocol["repeats"] == 5 and protocol["sigma"] == 0.005
    assert protocol["target"] == "metadata" and protocol["similarity"] == "spearman"
    assert protocol["degenerate_score"] == 0.0
    assert protocol["runtime"]["device"] == "cuda"
    predictions = rows(result / "predictions.csv")
    assert len(predictions) == 460 and len({p["image_id"] for p in predictions}) == 460
    ids = {p["image_id"] for p in predictions}
    targets = {p["image_id"]: int(p["target_class_id"]) for p in predictions}
    assert all(p["split"] == "eval" and p["dataset"] == dataset and p["model"] == model
               and p["checkpoint_sha256"] == identity["weight"]["sha256"] for p in predictions)
    trace_paths = list(output.glob(f"stability_trace_occlusion_{unit}_eval_*.csv"))
    assert len(trace_paths) == 1, trace_paths
    trace = rows(trace_paths[0])
    assert len(trace) == 2300
    assert {(r["image_id"], int(r["repeat"])) for r in trace} == {
        (image_id, repeat) for image_id in ids for repeat in range(5)}
    runner_logs = [json.loads(line) for line in (output / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(line["processed"] for line in runner_logs) == 460
    if require_resume:
        assert runner_logs[-1]["processed"] == 0 and runner_logs[-1]["skipped"] == 460
    hashes = {line["config_hash"] for line in runner_logs}
    assert len(hashes) == 1
    config_hash = hashes.pop()
    assert all(line["base_config_hash"] == identity["base_config_hash"]
               and line["checkpoint_sha256"] == identity["weight"]["sha256"]
               and line["split"] == "eval" for line in runner_logs)
    status_counts: dict[str, int] = {}
    for row in trace:
        assert row["image_id"] in ids and row["model"] == model and row["dataset"] == dataset
        assert row["method"] == "occlusion" and row["config_hash"] == config_hash
        assert int(row["target"]) == targets[row["image_id"]]
        assert int(row["seed"]) == stable_seed(dataset, row["image_id"], int(row["repeat"]))
        assert float(row["sigma"]) == 0.005
        score = float(row["score"])
        assert math.isfinite(score) and -1 <= score <= 1
        assert math.isfinite(float(row["attribution_time_ms"]))
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        if row["status"] == "valid":
            assert row["spearman"] and close(float(row["spearman"]), score)
        else:
            assert score == 0 and not row["spearman"]
    assert set(status_counts) <= {"valid", "nonfinite", "both_constant", "reference_constant",
                                  "candidate_constant", "undefined_spearman"}, status_counts
    per_image = rows(result / "per_image.csv")
    assert len(per_image) == 1840
    assert {(r["image_id"], r["metric"]) for r in per_image} == {
        (image_id, metric) for image_id in ids for metric in BASE_METRICS | STABILITY_METRICS}
    values = {(r["image_id"], r["metric"]): float(r["value"]) for r in per_image}
    for image_id in ids:
        repeats = [r for r in trace if r["image_id"] == image_id]
        assert close(values[image_id, "stability_spearman"],
                     sum(float(r["score"]) for r in repeats) / 5)
        assert close(values[image_id, "stability_valid_rate"],
                     sum(r["status"] == "valid" for r in repeats) / 5)
        assert 0 <= values[image_id, "stability_valid_rate"] <= 1
    original_metrics = rows(backup / "per_image.csv")
    assert len(original_metrics) == 920
    assert {(r["image_id"], r["metric"]): (r["value"], r["time_ms"]) for r in original_metrics} == {
        (r["image_id"], r["metric"]): (r["value"], r["time_ms"])
        for r in per_image if r["metric"] in BASE_METRICS}
    summaries = rows(result / "units.csv")
    original_summaries = rows(backup / "units.csv")
    assert len(summaries) == 4 and len(original_summaries) == 2
    base_summaries = {r["metric"]: r for r in summaries if r["metric"] in BASE_METRICS}
    assert set(base_summaries) == BASE_METRICS
    for original in original_summaries:
        current = base_summaries[original["metric"]]
        assert current["config_hash"] == original["config_hash"]
        assert int(current["n"]) == int(original["n"]) == 460
        assert close(float(current["mean"]), float(original["mean"]))
        assert close(float(current["std"]), float(original["std"]))
    for metric in STABILITY_METRICS:
        row = next(r for r in summaries if r["metric"] == metric)
        metric_values = [values[image_id, metric] for image_id in ids]
        assert int(row["n"]) == 460 and row["config_hash"] == config_hash
        assert close(float(row["mean"]), statistics.mean(metric_values))
        assert close(float(row["std"]), statistics.stdev(metric_values))
    unchanged = 0
    maps = 0
    for entry in manifest["source_files"]:
        if not entry["source"].startswith(f"results/occlusion_eval/{unit}/"):
            continue
        source = ROOT / entry["source"]
        saved = ROOT / entry["backup"]
        assert sha256(saved) == entry["sha256"]
        if source.name in ("per_image.csv", "units.csv"):
            continue
        assert source.stat().st_size == entry["bytes"] and sha256(source) == entry["sha256"]
        unchanged += 1
        if source.suffix == ".npy":
            maps += 1
            sidecar = Path(str(source) + ".provenance.json")
            evidence = json.loads(sidecar.read_text(encoding="utf-8"))
            assert evidence["binding"] == "verified_recomputation"
            assert evidence["image_id"] == source.stem and evidence["split"] == "eval"
            assert evidence["model"] == model and evidence["dataset"] == dataset
            assert evidence["base_config_hash"] == identity["base_config_hash"]
            assert evidence["prediction_context_hash"] == identity["prediction_context_hash"]
            assert evidence["checkpoint_sha256"] == identity["weight"]["sha256"]
            assert evidence["map_sha256"] == entry["sha256"]
    assert maps == 460
    if dataset == "voc":
        assert sha256(ROOT / identity["weight"]["path"]) == identity["weight"]["sha256"]
    processes = [json.loads(line) for line in (output / "process_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(row["status"] for row in processes) <= {"success", "failed"}
    assert sum(row["status"] == "success" for row in processes) == len(runner_logs)
    wall = sum(float(row["wall_seconds_this_process"]) for row in processes)
    report = {
        "status": "passed", "unit": unit, "split": "eval", "images": 460,
        "trace_rows": 2300, "stability_metric_rows": 920,
        "config_hash": config_hash, "base_config_hash": identity["base_config_hash"],
        "checkpoint_sha256": identity["weight"]["sha256"],
        "status_counts": status_counts, "degenerate_repeats": 2300 - status_counts.get("valid", 0),
        "spearman_mean": statistics.mean(values[image_id, "stability_spearman"] for image_id in ids),
        "spearman_sample_std": statistics.stdev(values[image_id, "stability_spearman"] for image_id in ids),
        "valid_rate_mean": statistics.mean(values[image_id, "stability_valid_rate"] for image_id in ids),
        "attribution_time_ms_mean": statistics.mean(float(r["attribution_time_ms"]) for r in trace),
        "processed_sequence": [r["processed"] for r in runner_logs],
        "skipped_sequence": [r["skipped"] for r in runner_logs],
        "process_wall_seconds_total": wall,
        "failed_attempts": sum(row["status"] == "failed" for row in processes),
        "failed_attempts_detail": [{"error": row.get("error"), "wall_seconds": row["wall_seconds_this_process"]}
                                   for row in processes if row["status"] == "failed"],
        "cuda_peak_allocated_mib": max(float(r["cuda_peak_allocated_mib"]) for r in processes),
        "cuda_peak_reserved_mib": max(float(r["cuda_peak_reserved_mib"]) for r in processes),
        "base_files_unchanged": unchanged, "reference_maps_unchanged": maps,
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", choices=UNITS, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.unit), ensure_ascii=False))
