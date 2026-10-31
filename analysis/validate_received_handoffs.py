"""Validate received Occlusion eval and stability handoff artifacts.

The validators are read-only. They do not execute handoff code and do not
modify the received packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from experiments.predictions import PREDICTION_COLUMNS

from .merge_eval_handoff import DATASETS, MODELS
from .schema import read_csv, validate_per_image, validate_units


UNITS = tuple(f"{model}_{dataset}" for model in MODELS for dataset in DATASETS)
EVAL_METRICS = {"efficiency_time_ms", "faithfulness_morf_auc_raw"}
STABILITY_METRICS = EVAL_METRICS | {
    "stability_spearman",
    "stability_valid_rate",
}
TRACE_COLUMNS = {
    "image_id",
    "dataset",
    "model",
    "method",
    "repeat",
    "seed",
    "protocol_version",
    "config_hash",
    "target",
    "sigma",
    "status",
    "spearman",
    "score",
    "top10_jaccard",
    "attribution_time_ms",
}


class ReceivedHandoffError(ValueError):
    """Raised when a received artifact cannot be accepted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceivedHandoffError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    normalized = str(value).replace("\\", "/")
    path = Path(normalized)
    _require(
        not path.is_absolute() and ".." not in path.parts,
        f"unsafe manifest path: {value}",
    )
    return path


def _resolve_unique(relative: str, roots: Iterable[Path]) -> Path:
    path = _relative_path(relative)
    matches: list[Path] = []
    for root in roots:
        resolved_root = root.resolve()
        candidate = (resolved_root / path).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as error:
            raise ReceivedHandoffError(f"manifest path escapes root: {relative}") from error
        if candidate.is_file():
            matches.append(candidate)
    _require(len(matches) == 1, f"manifest path must resolve once: {relative}")
    return matches[0]


def _close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12))


def validate_eval_assets(
    manifest_path: str | Path,
    roots: Iterable[str | Path],
    *,
    expected_maps: int = 2760,
) -> dict[str, Any]:
    """Validate the detailed eval manifest across light and float32 roots."""

    manifest_file = Path(manifest_path).resolve()
    resolved_roots = tuple(Path(root).resolve() for root in roots)
    _require(resolved_roots, "at least one eval asset root is required")
    _require(all(root.is_dir() for root in resolved_roots), "eval asset root missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1, "unsupported eval manifest")
    units = manifest.get("units")
    _require(isinstance(units, list), "eval manifest units must be a list")
    indexed = {str(unit.get("unit")): unit for unit in units}
    _require(
        len(indexed) == len(units) and set(indexed) == set(UNITS),
        "eval manifest must contain exactly six units",
    )

    entries: list[dict[str, Any]] = []
    for unit_name in UNITS:
        unit = indexed[unit_name]
        _require(int(unit.get("processed", -1)) == 460, f"bad eval count: {unit_name}")
        _require(int(unit.get("skipped", -1)) == 0, f"skipped eval rows: {unit_name}")
        _require(int(unit.get("failed", -1)) == 0, f"failed eval rows: {unit_name}")
        _require(int(unit.get("constant_maps", -1)) == 0, f"constant maps: {unit_name}")
        files = unit.get("files")
        _require(isinstance(files, list), f"missing file list: {unit_name}")
        entries.extend(files)

    paths = [str(entry.get("path")) for entry in entries]
    _require(len(paths) == len(set(paths)), "duplicate eval manifest paths")
    map_entries = [entry for entry in entries if entry.get("kind") == "float32_map"]
    _require(len(map_entries) == expected_maps, "unexpected float32 map count")
    _require(
        sum(entry.get("kind") == "float32_map" for entry in entries) == expected_maps,
        "float32 manifest kind mismatch",
    )

    map_min = math.inf
    map_max = -math.inf
    for entry in entries:
        relative = str(entry["path"])
        path = _resolve_unique(relative, resolved_roots)
        _require(path.stat().st_size == int(entry["bytes"]), f"size mismatch: {relative}")
        _require(
            sha256_file(path) == str(entry["sha256"]),
            f"sha256 mismatch: {relative}",
        )
        if entry.get("kind") != "float32_map":
            continue
        array = np.load(path, allow_pickle=False)
        _require(array.dtype == np.float32, f"map dtype mismatch: {relative}")
        _require(array.shape == (224, 224), f"map shape mismatch: {relative}")
        _require(np.isfinite(array).all(), f"non-finite map: {relative}")
        current_min = float(array.min())
        current_max = float(array.max())
        _require(0.0 <= current_min <= current_max <= 1.0, f"map range mismatch: {relative}")
        _require(current_max - current_min > 1e-12, f"constant map: {relative}")
        map_min = min(map_min, current_min)
        map_max = max(map_max, current_max)

    actual_maps: set[str] = set()
    for root in resolved_roots:
        actual_maps.update(path.relative_to(root).as_posix() for path in root.rglob("*.npy"))
    expected_map_paths = {str(entry["path"]).replace("\\", "/") for entry in map_entries}
    _require(actual_maps == expected_map_paths, "received float32 map set mismatch")
    return {
        "units": len(units),
        "files": len(entries),
        "float32_maps": len(map_entries),
        "map_shape": [224, 224],
        "map_dtype": "float32",
        "map_min": map_min,
        "map_max": map_max,
    }


def validate_package_manifest(root: str | Path) -> dict[str, Any]:
    """Validate every file listed by a lightweight package manifest."""

    package_root = Path(root).resolve()
    manifest_path = package_root / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("maps_included") is False, "unexpected maps_included value")
    entries = manifest.get("files")
    _require(isinstance(entries, list), "package manifest files must be a list")
    listed: set[str] = set()
    for entry in entries:
        relative = str(entry["path"]).replace("\\", "/")
        _require(relative not in listed, f"duplicate package path: {relative}")
        listed.add(relative)
        path = (package_root / _relative_path(relative)).resolve()
        try:
            path.relative_to(package_root)
        except ValueError as error:
            raise ReceivedHandoffError(f"package path escapes root: {relative}") from error
        _require(path.is_file(), f"missing package file: {relative}")
        _require(path.stat().st_size == int(entry["bytes"]), f"size mismatch: {relative}")
        _require(
            sha256_file(path) == str(entry["sha256"]),
            f"sha256 mismatch: {relative}",
        )
    actual = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    _require(actual == listed, "package file set differs from manifest")
    return {"files": len(entries), "maps_included": False}


def _stable_seed(dataset: str, image_id: str, repeat: int) -> int:
    payload = f"{dataset.lower()}\0{image_id}\0{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _validate_gate(gate: Path, stage: str, unit_name: str) -> dict[str, Any]:
    model, dataset = unit_name.rsplit("_", 1)
    expected_images = 1 if stage == "single" else 40
    report = json.loads((gate / "validation_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((gate / "input_manifest.json").read_text(encoding="utf-8"))
    protocol = yaml.safe_load((gate / "protocol.yaml").read_text(encoding="utf-8"))
    _require(report.get("status") == "passed", f"gate not passed: {stage}/{unit_name}")
    _require(report.get("stage") == stage and report.get("unit") == unit_name, f"gate identity mismatch: {stage}/{unit_name}")
    _require(manifest.get("stage") == stage and manifest.get("unit") == unit_name, f"input identity mismatch: {stage}/{unit_name}")
    _require(manifest.get("split") == "debug", f"non-debug gate: {stage}/{unit_name}")
    _require(protocol.get("protocol_version") == "rgb-gaussian-spearman-v1", f"protocol mismatch: {stage}/{unit_name}")
    _require(float(protocol.get("sigma")) == 0.005, f"sigma mismatch: {stage}/{unit_name}")
    _require(int(protocol.get("repeats")) == 5, f"repeat mismatch: {stage}/{unit_name}")
    _require(protocol.get("seed_fields") == ["dataset", "image_id", "repeat"], f"seed fields mismatch: {stage}/{unit_name}")
    _require(float(protocol.get("degenerate_score")) == 0.0, f"degenerate score mismatch: {stage}/{unit_name}")

    image_ids = tuple(str(value) for value in manifest.get("selected_image_ids", []))
    expected = set(image_ids)
    _require(len(image_ids) == len(expected) == expected_images, f"image set mismatch: {stage}/{unit_name}")
    predictions = read_csv(gate / "predictions.csv", "predictions")
    _require(list(predictions.columns) == PREDICTION_COLUMNS, f"prediction schema mismatch: {stage}/{unit_name}")
    _require(len(predictions) == expected_images, f"prediction count mismatch: {stage}/{unit_name}")
    _require(set(predictions["image_id"].astype(str)) == expected, f"prediction IDs mismatch: {stage}/{unit_name}")
    _require(set(predictions["dataset"].astype(str)) == {dataset} and set(predictions["model"].astype(str)) == {model}, f"prediction identity mismatch: {stage}/{unit_name}")
    targets = dict(zip(predictions["image_id"].astype(str), pd.to_numeric(predictions["target_class_id"], errors="raise").astype(int)))

    per_image = read_csv(gate / "per_image.csv", "per_image")
    validate_per_image(per_image)
    _require(len(per_image) == expected_images * 4, f"metric count mismatch: {stage}/{unit_name}")
    _require(set(per_image["metric"].astype(str)) == STABILITY_METRICS, f"metric set mismatch: {stage}/{unit_name}")
    _require(set(per_image["image_id"].astype(str)) == expected, f"metric IDs mismatch: {stage}/{unit_name}")
    _require(set(per_image["dataset"].astype(str)) == {dataset} and set(per_image["model"].astype(str)) == {model} and set(per_image["method"].astype(str)) == {"occlusion"}, f"metric identity mismatch: {stage}/{unit_name}")

    units = read_csv(gate / "units.csv", "units")
    validate_units(units)
    _require(len(units) == 4 and set(units["metric"].astype(str)) == STABILITY_METRICS, f"unit summary mismatch: {stage}/{unit_name}")
    trace_files = list(gate.glob("stability_trace_*.csv"))
    _require(len(trace_files) == 1, f"trace file count mismatch: {stage}/{unit_name}")
    traces = pd.read_csv(trace_files[0])
    _require(TRACE_COLUMNS <= set(traces.columns), f"trace schema mismatch: {stage}/{unit_name}")
    _require(len(traces) == expected_images * 5, f"trace count mismatch: {stage}/{unit_name}")
    _require(not traces.duplicated(["image_id", "repeat"]).any(), f"duplicate trace: {stage}/{unit_name}")
    _require(set(traces["image_id"].astype(str)) == expected, f"trace IDs mismatch: {stage}/{unit_name}")
    _require(set(pd.to_numeric(traces["repeat"], errors="raise").astype(int)) == set(range(5)), f"trace repeat mismatch: {stage}/{unit_name}")
    _require(set(traces["status"].astype(str)) == {"valid"}, f"non-valid gate trace: {stage}/{unit_name}")
    _require(set(traces["protocol_version"].astype(str)) == {"rgb-gaussian-spearman-v1"}, f"trace protocol mismatch: {stage}/{unit_name}")
    _require(set(pd.to_numeric(traces["sigma"], errors="raise")) == {0.005}, f"trace sigma mismatch: {stage}/{unit_name}")
    config_hash = str(report["config_hash"])
    _require(set(traces["config_hash"].astype(str)) == {config_hash}, f"trace config mismatch: {stage}/{unit_name}")
    _require(set(traces["dataset"].astype(str)) == {dataset} and set(traces["model"].astype(str)) == {model} and set(traces["method"].astype(str)) == {"occlusion"}, f"trace identity mismatch: {stage}/{unit_name}")

    traces["repeat"] = pd.to_numeric(traces["repeat"], errors="raise").astype(int)
    traces["seed"] = pd.to_numeric(traces["seed"], errors="raise").astype("int64")
    traces["target"] = pd.to_numeric(traces["target"], errors="raise").astype(int)
    traces["score"] = pd.to_numeric(traces["score"], errors="raise")
    traces["spearman"] = pd.to_numeric(traces["spearman"], errors="raise")
    traces["attribution_time_ms"] = pd.to_numeric(traces["attribution_time_ms"], errors="raise")
    _require(np.isfinite(traces[["score", "spearman", "attribution_time_ms"]]).all().all(), f"non-finite trace: {stage}/{unit_name}")
    _require(traces["score"].between(-1.0, 1.0).all(), f"score range mismatch: {stage}/{unit_name}")
    _require(np.allclose(traces["score"], traces["spearman"], rtol=0.0, atol=1e-12), f"score/spearman mismatch: {stage}/{unit_name}")
    for row in traces.itertuples(index=False):
        _require(int(row.seed) == _stable_seed(dataset, str(row.image_id), int(row.repeat)), f"seed mismatch: {stage}/{unit_name}/{row.image_id}")
        _require(int(row.target) == targets[str(row.image_id)], f"target mismatch: {stage}/{unit_name}/{row.image_id}")

    metric_values = per_image.assign(value=pd.to_numeric(per_image["value"], errors="raise"))
    indexed_metrics = metric_values.set_index(["image_id", "metric"])["value"]
    for image_id, group in traces.groupby("image_id"):
        _require(_close(float(indexed_metrics.loc[(image_id, "stability_spearman")]), float(group["score"].mean())), f"per-image stability mismatch: {stage}/{unit_name}/{image_id}")
        _require(_close(float(indexed_metrics.loc[(image_id, "stability_valid_rate")]), 1.0), f"valid-rate mismatch: {stage}/{unit_name}/{image_id}")
    recomputed = metric_values.groupby("metric")["value"].agg(["mean", "std", "count"])
    recomputed["std"] = recomputed["std"].fillna(0.0)
    for row in units.itertuples(index=False):
        current = recomputed.loc[str(row.metric)]
        _require(int(row.n) == expected_images == int(current["count"]), f"summary n mismatch: {stage}/{unit_name}/{row.metric}")
        _require(_close(float(row.mean), float(current["mean"])), f"summary mean mismatch: {stage}/{unit_name}/{row.metric}")
        _require(_close(float(row.std), float(current["std"])), f"summary std mismatch: {stage}/{unit_name}/{row.metric}")
        if str(row.metric).startswith("stability_"):
            _require(str(row.config_hash) == config_hash, f"summary config mismatch: {stage}/{unit_name}/{row.metric}")

    run_lines = _read_jsonl(gate / "stability_run_log.jsonl")
    _require(len(run_lines) >= 2, f"missing resume evidence: {stage}/{unit_name}")
    _require(int(run_lines[0]["processed"]) == expected_images and int(run_lines[0]["skipped"]) == 0, f"initial run mismatch: {stage}/{unit_name}")
    _require(int(run_lines[-1]["processed"]) == 0 and int(run_lines[-1]["skipped"]) == expected_images, f"resume mismatch: {stage}/{unit_name}")
    _require(all(str(line["config_hash"]) == config_hash for line in run_lines), f"run hash mismatch: {stage}/{unit_name}")
    base_hash = str(run_lines[0]["base_config_hash"])
    checkpoint_hash = str(run_lines[0]["checkpoint_sha256"])

    npy_entries = {
        Path(str(entry["copy"])).name: entry
        for entry in manifest.get("files", [])
        if str(entry.get("copy", "")).endswith(".npy")
    }
    sidecars = list(gate.rglob("*.npy.provenance.json"))
    _require(len(npy_entries) == len(sidecars) == expected_images, f"provenance count mismatch: {stage}/{unit_name}")
    for sidecar in sidecars:
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        map_name = sidecar.name.removesuffix(".provenance.json")
        _require(map_name in npy_entries, f"unlisted provenance: {stage}/{unit_name}/{map_name}")
        _require(provenance.get("binding") == "verified_recomputation", f"binding mismatch: {stage}/{unit_name}/{map_name}")
        _require(provenance.get("map_sha256") == npy_entries[map_name]["sha256"], f"map hash binding mismatch: {stage}/{unit_name}/{map_name}")
        _require(provenance.get("base_config_hash") == base_hash, f"base hash binding mismatch: {stage}/{unit_name}/{map_name}")
        _require(provenance.get("checkpoint_sha256") == checkpoint_hash, f"checkpoint binding mismatch: {stage}/{unit_name}/{map_name}")
        _require(provenance.get("split") == "debug" and provenance.get("image_id") in expected, f"provenance identity mismatch: {stage}/{unit_name}/{map_name}")

    _require(int(report["images"]) == expected_images, f"report image count mismatch: {stage}/{unit_name}")
    _require(int(report["trace_rows"]) == expected_images * 5, f"report trace count mismatch: {stage}/{unit_name}")
    _require(int(report["stability_metric_rows"]) == expected_images * 2, f"report metric count mismatch: {stage}/{unit_name}")
    _require(report.get("status_counts") == {"valid": expected_images * 5}, f"report statuses mismatch: {stage}/{unit_name}")
    _require(int(report.get("failed_runs", -1)) == 0, f"failed process: {stage}/{unit_name}")
    return {
        "stage": stage,
        "unit": unit_name,
        "dataset": dataset,
        "model": model,
        "images": expected_images,
        "trace_rows": len(traces),
        "stability_spearman_mean": float(recomputed.loc["stability_spearman", "mean"]),
        "stability_valid_rate_mean": float(recomputed.loc["stability_valid_rate", "mean"]),
        "attribution_time_ms_mean": float(traces["attribution_time_ms"].mean()),
        "config_hash": config_hash,
        "traces": traces[["image_id", "repeat", "seed"]].copy(),
    }


def validate_stability_light(root: str | Path) -> pd.DataFrame:
    """Validate all single/debug40 units in a received light gate package."""

    package_root = Path(root).resolve()
    validate_package_manifest(package_root)
    summaries: list[dict[str, Any]] = []
    paired_seeds: dict[tuple[str, str, str, int], list[tuple[str, int]]] = {}
    for stage in ("single", "debug40"):
        stage_root = package_root / stage
        _require(stage_root.is_dir(), f"missing stability stage: {stage}")
        actual_units = {path.name for path in stage_root.iterdir() if path.is_dir()}
        _require(actual_units == set(UNITS), f"stability unit set mismatch: {stage}")
        for unit_name in UNITS:
            result = _validate_gate(stage_root / unit_name, stage, unit_name)
            traces = result.pop("traces")
            summaries.append(result)
            for row in traces.itertuples(index=False):
                key = (stage, result["dataset"], str(row.image_id), int(row.repeat))
                paired_seeds.setdefault(key, []).append((result["model"], int(row.seed)))
    for key, values in paired_seeds.items():
        _require({model for model, _ in values} == set(MODELS), f"paired models missing: {key}")
        _require(len({seed for _, seed in values}) == 1, f"cross-model seed mismatch: {key}")
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-manifest", type=Path)
    parser.add_argument("--eval-root", type=Path, action="append", default=[])
    parser.add_argument("--stability-root", type=Path)
    args = parser.parse_args()
    _require(args.eval_manifest or args.stability_root, "select at least one validation")
    output: dict[str, Any] = {}
    if args.eval_manifest:
        _require(args.eval_root, "--eval-root is required with --eval-manifest")
        output["eval"] = validate_eval_assets(args.eval_manifest, args.eval_root)
    if args.stability_root:
        summary = validate_stability_light(args.stability_root)
        output["stability"] = summary.to_dict(orient="records")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
