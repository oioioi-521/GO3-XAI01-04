"""Independently verify the six frozen 460-image KernelSHAP eval units.

This reads outputs only. It never loads a model or modifies resumable state.
Run after a unit has completed, for example::

    python -m analysis.validate_kernelshap_formal --unit resnet50/imagenet

Omit --unit to require and validate all six units.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("resnet50", "densenet121", "vgg16")
DATASETS = ("imagenet", "voc")
BASE_METRICS = ("efficiency_time_ms", "faithfulness_morf_auc_raw")
STABILITY_METRICS = ("stability_spearman", "stability_valid_rate")
PROTOCOL_SHA256 = "324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf"
STATUSES = {
    "valid", "nonfinite", "both_constant", "reference_constant",
    "candidate_constant", "undefined_spearman",
}


class FormalValidationError(ValueError):
    """A formal unit failed its independent evidence check."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalValidationError(message)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_object(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _seed(dataset: str, image_id: str, repeat: int) -> int:
    payload = f"{dataset.lower()}\0{image_id}\0{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _rows(path: Path) -> list[dict[str, str]]:
    _require(path.is_file(), f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _yaml(path: Path) -> dict:
    _require(path.is_file(), f"missing YAML: {path}")
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    _require(isinstance(value, dict), f"invalid YAML mapping: {path}")
    return value


def _number(value: str, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise FormalValidationError(f"invalid number in {label}: {value!r}") from error
    _require(math.isfinite(number), f"non-finite number in {label}: {value!r}")
    return number


def _close(actual: float, expected: float, label: str) -> None:
    _require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
             f"{label}: {actual} != {expected}")


def _metadata(root: Path, dataset: str, expected_images: int) -> dict[str, int]:
    selected: dict[str, int] = {}
    for row in _rows(root / "data" / "metadata.csv"):
        if row["dataset"] != dataset or row["split"] != "eval":
            continue
        image_id = row["image_id"]
        target = _number(row["class_id"], f"metadata target {image_id}")
        _require(target.is_integer(), f"non-integral metadata target: {image_id}")
        _require(image_id not in selected, f"duplicate metadata image: {image_id}")
        selected[image_id] = int(target)
    _require(len(selected) == expected_images,
             f"{dataset} eval metadata: {len(selected)} != {expected_images}")
    return selected


def _prediction_context(root: Path, dataset: str, model: str, config: dict) -> dict:
    model_config = config["model"]
    weights = str(model_config.get("weights", "default")).lower()
    checkpoint = model_config.get("checkpoint")
    if checkpoint:
        checkpoint_path = root / str(checkpoint)
    else:
        _require(weights == "default", f"{model}/{dataset}: no checkpoint or default weights")
        checkpoints = list((root / ".cache" / "torch" / "hub" / "checkpoints").glob(f"{model}-*.pth"))
        _require(len(checkpoints) == 1, f"{model}: expected one cached default weight file")
        checkpoint_path = checkpoints[0]
    _require(checkpoint_path.is_file(), f"missing checkpoint: {checkpoint_path}")
    checkpoint_sha = _hash_file(checkpoint_path)
    task_type = str(model_config["task_type"]).lower()
    activation = str(model_config["output_activation"]).lower()
    identity = {
        "prediction_schema": 1,
        "dataset": dataset,
        "split": "eval",
        "metadata_sha256": _hash_file(root / "data" / "metadata.csv"),
        "model": {
            "name": model,
            "weights": weights,
            "num_classes": int(model_config["num_classes"]),
            "checkpoint_sha256": checkpoint_sha,
            "task_type": task_type,
            "output_activation": activation,
        },
        "preprocessing": {
            "resize": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    return {
        "dataset": dataset, "split": "eval", "model": model, "weights": weights,
        "checkpoint_sha256": checkpoint_sha, "task_type": task_type,
        "output_activation": activation, "config_hash": _hash_object(identity),
    }


def _safe_name(image_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in image_id)


def _check_maps(
    root: Path, config: dict, model: str, dataset: str, targets: dict[str, int],
    base_hash: str, context: dict,
) -> None:
    name = f"kernelshap_{model}_{dataset}"
    float_dir = root / config["output"]["float_maps_dir"] / name
    png_dir = root / config["output"]["maps_dir"] / name
    expected_stems = {_safe_name(image_id) for image_id in targets}
    _require({p.stem for p in float_dir.glob("*.npy")} == expected_stems,
             f"{name}: float32 map set differs from eval metadata")
    _require({p.stem for p in png_dir.glob("*.png")} == expected_stems,
             f"{name}: PNG map set differs from eval metadata")
    for image_id in targets:
        path = float_dir / f"{_safe_name(image_id)}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        _require(array.dtype == np.float32 and array.shape == (224, 224),
                 f"{path}: expected float32 224x224 map")
        _require(bool(np.isfinite(array).all()), f"{path}: non-finite map")
        sidecar_path = path.with_suffix(path.suffix + ".provenance.json")
        _require(sidecar_path.is_file(), f"missing map provenance: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": 1, "image_id": image_id, "dataset": dataset,
            "model": model, "method": "kernelshap", "split": "eval",
            "base_config_hash": base_hash,
            "prediction_context_hash": context["config_hash"],
            "checkpoint_sha256": context["checkpoint_sha256"],
            "map_sha256": _hash_file(path),
        }
        _require(all(sidecar.get(key) == value for key, value in expected.items()),
                 f"map provenance mismatch: {sidecar_path}")
        _require(sidecar.get("binding") in {"base_runner", "verified_recomputation"},
                 f"invalid map binding: {sidecar_path}")


def _check_trace(
    rows: list[dict[str, str]], model: str, dataset: str,
    targets: dict[str, int], stability_hash: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    _require(len(rows) == len(targets) * 5,
             f"{model}/{dataset}: {len(rows)} trace rows, expected {len(targets) * 5}")
    grouped: dict[str, dict[int, dict[str, str]]] = {}
    counts: dict[str, int] = {}
    for row in rows:
        image_id = row["image_id"]
        _require(image_id in targets, f"{model}/{dataset}: unexpected trace image {image_id}")
        _require((row["dataset"], row["model"], row["method"]) ==
                 (dataset, model, "kernelshap"), f"trace unit mismatch: {image_id}")
        repeat = int(row["repeat"])
        _require(repeat in range(5), f"invalid repeat: {image_id}/{repeat}")
        repeats = grouped.setdefault(image_id, {})
        _require(repeat not in repeats, f"duplicate trace key: {image_id}/{repeat}")
        repeats[repeat] = row
        _require(int(row["seed"]) == _seed(dataset, image_id, repeat),
                 f"paired seed mismatch: {image_id}/{repeat}")
        _require(int(row["target"]) == targets[image_id], f"target mismatch: {image_id}")
        _require(row["protocol_version"] == "rgb-gaussian-spearman-v1" and
                 row["config_hash"] == stability_hash,
                 f"protocol/config mismatch: {image_id}/{repeat}")
        _close(_number(row["sigma"], "sigma"), 0.005, "sigma")
        status = row["status"]
        _require(status in STATUSES, f"unknown trace status: {status}")
        counts[status] = counts.get(status, 0) + 1
        score = _number(row["score"], "score")
        _require(-1.0 <= score <= 1.0, f"out-of-range score: {image_id}/{repeat}")
        if status == "valid":
            _close(score, _number(row["spearman"], "spearman"), "valid score")
            jaccard = _number(row["top10_jaccard"], "top10_jaccard")
            _require(0.0 <= jaccard <= 1.0, f"invalid Jaccard: {image_id}/{repeat}")
        else:
            _require(row["spearman"] == "" and row["top10_jaccard"] == "" and score == 0.0,
                     f"unaccounted degenerate repeat: {image_id}/{repeat}")
        original = int(row["original_pred"])
        perturbed = int(row["perturbed_pred"])
        _require(row["prediction_preserved"].lower() == str(original == perturbed).lower(),
                 f"prediction flag mismatch: {image_id}/{repeat}")
        original_score = _number(row["original_target_score"], "original target score")
        perturbed_score = _number(row["perturbed_target_score"], "perturbed target score")
        _require(0 <= original_score <= 1 and 0 <= perturbed_score <= 1,
                 f"target score out of range: {image_id}/{repeat}")
        _close(_number(row["target_score_abs_delta"], "target score delta"),
               abs(perturbed_score - original_score), "target score delta")
        _require(_number(row["perturbation_mae_pixel"], "perturbation MAE") >= 0 and
                 _number(row["attribution_time_ms"], "attribution time") >= 0,
                 f"negative trace diagnostic: {image_id}/{repeat}")

    _require(set(grouped) == set(targets), f"{model}/{dataset}: incomplete trace image set")
    means: dict[str, float] = {}
    valid_rates: dict[str, float] = {}
    for image_id, repeats in grouped.items():
        _require(set(repeats) == set(range(5)), f"incomplete paired repeats: {image_id}")
        original_preds = {row["original_pred"] for row in repeats.values()}
        _require(len(original_preds) == 1, f"inconsistent original prediction: {image_id}")
        original_scores = [_number(row["original_target_score"], "original score")
                           for row in repeats.values()]
        _require(max(original_scores) - min(original_scores) <= 1e-9,
                 f"inconsistent original target score: {image_id}")
        means[image_id] = float(np.mean([_number(repeats[r]["score"], "score")
                                          for r in range(5)]))
        valid_rates[image_id] = sum(repeats[r]["status"] == "valid" for r in range(5)) / 5
    return means, valid_rates, counts


def _check_tables(
    per_image: list[dict[str, str]], units: list[dict[str, str]],
    model: str, dataset: str, targets: dict[str, int], base_hash: str,
    stability_hash: str, trace_means: dict[str, float], valid_rates: dict[str, float],
) -> dict[str, dict[str, float]]:
    selected = [row for row in per_image if (row["method"], row["model"], row["dataset"])
                == ("kernelshap", model, dataset)]
    by_metric: dict[str, dict[str, float]] = {}
    for row in selected:
        metric, image_id = row["metric"], row["image_id"]
        _require(metric in BASE_METRICS + STABILITY_METRICS and image_id in targets,
                 f"unexpected per-image row: {model}/{dataset}/{metric}/{image_id}")
        values = by_metric.setdefault(metric, {})
        _require(image_id not in values, f"duplicate per-image key: {image_id}/{metric}")
        values[image_id] = _number(row["value"], f"{metric} value")
        if metric == "efficiency_time_ms":
            elapsed = _number(row["time_ms"], "base time")
            _require(elapsed >= 0, f"negative base time: {image_id}")
            _close(values[image_id], elapsed, f"base time/value {image_id}")
        else:
            _require(row["time_ms"] == "", f"non-efficiency row has time: {image_id}")
    _require(set(by_metric) == set(BASE_METRICS + STABILITY_METRICS),
             f"{model}/{dataset}: missing metric")
    for metric, values in by_metric.items():
        _require(set(values) == set(targets),
                 f"{model}/{dataset}/{metric}: incomplete image set ({len(values)})")
    for image_id in targets:
        _close(by_metric["stability_spearman"][image_id], trace_means[image_id],
               f"stability mean {image_id}")
        _close(by_metric["stability_valid_rate"][image_id], valid_rates[image_id],
               f"valid rate {image_id}")

    selected_units = [row for row in units if
                      (row["method"], row["model"], row["dataset"]) ==
                      ("kernelshap", model, dataset)]
    _require(len(selected_units) == 4, f"{model}/{dataset}: expected four unit summaries")
    seen: set[str] = set()
    summaries: dict[str, dict[str, float]] = {}
    for row in selected_units:
        metric = row["metric"]
        _require(metric in by_metric and metric not in seen,
                 f"duplicate/unknown unit summary: {model}/{dataset}/{metric}")
        seen.add(metric)
        expected_hash = base_hash if metric in BASE_METRICS else stability_hash
        _require(row["config_hash"] == expected_hash,
                 f"unit config hash mismatch: {model}/{dataset}/{metric}")
        values = np.array(list(by_metric[metric].values()), dtype=float)
        _require(int(row["n"]) == len(targets), f"unit n mismatch: {metric}")
        mean = _number(row["mean"], f"{metric} unit mean")
        std = _number(row["std"], f"{metric} unit std")
        _close(mean, float(values.mean()), f"{metric} unit mean")
        _close(std, float(values.std(ddof=1)), f"{metric} unit sample std")
        summaries[metric] = {"mean": mean, "std": std, "n": len(targets)}
    return summaries


def validate_unit(root: Path, model: str, dataset: str) -> dict:
    _require(model in MODELS and dataset in DATASETS, f"unsupported unit: {model}/{dataset}")
    targets = _metadata(root, dataset, 460)
    config_name = f"kernelshap_eval_{model}_{dataset}.yaml"
    config = _yaml(root / "configs" / config_name)
    _require(config["method"] == "kernelshap" and
             config["dataset"] == {"name": dataset, "split": "eval"} and
             config["model"]["name"] == model,
             f"formal config identity mismatch: {config_name}")
    _require(config["runtime"].get("resume") is True and
             config["runtime"].get("max_images") is None,
             f"formal config resume/limit mismatch: {config_name}")
    _require(config["metrics"]["names"] == list(BASE_METRICS),
             f"formal base metrics mismatch: {config_name}")
    _require(config["output"]["per_image_csv"] == "results/kernelshap_eval/per_image.csv" and
             config["output"]["units_csv"] == "results/kernelshap_eval/units.csv" and
             config["output"]["float_maps_dir"] == "results/kernelshap_eval/maps_float",
             f"formal output not isolated: {config_name}")
    base_hash = _hash_object(config)
    protocol_path = root / "configs" / "stability_protocol_v1.yaml"
    protocol = _yaml(protocol_path)
    normalized_protocol = protocol_path.read_bytes().replace(b"\r\n", b"\n")
    _require(hashlib.sha256(normalized_protocol).hexdigest() == PROTOCOL_SHA256,
             "frozen stability protocol SHA-256 mismatch")
    _require(protocol["repeats"] == 5 and protocol["sigma"] == 0.005 and
             protocol["target"] == "metadata" and
             protocol["protocol_version"] == "rgb-gaussian-spearman-v1",
             "frozen stability protocol values mismatch")
    context = _prediction_context(root, dataset, model, config)
    execution = {
        "base_config": config, "base_config_hash": base_hash,
        "prediction_context": context, "stability": protocol, "max_images": None,
    }
    stability_hash = _hash_object(execution)
    base_snapshot = _yaml(root / config["output"]["config_dir"] /
                          f"{base_hash}_{config_name}")
    _require(base_snapshot == config, f"base config snapshot mismatch: {config_name}")
    stability_snapshot = _yaml(root / protocol["output"]["config_dir"] /
                               f"{stability_hash}_stability_{config_name}")
    _require(all(stability_snapshot.get(key) == value for key, value in execution.items()),
             f"stability snapshot mismatch: {config_name}")

    trace_root = root / protocol["output"]["trace_csv"]
    trace_path = trace_root.with_name(
        f"{trace_root.stem}_kernelshap_{model}_{dataset}_eval_{stability_hash}{trace_root.suffix}"
    )
    trace_means, valid_rates, statuses = _check_trace(
        _rows(trace_path), model, dataset, targets, stability_hash
    )
    per_image = _rows(root / config["output"]["per_image_csv"])
    units = _rows(root / config["output"]["units_csv"])
    summaries = _check_tables(per_image, units, model, dataset, targets,
                              base_hash, stability_hash, trace_means, valid_rates)
    _check_maps(root, config, model, dataset, targets, base_hash, context)

    log_path = root / protocol["output"]["run_log"]
    logs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    matching = [row for row in logs if
                (row.get("method"), row.get("model"), row.get("dataset"), row.get("split"),
                 row.get("config_hash")) ==
                ("kernelshap", model, dataset, "eval", stability_hash)]
    _require(matching and matching[-1].get("processed") == 0 and
             matching[-1].get("skipped") == 460,
             f"missing immediate 0/460 resume evidence: {model}/{dataset}")
    _require(matching[-1].get("base_config_hash") == base_hash and
             matching[-1].get("checkpoint_sha256") == context["checkpoint_sha256"],
             f"run log provenance mismatch: {model}/{dataset}")
    return {
        "model": model, "dataset": dataset, "images": len(targets),
        "trace_rows": len(targets) * 5, "statuses": statuses,
        "base_config_hash": base_hash, "stability_config_hash": stability_hash,
        "resume": {"processed": 0, "skipped": 460}, "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--unit", action="append", metavar="MODEL/DATASET",
                        help="validate a completed unit; may be repeated (default: all six)")
    args = parser.parse_args()
    root = args.root.resolve()
    requested = args.unit or [f"{model}/{dataset}" for dataset in DATASETS for model in MODELS]
    reports = []
    for name in requested:
        try:
            model, dataset = name.split("/", 1)
        except ValueError as error:
            raise FormalValidationError(f"unit must be MODEL/DATASET: {name}") from error
        reports.append(validate_unit(root, model, dataset))
    print(json.dumps({"protocol_sha256": PROTOCOL_SHA256, "units": reports},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
