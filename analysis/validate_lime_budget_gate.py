"""Independently check E's 40-image LIME gate directory as data, never code.

Recompute comparisons/variance from float32 maps. This is a statistical integrity
check, NOT budget approval, formal eval, or proof of generation-time provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.stats import spearmanr


IDENTITY_KEYS = ("config", "budgets", "seeds", "metadata_sha256",
                 "prediction_context", "image_ids", "versions")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, root=None):
    if root is not None:
        require(path.resolve().is_relative_to(root), "JSON path escapes gate directory")
    require(path.is_file() and path.stat().st_size <= 2_000_000,
            f"missing or oversized JSON: {path}")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8-sig"),
                      object_pairs_hook=pairs, parse_constant=constant)


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid integer: {value!r}")
    return value


def number(value, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value),
            f"non-finite/non-numeric value: {value!r}")
    require(minimum is None or value >= minimum, f"value below range: {value}")
    require(maximum is None or value <= maximum, f"value above range: {value}")
    return value


def close(actual, expected, name):
    if expected is None:
        require(actual is None, f"{name}: expected null")
    else:
        number(actual)
        require(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8),
                f"{name}: recorded {actual!r}, recomputed {expected!r}")


def map_path(root, value):
    require(isinstance(value, str), "map path must be a string")
    path = PurePosixPath(value.replace("\\", "/"))
    require(not path.is_absolute() and ".." not in path.parts and ":" not in value
            and len(path.parts) >= 2 and path.parts[0] == "maps_float"
            and path.suffix == ".npy", "unsafe map path")
    resolved = (root / Path(*path.parts)).resolve()
    base = (root / "maps_float").resolve()
    require(base.is_relative_to(root) and resolved.is_relative_to(base), "map escapes maps_float")
    return resolved


def comparison(a, b):
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return dict(status="constant_map", spearman=None, rmse=rmse)
    value = float(spearmanr(a.ravel(), b.ravel()).statistic)
    return dict(status="valid" if math.isfinite(value) else "undefined",
                spearman=value if math.isfinite(value) else None, rmse=rmse)


def check_comparison(row, expected, label):
    require(row["status"] == expected["status"], f"{label}: comparison status")
    if row["spearman"] is not None:
        number(row["spearman"], -1, 1)
    number(row["rmse"], 0)
    close(row["spearman"], expected["spearman"], f"{label} Spearman")
    close(row["rmse"], expected["rmse"], f"{label} RMSE")


def index_rows(rows, fields, expected_keys, label):
    require(isinstance(rows, list), f"{label}: expected list")
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), f"{label}: expected record")
        key = tuple(integer(row[name]) for name in fields)
        require(key not in indexed, f"{label}: duplicate key {key}")
        indexed[key] = row
    require(set(indexed) == set(expected_keys), f"{label}: incomplete or extra keys")
    return indexed


def validate(gate: Path, project: Path) -> dict:
    gate, project = gate.resolve(), project.resolve()
    plan = read_json(gate / "plan.json", gate)
    require(all(key in plan for key in IDENTITY_KEYS), "plan identity fields missing")
    digest = canonical_hash({key: plan[key] for key in IDENTITY_KEYS})
    require(plan["config_hash"] == digest, "plan canonical config hash mismatch")
    config = plan["config"]
    dataset = config["dataset"]["name"]
    model = config["model"]["name"]
    require(config["method"] == "lime" and dataset in {"imagenet", "voc"}
            and model in {"resnet50", "densenet121", "vgg16"}
            and config["dataset"]["split"] == "debug"
            and config["runtime"]["max_images"] == 40, "not a full debug LIME gate")
    activation = "sigmoid" if dataset == "voc" else "softmax"
    require(config["model"]["output_activation"] == activation
            and config["attribution"]["output_activation"] == activation,
            "model/attribution output semantics mismatch")
    budgets, seeds = plan["budgets"], plan["seeds"]
    require(isinstance(budgets, list) and isinstance(seeds, list), "budgets/seeds must be lists")
    for value in budgets:
        integer(value, 2)
    for value in seeds:
        require(integer(value) < 2**32, "seed outside uint32")
    require(len(budgets) >= 2 and budgets == sorted(set(budgets))
            and len(seeds) >= 3 and seeds == sorted(set(seeds)), "budget/seed coverage")
    require(plan["expected_images"] == 40
            and plan["attribution_calls"] == 41 * len(budgets) * len(seeds)
            and plan["masked_forwards"] == 41 * len(seeds) * sum(budgets), "plan counts")
    version_names = {"torch", "torchvision", "lime", "scikit-image", "scikit-learn", "scipy", "numpy"}
    require(version_names <= set(plan["versions"])
            and all(isinstance(plan["versions"][name], str) and plan["versions"][name]
                    for name in version_names), "environment versions missing")
    context = plan["prediction_context"]
    require(isinstance(context, dict) and context.get("dataset") == dataset
            and context.get("split") == "debug" and context.get("model") == model
            and context.get("output_activation") == activation, "prediction context identity")
    weight_sha = context.get("checkpoint_sha256")
    require(isinstance(weight_sha, str) and len(weight_sha) == 64
            and all(c in "0123456789abcdef" for c in weight_sha), "checkpoint byte SHA missing")
    metadata = project / "data/metadata.csv"
    require(file_sha(metadata) == plan["metadata_sha256"]
            == read_json(project / "data/DATA_VERSION.json")["sha256"]["metadata.csv"],
            "frozen metadata byte SHA mismatch")
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["dataset"] == dataset and r["split"] == "debug"]
    ids = [r["image_id"] for r in rows]
    require(len(ids) == len(set(ids)) == 40
            and [str(i) for i in plan["image_ids"]] == ids, "frozen 40-image ID/order mismatch")
    paths = list((gate / "records").glob("*.json"))
    require(len(paths) == 40, "gate must contain exactly 40 records")
    expected_files = {hashlib.sha256(i.encode()).hexdigest()[:16] + ".json" for i in ids}
    require({p.name for p in paths} == expected_files, "record file IDs mismatch")
    runs_keys = set(itertools.product(budgets, seeds))
    comp_keys = set(itertools.product(budgets[:-1], seeds))
    seed_keys = set(itertools.combinations(seeds, 2))
    timings = {b: [] for b in budgets}
    variances = {b: [] for b in budgets}
    comparisons = {b: [] for b in budgets[:-1]}
    seed_similarities = {b: [] for b in budgets}
    used_paths, constant_count = set(), 0
    # Repeated byte-identical maps are common in synthetic fixtures and
    # degenerate real maps. Cache computations, never path/hash/record checks.
    comparison_cache, variance_cache = {}, {}
    source_hash_records = 0
    for row in rows:
        image_id = row["image_id"]
        record = read_json(gate / "records" / (hashlib.sha256(image_id.encode()).hexdigest()[:16] + ".json"), gate)
        target = float(row["class_id"])
        require(target.is_integer() and record["image_id"] == image_id
                and integer(record["target"]) == int(target)
                and record["config_hash"] == digest, f"{image_id}: target/identity mismatch")
        # These hashes are optional in E's current format. Missing hashes are
        # explicitly NOT treated as generation-time input provenance.
        if "source_image_sha256" in record:
            source = (project / "data" / row["image_path"].replace("\\", "/")).resolve()
            require(source.is_relative_to((project / "data").resolve()), "metadata image escapes data")
            require(source.is_file() and file_sha(source) == record["source_image_sha256"],
                    f"{image_id}: source image hash mismatch")
            source_hash_records += 1
        runs = index_rows(record["runs"], ("budget", "seed"), runs_keys, f"{image_id} runs")
        arrays, map_shas = {}, {}
        for (budget, seed), run in runs.items():
            require(integer(run["num_samples"], 2) == budget, "budget diagnostic mismatch")
            integer(run["n_superpixels"], 1)
            number(run["weighted_r2"])
            timings[budget].append(number(run["time_ms"], 0))
            path = map_path(gate, run["map_path"])
            require(path not in used_paths, "duplicate map path across settings/images")
            used_paths.add(path)
            require(path.is_file() and path.stat().st_size <= 2_000_000, "missing/oversized map")
            require(file_sha(path) == run["map_sha256"], "map byte SHA mismatch")
            array = np.load(path, allow_pickle=False)
            require(isinstance(array, np.ndarray) and array.shape == (224, 224)
                    and array.dtype == np.float32 and np.isfinite(array).all()
                    and array.min() >= 0 and array.max() <= 1, "map shape/dtype/range")
            arrays[budget, seed] = array
            map_shas[budget, seed] = run["map_sha256"]
            constant_count += int(np.ptp(array) <= 1e-12)

        def compare_settings(left, right):
            key = tuple(sorted((map_shas[left], map_shas[right])))
            if key not in comparison_cache:
                comparison_cache[key] = comparison(arrays[left], arrays[right])
            return comparison_cache[key]

        saved = index_rows(record["budget_comparisons"], ("budget", "seed"), comp_keys,
                           f"{image_id} comparisons")
        for (budget, seed), recorded in saved.items():
            require(integer(recorded["reference_budget"], 2) == budgets[-1], "reference budget")
            calculated = compare_settings((budget, seed), (budgets[-1], seed))
            check_comparison(recorded, calculated, f"{image_id}/{budget}/{seed}")
            comparisons[budget].append(calculated)
        saved_var = index_rows(record["seed_variance"], ("budget",), {(b,) for b in budgets},
                               f"{image_id} variance")
        for budget in budgets:
            recorded = saved_var[budget,]
            variance_key = tuple(map_shas[budget, s] for s in seeds)
            if variance_key not in variance_cache:
                variance_cache[variance_key] = float(np.stack([arrays[budget, s] for s in seeds]).var(0, ddof=1).mean())
            value = variance_cache[variance_key]
            close(recorded["pixel_variance_mean"], value, "pixel sample variance")
            variances[budget].append(value)
            pairs = index_rows(recorded["seed_pairs"], ("seed_a", "seed_b"), seed_keys,
                               f"{image_id}/{budget} seed pairs")
            for pair, recorded_pair in pairs.items():
                calculated = compare_settings((budget, pair[0]), (budget, pair[1]))
                check_comparison(recorded_pair, calculated, "seed pair")
                seed_similarities[budget].append(calculated)
    summary = read_json(gate / "summary.json", gate)
    require(summary["config_hash"] == digest and summary["n_images"] == 40, "summary identity/count")
    require(set(summary["similarity_by_budget"]) == {str(b) for b in budgets[:-1]}
            and set(summary["mean_time_ms"]) == set(summary["mean_pixel_variance"])
            == {str(b) for b in budgets}, "summary budget set")
    for budget, values in comparisons.items():
        valid = [v["spearman"] for v in values if v["status"] == "valid"]
        saved = summary["similarity_by_budget"][str(budget)]
        require(saved["reference_budget"] == budgets[-1]
                and saved["valid_pairs"] == len(valid)
                and saved["total_pairs"] == len(values), "summary comparison counts")
        close(saved["mean_spearman"], float(np.mean(valid)) if valid else None, "mean Spearman")
        close(saved["p10_spearman"], float(np.quantile(valid, .1)) if valid else None, "P10 Spearman")
        close(saved["mean_rmse"], float(np.mean([v["rmse"] for v in values])), "mean RMSE")
    for budget in budgets:
        close(summary["mean_time_ms"][str(budget)], float(np.mean(timings[budget])), "mean time")
        close(summary["mean_pixel_variance"][str(budget)], float(np.mean(variances[budget])), "mean variance")
    number(summary["current_invocation_wall_seconds"], 0)
    return dict(scope="LIME debug gate arrays/statistics only", statistical_integrity_passed=True,
                formal_budget_approved=False, generation_time_provenance_verified=False,
                dataset=dataset, model=model, config_hash=digest, n_images=40,
                n_maps=len(used_paths), constant_maps=constant_count,
                budget_pairs=sum(map(len, comparisons.values())),
                seed_pairs=sum(map(len, seed_similarities.values())),
                source_image_hashes_verified=source_hash_records,
                recorded_timed_attribution_seconds=sum(sum(v) for v in timings.values()) / 1000,
                current_invocation_wall_seconds=summary["current_invocation_wall_seconds"],
                warnings=["No automatic threshold or budget approval; inspect low-similarity images.",
                          "Source/tensor generation-time hashes, code, machine and full logs require separate audit.",
                          "Current-invocation wall time is not total resumed-run wall time."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", type=Path, help="extracted gate directory containing plan/records/maps/summary")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.gate, args.project)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"LIME gate rejected: {error}\n")
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.report:
        destination = args.report.resolve()
        gate = args.gate.resolve()
        require(not destination.is_relative_to(gate), "report must not modify the gate package")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
