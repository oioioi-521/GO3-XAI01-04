"""Strict read-side checks for new LIME debug gates (no model execution)."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import numpy as np

SCHEMA = 2
IDENTITY_KEYS = ("config", "budgets", "seeds", "metadata_sha256", "prediction_context",
                 "image_ids", "versions", "gate_schema_version", "image_assets", "execution_identity")


@contextmanager
def output_lock(output):
    """OS-managed nonblocking lock; crashes release it without deleting data."""
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".gate.lock"
    require(path.resolve().is_relative_to(root), "unsafe gate lock path")
    with path.open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("gate output is already locked by another run") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, root):
    path, root = Path(path), Path(root).resolve()
    require(path.resolve().is_relative_to(root) and path.is_file()
            and path.stat().st_size < 2_000_000, "missing/unsafe/oversized gate JSON")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("non-finite JSON: " + value)

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def tensor_sha(image):
    array = image.detach().cpu().contiguous().numpy()
    require(array.dtype == np.float32 and array.ndim == 3 and array.shape[0] == 3
            and np.isfinite(array).all(), "preprocessed image must be finite CHW float32 RGB")
    header = json.dumps(dict(dtype=str(array.dtype), shape=list(array.shape)), sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def source_path(project, image_path):
    relative = PurePosixPath(str(image_path).replace("\\", "/"))
    require(not relative.is_absolute() and ".." not in relative.parts
            and ":" not in str(relative), "unsafe source image path")
    root = (Path(project) / "data").resolve()
    result = (root / Path(*relative.parts)).resolve()
    require(result.is_relative_to(root), "source image escapes data")
    return result


def asset_identity(project, row, sample):
    path = source_path(project, row.image_path)
    require(sample["image_id"] == str(row.image_id) and int(sample["target"]) == int(row.class_id),
            "dataset target/image identity mismatch")
    return dict(image_id=str(row.image_id), target=int(row.class_id),
                source_image_sha256=file_sha(path), preprocessed_image_sha256=tensor_sha(sample["image"]),
                map_shape=list(sample["image"].shape[-2:]))


def check_sample(project, row, sample, expected):
    require(asset_identity(project, row, sample) == expected, "source/preprocessed image changed")


def number(value, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), "non-finite/non-numeric diagnostic")
    require((minimum is None or value >= minimum) and (maximum is None or value <= maximum),
            "diagnostic outside range")
    return value


def indexed(rows, fields, keys):
    require(isinstance(rows, list), "record field must be a list")
    result = {}
    for row in rows:
        require(isinstance(row, dict), "record entry must be an object")
        key = tuple(row[f] for f in fields)
        require(all(type(v) is int for v in key) and key not in result, "invalid/duplicate record key")
        result[key] = row
    require(set(result) == set(keys), "incomplete/extra budget or seed combinations")
    return result


def close(actual, expected):
    if expected is None:
        require(actual is None, "expected null correlation")
    else:
        number(actual)
        require(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-8), "saved statistics mismatch")


def comparison_check(saved, computed):
    require(saved["status"] == computed["status"], "comparison status mismatch")
    if saved["spearman"] is not None:
        number(saved["spearman"], -1, 1)
    number(saved["rmse"], 0)
    close(saved["spearman"], computed["spearman"])
    close(saved["rmse"], computed["rmse"])


def validate_record(record, report, asset, output, compare):
    """Recompute maps -> comparisons/sample variance before trusting a record."""
    require(record["gate_schema_version"] == SCHEMA and record["config_hash"] == report["config_hash"],
            "resume schema/plan mismatch; legacy gates need a new output directory")
    require(record["image_id"] == asset["image_id"] and type(record["target"]) is int
            and record["target"] == asset["target"], "resume target/image mismatch")
    for field in ("source_image_sha256", "preprocessed_image_sha256"):
        require(record[field] == asset[field], "resume input hash mismatch")
    budgets, seeds = report["budgets"], report["seeds"]
    runs = indexed(record["runs"], ("budget", "seed"), itertools.product(budgets, seeds))
    arrays = {}
    stem = hashlib.sha256(asset["image_id"].encode()).hexdigest()[:16]
    root = Path(output).resolve()
    for (budget, seed), run in runs.items():
        require(type(run["num_samples"]) is int and run["num_samples"] == budget
                and type(run["n_superpixels"]) is int and run["n_superpixels"] >= 1, "budget/superpixel diagnostic")
        number(run["weighted_r2"])  # Negative R2 is legal; NaN is not.
        number(run["time_ms"], 0)
        for field in ("coefficient_min", "coefficient_max"):
            if field in run:
                number(run[field])
        wanted = f"maps_float/{stem}_{budget}_{seed}.npy"
        require(isinstance(run["map_path"], str) and run["map_path"].replace("\\", "/") == wanted,
                "unsafe or non-canonical map path")
        path = root / wanted
        require(path.resolve().is_relative_to(root / "maps_float") and path.is_file()
                and path.stat().st_size <= 2_000_000 and file_sha(path) == run["map_sha256"],
                "gate resume map missing or changed")
        array = np.load(path, allow_pickle=False)
        require(isinstance(array, np.ndarray) and array.dtype == np.float32
                and list(array.shape) == asset["map_shape"] and np.isfinite(array).all()
                and array.min() >= 0 and array.max() <= 1, "map shape/dtype/range")
        arrays[budget, seed] = array
    comparisons = indexed(record["budget_comparisons"], ("budget", "seed"), itertools.product(budgets[:-1], seeds))
    for (budget, seed), saved in comparisons.items():
        require(type(saved["reference_budget"]) is int and saved["reference_budget"] == budgets[-1], "reference budget")
        comparison_check(saved, compare(arrays[budget, seed], arrays[budgets[-1], seed]))
    variance = indexed(record["seed_variance"], ("budget",), [(b,) for b in budgets])
    for budget in budgets:
        saved = variance[budget,]
        number(saved["pixel_variance_mean"], 0)
        close(saved["pixel_variance_mean"], float(np.stack([arrays[budget, s] for s in seeds]).var(0, ddof=1).mean()))
        pairs = indexed(saved["seed_pairs"], ("seed_a", "seed_b"), itertools.combinations(seeds, 2))
        for (a, b), pair in pairs.items():
            comparison_check(pair, compare(arrays[budget, a], arrays[budget, b]))
    return record


def existing_records(output, report, compare):
    """Validate EVERY existing record before model loading, warmup or writes."""
    require(report.get("gate_schema_version") == SCHEMA and len(report["image_assets"]) == 40,
            "schema-2 input identity missing")
    root = Path(output).resolve()
    expected = {hashlib.sha256(a["image_id"].encode()).hexdigest()[:16] + ".json": a for a in report["image_assets"]}
    paths = list((root / "records").glob("*.json"))
    require(len(expected) == 40 and {p.name for p in paths} <= set(expected), "unknown/duplicate image records")
    return {expected[p.name]["image_id"]: validate_record(read_json(p, root), report, expected[p.name], root, compare)
            for p in paths}
