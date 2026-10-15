"""Synthetic package integrity tests; never real LIME experiment results."""
import csv
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis.validate_lime_budget_gate import canonical_hash, validate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path, constant=False):
    project, gate = tmp_path / "project", tmp_path / "gate"
    data = project / "data"
    data.mkdir(parents=True)
    ids = [f"synthetic-{i}" for i in range(40)]
    with (data / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "image_id", "split", "class_id", "image_path"])
        writer.writerows(["voc", i, "debug", "18", "synthetic.bin"] for i in ids)
    (data / "synthetic.bin").write_bytes(b"synthetic source fixture; not a real image")
    meta_sha = hashlib.sha256((data / "metadata.csv").read_bytes()).hexdigest()
    write_json(data / "DATA_VERSION.json", {"sha256": {"metadata.csv": meta_sha}})
    config = dict(method="lime", dataset=dict(name="voc", split="debug"),
                  model=dict(name="resnet50", output_activation="sigmoid"),
                  runtime=dict(max_images=40), attribution=dict(output_activation="sigmoid"))
    identity = dict(config=config, budgets=[1000, 2000], seeds=[42, 43, 44],
                    metadata_sha256=meta_sha, image_ids=ids,
                    prediction_context=dict(dataset="voc", split="debug", model="resnet50",
                        output_activation="sigmoid", checkpoint_sha256="a" * 64),
                    versions={n: "synthetic-test-only" for n in
                        ("torch", "torchvision", "lime", "scikit-image", "scikit-learn", "scipy", "numpy")})
    digest = canonical_hash(identity)
    write_json(gate / "plan.json", dict(**identity, config_hash=digest,
        expected_images=40, attribution_calls=246, masked_forwards=369000))
    array = (np.zeros((224, 224), dtype=np.float32) if constant else
             np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(224, 224))
    template = tmp_path / "map.npy"
    np.save(template, array, allow_pickle=False)
    sha = hashlib.sha256(template.read_bytes()).hexdigest()
    variance = float(np.stack([array] * 3).var(0, ddof=1).mean())
    comparison = dict(status="constant_map" if constant else "valid",
                      spearman=None if constant else 1.0, rmse=0.0)
    for image_id in ids:
        stem = hashlib.sha256(image_id.encode()).hexdigest()[:16]
        runs = []
        for budget, seed in itertools.product(identity["budgets"], identity["seeds"]):
            path = gate / "maps_float" / f"{stem}_{budget}_{seed}.npy"
            path.parent.mkdir(exist_ok=True)
            # Regular copies also work in restricted Windows environments
            # where creating hard links requires additional permissions.
            shutil.copyfile(template, path)
            runs.append(dict(budget=budget, num_samples=budget, seed=seed,
                n_superpixels=2, weighted_r2=-0.5, time_ms=float(budget + seed),
                map_path=str(path.relative_to(gate)), map_sha256=sha))
        record = dict(image_id=image_id, target=18, config_hash=digest, runs=runs,
            budget_comparisons=[dict(budget=1000, seed=s, reference_budget=2000, **comparison)
                                for s in identity["seeds"]],
            seed_variance=[dict(budget=b, pixel_variance_mean=variance,
                seed_pairs=[dict(seed_a=a, seed_b=c, **comparison)
                            for a, c in itertools.combinations(identity["seeds"], 2)])
                           for b in identity["budgets"]])
        write_json(gate / "records" / (stem + ".json"), record)
    write_json(gate / "summary.json", dict(config_hash=digest, n_images=40,
        current_invocation_wall_seconds=1.0,
        similarity_by_budget={"1000": dict(reference_budget=2000,
            valid_pairs=0 if constant else 120, total_pairs=120,
            mean_spearman=None if constant else 1.0,
            p10_spearman=None if constant else 1.0, mean_rmse=0.0)},
        mean_time_ms={"1000": 1043., "2000": 2043.},
        mean_pixel_variance={"1000": variance, "2000": variance}))
    return gate, project


@pytest.mark.parametrize("constant", [False, True])
def test_valid_statistics_are_not_formal_approval(tmp_path, constant):
    gate, project = fixture(tmp_path, constant)
    report = validate(gate, project)
    assert report["statistical_integrity_passed"] is True
    assert report["formal_budget_approved"] is False
    assert report["generation_time_provenance_verified"] is False
    assert report["n_images"] == 40 and report["n_maps"] == 240
    assert report["budget_pairs"] == 120 and report["seed_pairs"] == 240
    assert report["constant_maps"] == (240 if constant else 0)
    assert report["source_image_hashes_verified"] == 0
    assert report["recorded_timed_attribution_seconds"] == pytest.approx(370.32)


@pytest.mark.parametrize("change", ["missing_runs", "duplicate_runs", "target", "comparison",
    "variance", "seed_pairs", "map_hash", "map_dtype", "escape_path", "absolute_path",
    "input_hash", "nan_time", "negative_time", "budget_diagnostic", "changed_map"])
def test_reject_corrupt_record(tmp_path, change):
    gate, project = fixture(tmp_path)
    path = next((gate / "records").glob("*.json"))
    record = json.loads(path.read_text())
    if change == "missing_runs":
        record["runs"] = []
    elif change == "duplicate_runs":
        record["runs"][1] = record["runs"][0]
    elif change == "target":
        record["target"] = 0
    elif change == "comparison":
        record["budget_comparisons"][0]["spearman"] = 0.5
    elif change == "variance":
        record["seed_variance"][0]["pixel_variance_mean"] = 1.0
    elif change == "seed_pairs":
        record["seed_variance"][0]["seed_pairs"].pop()
    elif change == "map_hash":
        record["runs"][0]["map_sha256"] = "b" * 64
    elif change == "map_dtype":
        map_path = gate / record["runs"][0]["map_path"]
        map_path.unlink()  # This is an isolated pytest fixture, not user data.
        np.save(map_path, np.zeros((224, 224), dtype=np.float64), allow_pickle=False)
        record["runs"][0]["map_sha256"] = hashlib.sha256(map_path.read_bytes()).hexdigest()
    elif change == "escape_path":
        record["runs"][0]["map_path"] = "maps_float/../../secret.npy"
    elif change == "absolute_path":
        record["runs"][0]["map_path"] = "C:/secret.npy"
    elif change == "input_hash":
        record["source_image_sha256"] = "0" * 64
    elif change == "nan_time":
        record["runs"][0]["time_ms"] = float("nan")
    elif change == "negative_time":
        record["runs"][0]["time_ms"] = -1
    elif change == "budget_diagnostic":
        record["runs"][0]["num_samples"] = 8
    elif change == "changed_map":
        map_path = gate / record["runs"][0]["map_path"]
        np.save(map_path, 1 - np.load(map_path, allow_pickle=False), allow_pickle=False)
        record["runs"][0]["map_sha256"] = hashlib.sha256(map_path.read_bytes()).hexdigest()
    write_json(path, record)
    with pytest.raises(ValueError):
        validate(gate, project)


@pytest.mark.parametrize("change", ["missing_record", "plan_hash", "metadata", "summary_mean",
                                    "summary_count", "summary_extra_budget", "json_duplicate_key"])
def test_reject_package_mismatch(tmp_path, change):
    gate, project = fixture(tmp_path)
    if change == "missing_record":
        next((gate / "records").glob("*.json")).unlink()
    elif change == "plan_hash":
        path = gate / "plan.json"
        plan = json.loads(path.read_text())
        plan["config_hash"] = "bad"
        write_json(path, plan)
    elif change == "metadata":
        (project / "data/metadata.csv").write_text("changed")
    elif change == "json_duplicate_key":
        (gate / "summary.json").write_text('{"n_images":40,"n_images":40}')
    else:
        path = gate / "summary.json"
        summary = json.loads(path.read_text())
        if change == "summary_mean":
            summary["mean_time_ms"]["1000"] = 1
        elif change == "summary_count":
            summary["n_images"] = 39
        else:
            summary["similarity_by_budget"]["8"] = {}
        write_json(path, summary)
    with pytest.raises(ValueError):
        validate(gate, project)


def test_verify_available_source_hash_without_claiming_generation_proof(tmp_path):
    gate, project = fixture(tmp_path)
    path = next((gate / "records").glob("*.json"))
    record = json.loads(path.read_text())
    record["source_image_sha256"] = hashlib.sha256((project / "data/synthetic.bin").read_bytes()).hexdigest()
    write_json(path, record)
    report = validate(gate, project)
    assert report["source_image_hashes_verified"] == 1
    assert report["generation_time_provenance_verified"] is False
