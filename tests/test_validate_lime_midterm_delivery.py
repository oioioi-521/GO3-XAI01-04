"""Reject malformed sample packages without executing delivered code."""

import csv
import hashlib
import io
import json
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest
from PIL import Image

from analysis.validate_lime_midterm_delivery import DATASETS, MAPS, MODELS, ROOT, validate


def _fixture(tmp_path, change=None):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/metadata.csv").write_text(
        "dataset,image_id,split,class_id\nvoc,8173,debug,18.0\n", encoding="utf-8")
    array = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(224, 224)
    map_buffer = io.BytesIO()
    np.save(map_buffer, array, allow_pickle=False)
    raw_map = map_buffer.getvalue()
    sha = hashlib.sha256(raw_map).hexdigest()
    manifest = {"split": "debug", "image_id": "8173", "model": "resnet50",
                "target": {"class_id": 18}, "checkpoint_sha256": "a" * 64,
                "reused_lime_map": {"sha256": sha, "provenance": {
                    "method": "lime", "model": "resnet50", "dataset": "voc", "split": "debug",
                    "image_id": "8173", "map_sha256": sha, "checkpoint_sha256": "a" * 64}}}
    units = [{"unit": f"{model}_{dataset}", "completed_images": 40, "debug_ids_unique": 40,
              "map_count": 240, "constant_map_count": 0, "valid_budget_pairs": 120,
              "total_budget_pairs": 120, "mean_spearman_1000_vs_2000": 0.95,
              "p10_spearman_1000_vs_2000": 0.8, "mean_rmse_1000_vs_2000": 0.1,
              "estimated_attribution_seconds": 360.0, "measured_gate_wall_seconds": None,
              "mean_attribution_ms": {"1000": 1000.0, "2000": 2000.0},
              "seed_pair_similarity": {budget: {"valid_pairs": 120, "total_pairs": 120,
                                                 "p10_spearman": 0.6}
                                       for budget in ("1000", "2000")}}
             for model in MODELS for dataset in DATASETS]
    completed = [row["unit"] for row in units]
    gates = {"completed_units": completed, "formal_evaluation_started": False,
             "formal_budget_frozen": False, "units": units}
    if change:
        change(manifest, gates)
    columns = ["unit", "completed_images", "map_count", "constant_map_count",
               "mean_spearman_1000_vs_2000", "p10_spearman_1000_vs_2000",
               "mean_rmse_1000_vs_2000", "estimated_attribution_seconds"]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(units)
    npz_buffer = io.BytesIO()
    np.savez(npz_buffer, **{method: array for method in MAPS})
    png_buffer = io.BytesIO()
    Image.new("RGB", (300, 200), "white").save(png_buffer, format="PNG")
    path = tmp_path / "sample.zip"
    with ZipFile(path, "w") as archive:
        for name, payload in {
            "manifest.json": json.dumps(manifest).encode(),
            "handoff_evidence/gate_summary.json": json.dumps(gates).encode(),
            "handoff_evidence/gate_summary.csv": csv_buffer.getvalue().encode(),
            "handoff_evidence/final_pipeline_status.json": json.dumps({
                "completed_units": completed, "formal_evaluation_started": False}).encode(),
            "maps_float32.npz": npz_buffer.getvalue(),
            "same_image_attribution_comparison.png": png_buffer.getvalue(),
            **{method + ".npy": raw_map for method in MAPS},
        }.items():
            archive.writestr(ROOT + name, payload)
    return path


def test_midterm_sample_and_summary_checks_do_not_certify_full_gates(tmp_path):
    path = _fixture(tmp_path)
    report, panel = validate(path, tmp_path)
    assert len(report["maps"]) == 5 and len(report["reported_gate_summaries"]) == 6
    assert report["formal_eval_started"] is False
    assert len(report["warnings"]) == 6
    assert report["sample"]["local_checkpoint_hash_matches"] is None
    assert panel.startswith(b"\x89PNG")
    assert "not independently accepted" in report["limitations"][0]


@pytest.mark.parametrize("change,message", [
    (lambda manifest, gates: manifest.update(split="eval"), "sample identity"),
    (lambda manifest, gates: manifest["reused_lime_map"].update(sha256="b" * 64), "LIME map hash"),
    (lambda manifest, gates: gates.update(formal_budget_frozen=True), "formal stage"),
    (lambda manifest, gates: gates["units"][0].update(completed_images=39), "gate counts"),
    (lambda manifest, gates: gates["units"][0].update(mean_spearman_1000_vs_2000=2), "score range"),
])
def test_rejects_inconsistent_claims(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        validate(_fixture(tmp_path, change), tmp_path)


@pytest.mark.parametrize("name", [ROOT + "../outside.txt", "foreign.txt", ROOT + "map\\bad.npy"])
def test_rejects_unsafe_member_names(tmp_path, name):
    path = _fixture(tmp_path)
    with ZipFile(path, "a") as archive:
        # ZipInfo normalizes backslashes on Windows; retain the deliberately
        # malformed name to exercise the reader, not the writer's cleanup.
        member = ZipInfo(name)
        member.filename = name
        archive.writestr(member, b"not executed")
    with pytest.raises(ValueError, match="unsafe ZIP"):
        validate(path, tmp_path)


def test_rejects_duplicate_members(tmp_path):
    path = _fixture(tmp_path)
    with pytest.warns(UserWarning, match="Duplicate"):
        with ZipFile(path, "a") as archive:
            archive.writestr(ROOT + "manifest.json", b"{}")
    with pytest.raises(ValueError, match="duplicate ZIP"):
        validate(path, tmp_path)


def test_unreconciled_timing_remains_visible(tmp_path):
    path = _fixture(tmp_path, lambda manifest, gates: gates["units"][0].update(
        estimated_attribution_seconds=1.0))
    report, _ = validate(path, tmp_path)
    assert any("source timing logs needed" in warning for warning in report["warnings"])
