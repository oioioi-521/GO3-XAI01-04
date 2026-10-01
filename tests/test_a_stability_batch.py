from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.map_provenance import sha256_file
from experiments.stability import stable_seed
from scripts.run_a_stability import base_rows, check_base_unchanged, prepare_gate, validate_stability


@pytest.fixture
def completed_unit(tmp_path):
    config = {"method": "ig", "dataset": {"name": "voc", "split": "eval"},
              "model": {"name": "resnet50"}, "metrics": {"names": ["efficiency_time_ms"]},
              "output": {"per_image_csv": str(tmp_path / "per_image.csv"),
                         "units_csv": str(tmp_path / "units.csv"),
                         "float_maps_dir": str(tmp_path / "maps_float")}}
    protocol = {"protocol_version": "rgb-gaussian-spearman-v1", "output": {
        "trace_csv": str(tmp_path / "trace.csv"), "run_log": str(tmp_path / "run_log.jsonl")}}
    unit = {"dataset": "voc", "model": "resnet50", "method": "ig"}
    expected = {"1": 0, "2": 1}
    rows = []
    trace = []
    for image_id, target in expected.items():
        score = 0.8 if image_id == "1" else 0.6
        rows.append({**unit, "image_id": image_id, "metric": "efficiency_time_ms", "value": 12.0})
        rows.extend({**unit, "image_id": image_id, "metric": metric, "value": value}
                    for metric, value in (("stability_spearman", score), ("stability_valid_rate", 1.0)))
        trace.extend({**unit, "image_id": image_id, "target": target, "repeat": repeat,
                      "seed": stable_seed("voc", image_id, repeat), "config_hash": "abc123",
                      "protocol_version": protocol["protocol_version"], "status": "valid",
                      "score": score, "spearman": score} for repeat in range(5))
    pd.DataFrame(rows).to_csv(config["output"]["per_image_csv"], index=False)
    pd.DataFrame([
        {**unit, "metric": "efficiency_time_ms", "mean": 12., "std": 0., "n": 2, "config_hash": "base"},
        {**unit, "metric": "stability_spearman", "mean": .7, "std": np.sqrt(.02), "n": 2, "config_hash": "abc123"},
        {**unit, "metric": "stability_valid_rate", "mean": 1., "std": 0., "n": 2, "config_hash": "abc123"},
    ]).to_csv(config["output"]["units_csv"], index=False)
    trace_path = tmp_path / "trace_ig_resnet50_voc_eval_abc123.csv"
    pd.DataFrame(trace).to_csv(trace_path, index=False)
    (tmp_path / "run_log.jsonl").write_text(json.dumps({**unit, "config_hash": "abc123", "processed": 0, "skipped": 2}) + "\n")
    return config, protocol, expected, trace_path


def test_complete_unit_and_base_preservation(completed_unit):
    config, protocol, expected, _ = completed_unit
    report = validate_stability(config, protocol, expected)
    assert report["images"] == 2 and report["trace_rows"] == 10
    from pathlib import Path
    path = Path(config["output"]["units_csv"])
    before = {path: base_rows(path)}
    frame = pd.read_csv(path)
    frame.loc[frame.metric.eq("stability_spearman"), "config_hash"] = "new-stability"
    frame.to_csv(path, index=False)
    check_base_unchanged(before)
    frame.loc[frame.metric.eq("efficiency_time_ms"), "config_hash"] = "wrong-base"
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        check_base_unchanged(before)


@pytest.mark.parametrize("mutation,error", [("missing", "trace repeats"), ("seed", "seed mismatch"), ("target", "target mismatch")])
def test_trace_corruption_is_rejected(completed_unit, mutation, error):
    config, protocol, expected, trace_path = completed_unit
    trace = pd.read_csv(trace_path)
    if mutation == "missing":
        trace = trace.iloc[:-1]
    else:
        trace.loc[0, mutation] += 1
    trace.to_csv(trace_path, index=False)
    with pytest.raises(ValueError, match=error):
        validate_stability(config, protocol, expected)


def test_duplicate_summary_metric_is_rejected(completed_unit):
    config, protocol, expected, _ = completed_unit
    units = pd.read_csv(config["output"]["units_csv"])
    units.loc[units.metric.eq("stability_valid_rate"), "metric"] = "stability_spearman"
    units.to_csv(config["output"]["units_csv"], index=False)
    with pytest.raises(ValueError, match="summary identity"):
        validate_stability(config, protocol, expected)


def test_gate_copies_reference_and_does_not_change_formal_output(completed_unit, tmp_path, monkeypatch):
    from pathlib import Path
    from scripts import run_a_stability
    config, protocol, _, _ = completed_unit
    monkeypatch.setattr(run_a_stability, "ROOT", tmp_path)
    name = "ig_resnet50_voc"
    original_map = Path(config["output"]["float_maps_dir"]) / name / "1.npy"
    original_map.parent.mkdir(parents=True)
    np.save(original_map, np.arange(16, dtype=np.float32).reshape(4, 4))
    source_csv = Path(config["output"]["per_image_csv"])
    csv_hash, map_hash = sha256_file(source_csv), sha256_file(original_map)
    original_config = deepcopy(config)
    gate_config_path, gate_protocol_path = prepare_gate(config, protocol, name, "1")
    gate_config = yaml.safe_load(gate_config_path.read_text())
    copied_map = Path(gate_config["output"]["float_maps_dir"]) / name / "1.npy"
    assert copied_map != original_map and sha256_file(copied_map) == map_hash
    assert sha256_file(source_csv) == csv_hash and config == original_config
    assert prepare_gate(config, protocol, name, "1") == (gate_config_path, gate_protocol_path)
    np.save(original_map, np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="gate inputs changed"):
        prepare_gate(config, protocol, name, "1")
