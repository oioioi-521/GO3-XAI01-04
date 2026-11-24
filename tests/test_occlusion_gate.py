"""Contracts for the isolated Occlusion GPU efficiency/raw-MoRF gates."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments import run_occlusion_gate as gate
from experiments import run_unit
from experiments.predictions import PREDICTION_COLUMNS


CONFIGS = sorted((Path(__file__).resolve().parents[1] / "configs").glob("occlusion_gate_*.yaml"))
assert len(CONFIGS) == 6


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda path: path.stem)
def test_six_configs_keep_raw_gpu_debug_contract(config_path):
    config = run_unit._load_config(config_path)
    model, dataset = gate.validate_gate_config(config)
    assert config_path.stem == f"occlusion_gate_{model}_{dataset}"
    assert config["metrics"]["deletion_fractions"] == gate.GRID_21
    assert config["output"]["float_maps_dir"].startswith("results/occlusion_gates/debug40/")


def test_single_gate_is_isolated_and_hashed_separately():
    config_path = CONFIGS[0]
    forty, original_path = gate.effective_gate_config(config_path, 40)
    single, generated_path = gate.effective_gate_config(config_path, 1)
    assert original_path == config_path
    assert single["runtime"]["max_images"] == 1
    assert forty["runtime"]["max_images"] == 40
    assert "/single/" in single["output"]["per_image_csv"]
    assert "/debug40/" in forty["output"]["per_image_csv"]
    assert "results/occlusion_gates/single/" in generated_path.as_posix()
    assert run_unit._config_hash(single) != run_unit._config_hash(forty)


@pytest.mark.parametrize("change", [
    lambda config: config["metrics"].update(names=["efficiency_time_ms", "faithfulness_morf_auc"]),
    lambda config: config["metrics"].update(deletion_fractions=[0.0, 0.5, 1.0]),
    lambda config: config["dataset"].update(split="eval"),
    lambda config: config["model"].update(output_activation="sigmoid"),
    lambda config: config["output"].update(per_image_csv="results/per_image.csv"),
    lambda config: config["runtime"].update(warmup_runs=0),
    lambda config: config["runtime"].update(seed=43),
])
def test_gate_rejects_protocol_or_output_drift(change):
    config = run_unit._load_config(CONFIGS[0])
    change(config)
    with pytest.raises(ValueError):
        gate.validate_gate_config(config)


def _csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_output_inspector_checks_raw_maps_and_snapshot(tmp_path, monkeypatch):
    config = run_unit._load_config(CONFIGS[0])
    model, dataset = gate.validate_gate_config(config)
    config["runtime"]["max_images"] = 1
    config["output"] = gate._outputs_for("single", model, dataset)
    monkeypatch.setattr(gate, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gate, "get_split", lambda *_: pd.DataFrame({"image_id": ["image-1"]}))
    output = config["output"]
    per_image = tmp_path / output["per_image_csv"]
    rows = [
        {"image_id": "image-1", "dataset": dataset, "model": model, "method": "occlusion", "metric": "efficiency_time_ms", "value": "12.5", "time_ms": "12.5"},
        {"image_id": "image-1", "dataset": dataset, "model": model, "method": "occlusion", "metric": "faithfulness_morf_auc_raw", "value": "0.4", "time_ms": ""},
    ]
    _csv(per_image, run_unit.PER_IMAGE_COLUMNS, rows)
    digest = run_unit._config_hash(config)
    _csv(tmp_path / output["units_csv"], run_unit.UNIT_COLUMNS, [
        {"method": "occlusion", "model": model, "dataset": dataset, "metric": metric, "mean": 0.4, "std": 0, "n": 1, "config_hash": digest}
        for metric in gate.METRICS
    ])
    _csv(tmp_path / output["predictions_csv"], PREDICTION_COLUMNS, [{
        **dict.fromkeys(PREDICTION_COLUMNS, ""), "image_id": "image-1", "dataset": dataset,
        "split": "debug", "model": model, "target_class_id": "0",
        "predicted_class_id": "0", "confidence": "0.8"
    }])
    config_path = tmp_path / "effective_config.yaml"
    snapshot = tmp_path / output["config_dir"] / f"{digest}_{config_path.name}"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(yaml.safe_dump(config), encoding="utf-8")
    map_path = tmp_path / output["float_maps_dir"] / f"occlusion_{model}_{dataset}" / "image-1.npy"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(map_path, np.zeros((224, 224), dtype=np.float32), allow_pickle=False)

    result = gate.inspect_gate_outputs(config, config_path, 1)
    assert result["images"] == 1
    assert result["constant_maps"] == ["image-1"]
    assert result["raw_morf_mean"] == pytest.approx(0.4)

    rows[1]["metric"] = "faithfulness_morf_auc"
    _csv(per_image, run_unit.PER_IMAGE_COLUMNS, rows)
    with pytest.raises(ValueError, match="missing, duplicate, or extra"):
        gate.inspect_gate_outputs(config, config_path, 1)
