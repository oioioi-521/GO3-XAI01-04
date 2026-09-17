import csv
import json
from pathlib import Path
import warnings

import pytest
import torch
import yaml

from experiments import run_unit


class TinyClassifier(torch.nn.Module):
    def forward(self, inputs):
        score = inputs[:, 0].sum(dim=(1, 2)) - inputs[:, 1].sum(dim=(1, 2))
        return torch.stack((score, -score), dim=1)


class SyntheticDataset:
    def __init__(self, dataset, split="debug", limit=None):
        assert dataset == "synthetic"
        assert split == "debug"
        image = torch.zeros(3, 4, 4)
        image[0, :, :2] = 1.0
        image[1, :, 2:] = 1.0
        self.samples = [{
            "image": image,
            "image_id": "synthetic-0001",
            "target": 0,
            "class_name": "positive",
            "image_path": "synthetic.png",
        }]
        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


def _config(tmp_path: Path):
    output = tmp_path / "results"
    return {
        "method": "occlusion",
        "dataset": {"name": "synthetic", "split": "debug"},
        "model": {"name": "tiny", "weights": "none", "num_classes": 2, "checkpoint": None},
        "attribution": {
            "window_height": 2,
            "window_width": 2,
            "stride_height": 2,
            "stride_width": 2,
            "perturbations_per_eval": 2,
        },
        "metrics": {
            "names": ["efficiency_time_ms", "faithfulness_morf_auc"],
            "deletion_fractions": [0.0, 0.5, 1.0],
        },
        "runtime": {"device": "cpu", "seed": 7, "max_images": 1, "resume": True},
        "output": {
            "per_image_csv": str(output / "per_image.csv"),
            "units_csv": str(output / "units.csv"),
            "config_dir": str(output / "configs"),
            "run_log": str(output / "run_log.jsonl"),
            "save_maps": False,
        },
    }


def test_runner_executes_occlusion_from_yaml_and_resumes(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config_path = tmp_path / "occlusion.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    model = TinyClassifier().eval()
    seen = {}
    original_occlusion = run_unit.Occlusion

    def fake_load_model(**kwargs):
        seen["model"] = kwargs
        return model

    def recording_occlusion(*args, **kwargs):
        seen["attribution"] = kwargs
        return original_occlusion(*args, **kwargs)

    monkeypatch.setattr(run_unit, "load_model", fake_load_model)
    monkeypatch.setattr(run_unit, "MetadataDataset", SyntheticDataset)
    monkeypatch.setattr(run_unit, "Occlusion", recording_occlusion)
    monkeypatch.setitem(run_unit.SUPPORTED_METHODS, "occlusion", recording_occlusion)

    run_unit.run(config_path)
    per_image_path = Path(config["output"]["per_image_csv"])
    with per_image_path.open(newline="", encoding="utf-8") as handle:
        initial_rows = list(csv.DictReader(handle))
    # Simulate interruption after one metric row was flushed.  Resume must
    # discard that incomplete image before recomputing it, not append a
    # duplicate metric value.
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_unit.PER_IMAGE_COLUMNS)
        writer.writeheader()
        writer.writerow(initial_rows[0])

    run_unit.run(config_path)
    run_unit.run(config_path)

    changed = _config(tmp_path)
    changed["attribution"]["stride_width"] = 1
    config_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        run_unit.run(config_path)

    assert seen["model"] == {
        "name": "tiny",
        "device": torch.device("cpu"),
        "weights": "none",
        "num_classes": 2,
        "checkpoint": None,
    }
    assert seen["attribution"]["model"] is model
    assert {key: value for key, value in seen["attribution"].items() if key != "model"} == changed["attribution"]

    with per_image_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert list(rows[0]) == run_unit.PER_IMAGE_COLUMNS
    assert {row["method"] for row in rows} == {"occlusion"}
    assert {row["metric"] for row in rows} == {"efficiency_time_ms", "faithfulness_morf_auc"}
    assert all(row["image_id"] == "synthetic-0001" for row in rows)

    with Path(config["output"]["units_csv"]).open(newline="", encoding="utf-8") as handle:
        unit_rows = list(csv.DictReader(handle))
    assert len(unit_rows) == 2
    assert {row["method"] for row in unit_rows} == {"occlusion"}
    assert {row["config_hash"] for row in unit_rows} == {run_unit._config_hash(changed)}

    logs = [json.loads(line) for line in Path(config["output"]["run_log"]).read_text(encoding="utf-8").splitlines()]
    assert [(entry["method"], entry["processed"], entry["skipped"]) for entry in logs] == [
        ("occlusion", 1, 0),
        ("occlusion", 1, 0),
        ("occlusion", 0, 1),
        ("occlusion", 1, 0),
    ]
    assert len(list(Path(config["output"]["config_dir"]).glob("*.yaml"))) == 2
    state = json.loads((Path(config["output"]["per_image_csv"]).parent / "run_state" / "occlusion_tiny_synthetic.json").read_text(encoding="utf-8"))
    assert state == {"dataset": "synthetic", "model": "tiny", "method": "occlusion", "config_hash": run_unit._config_hash(changed)}


def test_runner_accepts_existing_rise_and_rejects_unknown_method(tmp_path):
    rise_config = Path("configs/rise_smoke_imagenet.yaml")
    assert run_unit._load_config(rise_config)["method"] == "rise"
    assert run_unit._load_config(Path("configs/occlusion_smoke_imagenet.yaml"))["method"] == "occlusion"

    config = _config(tmp_path)
    config["method"] = "not-a-method"
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported method"):
        run_unit._load_config(config_path)
