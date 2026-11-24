"""KernelSHAP integration checks without project images or model weights."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from experiments import run_unit
from experiments.attribution import kernelshap as kernelshap_module


class TinyClassifier(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        score = inputs[:, 0].sum(dim=(1, 2)) - inputs[:, 1].sum(dim=(1, 2))
        return torch.stack((score, -score), dim=1)


class SyntheticDataset:
    def __init__(self, dataset: str, split: str = "debug", limit: int | None = None):
        assert (dataset, split) == ("synthetic", "debug")
        image = torch.zeros(3, 4, 4)
        image[0, :, :2] = 1.0
        image[1, :, 2:] = 1.0
        self.samples = [{
            "image": image,
            "image_id": "synthetic-kernelshap-0001",
            "target": 0,
            "class_name": "positive",
            "image_path": "synthetic.png",
        }]
        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


class FakeCaptumKernelShap:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def attribute(self, **kwargs):
        image = kwargs["inputs"]
        height, width = image.shape[-2:]
        signed = torch.linspace(-1, 1, height * width).view(1, 1, height, width)
        return signed.expand_as(image)


def _config(tmp_path: Path) -> dict:
    output = tmp_path / "results"
    return {
        "method": "kernelshap",
        "dataset": {"name": "synthetic", "split": "debug"},
        "model": {
            "name": "tiny", "weights": "none", "num_classes": 2,
            "task_type": "multiclass", "output_activation": "softmax", "checkpoint": None,
        },
        "attribution": {
            "n_samples": 8, "perturbations_per_eval": 4,
            "feature_grid_size": 2, "seed": 42,
        },
        "metrics": {
            "names": ["efficiency_time_ms", "faithfulness_morf_auc_raw"],
            "deletion_fractions": [0.0, 0.5, 1.0],
        },
        "runtime": {
            "device": "cpu", "seed": 42, "max_images": 1,
            "warmup_runs": 1, "resume": True,
        },
        "output": {
            "per_image_csv": str(output / "per_image.csv"),
            "units_csv": str(output / "units.csv"),
            "predictions_csv": str(output / "predictions.csv"),
            "config_dir": str(output / "configs"),
            "run_log": str(output / "run_log.jsonl"),
            "save_maps": False,
        },
    }


def test_runner_dispatches_kernelshap_writes_raw_metric_and_resumes(tmp_path, monkeypatch):
    config = _config(tmp_path)
    path = tmp_path / "kernelshap.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(run_unit, "load_model", lambda **kwargs: TinyClassifier().eval())
    monkeypatch.setattr(run_unit, "MetadataDataset", SyntheticDataset)
    monkeypatch.setattr(
        kernelshap_module, "_load_captum_kernel_shap", lambda: FakeCaptumKernelShap
    )

    run_unit.run(path)
    run_unit.run(path)

    with Path(config["output"]["per_image_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"kernelshap"}
    assert {row["metric"] for row in rows} == {
        "efficiency_time_ms", "faithfulness_morf_auc_raw"
    }
    with Path(config["output"]["units_csv"]).open(newline="", encoding="utf-8") as handle:
        units = list(csv.DictReader(handle))
    assert len(units) == 2
    assert {row["config_hash"] for row in units} == {run_unit._config_hash(config)}
    logs = [json.loads(line) for line in Path(config["output"]["run_log"]).read_text(
        encoding="utf-8"
    ).splitlines()]
    assert [(entry["processed"], entry["skipped"]) for entry in logs] == [(1, 0), (0, 1)]


def test_six_configs_match_current_40_image_candidate_protocol():
    for model in ("vgg16", "resnet50", "densenet121"):
        for dataset in ("imagenet", "voc"):
            path = Path("configs") / f"kernelshap_{model}_{dataset}.yaml"
            config = run_unit._load_config(path)
            assert config["model"]["name"] == model
            assert config["dataset"] == {"name": dataset, "split": "debug"}
            assert config["model"]["output_activation"] == (
                "sigmoid" if dataset == "voc" else "softmax"
            )
            assert config["metrics"]["names"] == [
                "efficiency_time_ms", "faithfulness_morf_auc_raw"
            ]
            assert len(config["metrics"]["deletion_fractions"]) == 21
            assert config["attribution"]["n_samples"] == 2048
            assert config["attribution"]["feature_grid_size"] == 7
            assert config["runtime"]["warmup_runs"] == 1
            assert config["runtime"]["max_images"] == 40
            assert config["output"]["save_float_maps"] is True


def test_real_captum_kernelshap_on_tiny_model_if_installed():
    pytest.importorskip("captum")
    from experiments.attribution.kernelshap import KernelSHAP

    model = TinyClassifier().eval()
    image = next(iter(SyntheticDataset("synthetic")))["image"].unsqueeze(0)
    attribution = KernelSHAP(
        model, n_samples=12, perturbations_per_eval=4, feature_grid_size=2
    ).attribute(image, target=0, baseline=torch.zeros(1, 3, 1, 1))
    assert attribution.shape == (4, 4)
    assert torch.isfinite(attribution).all()
    assert 0 <= float(attribution.min()) <= float(attribution.max()) <= 1


def test_real_captum_kernelshap_runs_through_runner(tmp_path, monkeypatch):
    pytest.importorskip("captum")
    config = _config(tmp_path)
    config["runtime"]["warmup_runs"] = 0
    config["attribution"]["n_samples"] = 12
    config["output"]["save_maps"] = True
    config["output"]["maps_dir"] = str(tmp_path / "maps")
    config["output"]["save_float_maps"] = True
    config["output"]["float_maps_dir"] = str(tmp_path / "maps_float")
    path = tmp_path / "real_kernelshap.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(run_unit, "load_model", lambda **kwargs: TinyClassifier().eval())
    monkeypatch.setattr(run_unit, "MetadataDataset", SyntheticDataset)

    run_unit.run(path)

    with Path(config["output"]["per_image_csv"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == {
        "efficiency_time_ms", "faithfulness_morf_auc_raw"
    }
    for row in rows:
        assert float(row["value"]) >= 0
    assert list((tmp_path / "maps").rglob("*.png"))
    assert list((tmp_path / "maps_float").rglob("*.npy"))
