import csv
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("captum")
yaml = pytest.importorskip("yaml")

from experiments import run_unit


class TinyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, kernel_size=1, bias=False),
            torch.nn.ReLU(),
        )
        with torch.no_grad():
            self.features[0].weight.zero_()
            self.features[0].weight[0, 0, 0, 0] = 1.0

    def forward(self, inputs):
        activation = self.features(inputs)
        score = activation.sum(dim=(1, 2, 3))
        return torch.stack((score, -score), dim=1)


class SyntheticDataset:
    def __init__(self, dataset, split="debug", limit=None):
        assert (dataset, split) == ("synthetic", "debug")
        image = torch.zeros(3, 4, 4)
        image[0, :, :2] = 1.0
        self.samples = [{
            "image": image,
            "image_id": "synthetic-gradient-0001",
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


def _config(tmp_path: Path, method: str):
    output = tmp_path / method
    attribution = (
        {
            "n_steps": 4,
            "internal_batch_size": 2,
            "approximation_method": "riemann_middle",
        }
        if method == "ig"
        else {"target_layer": "features.0"}
    )
    return {
        "method": method,
        "dataset": {"name": "synthetic", "split": "debug"},
        "model": {
            "name": "tiny",
            "weights": "none",
            "num_classes": 2,
            "checkpoint": None,
        },
        "attribution": attribution,
        "metrics": {"names": ["efficiency_time_ms"]},
        "runtime": {"device": "cpu", "seed": 7, "max_images": 1, "resume": True},
        "output": {
            "per_image_csv": str(output / "per_image.csv"),
            "units_csv": str(output / "units.csv"),
            "predictions_csv": str(output / "predictions.csv"),
            "config_dir": str(output / "configs"),
            "run_log": str(output / "run_log.jsonl"),
            "save_maps": False,
        },
    }


@pytest.mark.parametrize("method", ["ig", "gradcam"])
def test_runner_executes_and_resumes_gradient_methods(tmp_path, monkeypatch, method):
    config = _config(tmp_path, method)
    config_path = tmp_path / f"{method}.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(run_unit, "load_model", lambda **kwargs: TinyCNN().eval())
    monkeypatch.setattr(run_unit, "MetadataDataset", SyntheticDataset)

    run_unit.run(config_path)
    run_unit.run(config_path)

    with Path(config["output"]["per_image_csv"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["method"] == method
    assert rows[0]["metric"] == "efficiency_time_ms"

    with Path(config["output"]["units_csv"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 1
    assert summaries[0]["config_hash"] == run_unit._config_hash(config)

    logs = [
        json.loads(line)
        for line in Path(config["output"]["run_log"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(entry["processed"], entry["skipped"]) for entry in logs] == [
        (1, 0),
        (0, 1),
    ]


def test_twelve_formal_gradient_configs_cover_the_assigned_matrix():
    configs = []
    for method in ("ig", "gradcam"):
        for model in ("vgg16", "resnet50", "densenet121"):
            for dataset in ("imagenet", "voc"):
                path = Path("configs") / f"{method}_{model}_{dataset}.yaml"
                assert path.is_file(), path
                config = run_unit._load_config(path)
                configs.append(
                    (
                        config["method"],
                        config["model"]["name"],
                        config["dataset"]["name"],
                    )
                )
                assert config["dataset"]["split"] == "eval"
                assert config["runtime"]["max_images"] is None
    assert len(set(configs)) == 12
