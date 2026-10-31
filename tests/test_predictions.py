import csv
import json

import torch
import yaml

from experiments import run_predictions
from experiments.predictions import PredictionStore, build_prediction_context


def _model_config(num_classes=1000):
    return {"name": "resnet50", "weights": "none", "num_classes": num_classes}


def test_prediction_store_reuses_across_methods_and_isolates_model_changes(tmp_path):
    # The context deliberately has no attribution method, so RISE and
    # Occlusion with the same frozen data/model identity share one prediction.
    context = build_prediction_context(
        dataset="imagenet", split="debug", model_config=_model_config(), checkpoint=None
    )
    same_context_for_another_method = build_prediction_context(
        dataset="imagenet", split="debug", model_config=_model_config(), checkpoint=None
    )
    assert same_context_for_another_method["config_hash"] == context["config_hash"]

    store = PredictionStore(tmp_path / "predictions.csv", context)
    stored = store.record(
        image_id="image-1",
        target_class_id=23,
        target_class_name="vulture",
        predicted_class_id=23,
        confidence=0.75,
    )
    assert stored["correct"] == "1"

    resumed = PredictionStore(tmp_path / "predictions.csv", same_context_for_another_method)
    assert resumed.get("image-1", 23)["predicted_class_id"] == "23"
    assert resumed.accuracy() == (1, 1, 1.0)

    changed_model = build_prediction_context(
        dataset="imagenet", split="debug", model_config=_model_config(num_classes=20), checkpoint=None
    )
    assert changed_model["config_hash"] != context["config_hash"]
    assert PredictionStore(tmp_path / "predictions.csv", changed_model).get("image-1", 23) is None


def test_prediction_context_isolates_output_activation():
    softmax = build_prediction_context(dataset="voc", split="debug", model_config={**_model_config(20), "task_type": "multiclass", "output_activation": "softmax"}, checkpoint=None)
    sigmoid = build_prediction_context(dataset="voc", split="debug", model_config={**_model_config(20), "task_type": "multilabel", "output_activation": "sigmoid"}, checkpoint=None)
    assert softmax["config_hash"] != sigmoid["config_hash"]


class TinyClassifier(torch.nn.Module):
    def forward(self, inputs):
        score = inputs[:, 0].sum(dim=(1, 2))
        return torch.stack((score, -score), dim=1)


class TwoImageDataset:
    def __init__(self, dataset, split="debug", limit=None):
        assert (dataset, split) == ("synthetic", "debug")
        self.samples = [
            {"image": torch.ones(3, 4, 4), "image_id": "one", "target": 0, "class_name": "positive"},
            {"image": torch.zeros(3, 4, 4), "image_id": "two", "target": 0, "class_name": "positive"},
        ]
        if limit is not None:
            self.samples = self.samples[:limit]

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)


def test_prediction_execution_snapshots_include_cli_max_images(tmp_path, monkeypatch):
    config = {
        "method": "occlusion",
        "dataset": {"name": "synthetic", "split": "debug"},
        "model": {"name": "resnet50", "weights": "none", "num_classes": 2, "checkpoint": None},
        "attribution": {"window_height": 2, "window_width": 2, "stride_height": 2, "stride_width": 2},
        "metrics": {"names": ["efficiency_time_ms"]},
        "runtime": {"device": "cpu", "max_images": 1},
        "output": {"per_image_csv": str(tmp_path / "unused.csv"), "units_csv": str(tmp_path / "unused_units.csv"), "predictions_csv": str(tmp_path / "predictions.csv")},
    }
    config_path = tmp_path / "predict.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(run_predictions, "load_model", lambda **kwargs: TinyClassifier().eval())
    monkeypatch.setattr(run_predictions, "MetadataDataset", TwoImageDataset)

    run_predictions.run(config_path, max_images=1)
    run_predictions.run(config_path, max_images=2)
    run_predictions.run(config_path, max_images=2)

    with (tmp_path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    logs = [json.loads(line) for line in (tmp_path / "prediction_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["processed"], row["skipped"]) for row in logs] == [(1, 0), (1, 1), (0, 2)]
    assert logs[0]["config_hash"] == logs[1]["config_hash"]
    assert logs[0]["execution_config_hash"] != logs[1]["execution_config_hash"]
    snapshots = list((tmp_path / "configs").glob("*.yaml"))
    assert {yaml.safe_load(path.read_text(encoding="utf-8"))["runtime"]["max_images"] for path in snapshots} == {1, 2}
