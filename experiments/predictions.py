"""Traceable, method-independent per-image prediction records."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from torchvision import models

from preprocessing.dataset import DATA_DIR, MEAN, STD


PREDICTION_COLUMNS = [
    "image_id",
    "dataset",
    "split",
    "model",
    "weights",
    "target_class_id",
    "target_class_name",
    "predicted_class_id",
    "predicted_class_name",
    "confidence",
    "correct",
    "checkpoint_sha256",
    "config_hash",
]

_DEFAULT_WEIGHTS = {
    "vgg16": models.VGG16_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "densenet121": models.DenseNet121_Weights.DEFAULT,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _weight_sha256(name: str, weights: str, checkpoint: Path | None) -> str:
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found for prediction provenance: {checkpoint}")
        return sha256_file(checkpoint)
    if weights.lower() != "default":
        return ""
    try:
        url = _DEFAULT_WEIGHTS[name].url
    except KeyError as error:
        raise ValueError(f"unsupported model {name!r} for prediction provenance") from error
    path = Path(torch.hub.get_dir()) / "checkpoints" / Path(url).name
    if not path.is_file():
        raise FileNotFoundError(
            f"default torchvision weights were not cached after model loading: {path}"
        )
    return sha256_file(path)


def build_prediction_context(
    *,
    dataset: str,
    split: str,
    model_config: Mapping[str, Any],
    checkpoint: Path | None,
) -> Dict[str, str]:
    """Build a hashable identity excluding attribution-only configuration.

    The frozen metadata hash and image preprocessing are part of the identity,
    so predictions cannot be reused after a data or input-contract change.
    """

    name = str(model_config["name"]).lower()
    weights = str(model_config.get("weights", "default")).lower()
    num_classes = int(model_config.get("num_classes", 1000))
    checkpoint_sha256 = _weight_sha256(name, weights, checkpoint)
    payload = {
        "prediction_schema": 1,
        "dataset": dataset,
        "split": split,
        "metadata_sha256": sha256_file(Path(DATA_DIR) / "metadata.csv"),
        "model": {
            "name": name,
            "weights": weights,
            "num_classes": num_classes,
            "checkpoint_sha256": checkpoint_sha256,
        },
        "preprocessing": {"resize": [224, 224], "mean": MEAN, "std": STD},
    }
    return {
        "dataset": dataset,
        "split": split,
        "model": name,
        "weights": weights,
        "checkpoint_sha256": checkpoint_sha256,
        "config_hash": _canonical_hash(payload),
    }


def class_name_for(dataset: str, class_id: int) -> str:
    """Return the tracked label spelling for a dataset class when available."""

    if dataset == "imagenet":
        index = json.loads((Path(DATA_DIR) / "imagenet_class_index.json").read_text(encoding="utf-8"))
        entry = index.get(str(class_id))
        return str(entry[1]) if entry else ""
    if dataset == "voc":
        with (Path(DATA_DIR) / "voc_labels.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["class_id"]) == class_id:
                    return str(row["class_name"])
    return ""


class PredictionStore:
    """Atomically upsert predictions for one method-independent context."""

    def __init__(self, path: Path, context: Mapping[str, str]) -> None:
        self.path = path
        self.context = dict(context)
        self.rows = self._read_rows()
        self._index: Dict[str, Dict[str, str]] = {}
        for row in self.rows:
            if row.get("config_hash") != self.context["config_hash"]:
                continue
            if any(row.get(key) != self.context[key] for key in ("dataset", "split", "model", "weights", "checkpoint_sha256")):
                continue
            image_id = row.get("image_id", "")
            if not image_id:
                raise ValueError(f"prediction row without image_id in {self.path}")
            if image_id in self._index:
                raise ValueError(
                    f"duplicate prediction rows for image_id={image_id!r}, config_hash={self.context['config_hash']}"
                )
            self._index[image_id] = row

    def _read_rows(self) -> list[Dict[str, str]]:
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != PREDICTION_COLUMNS:
                raise ValueError(f"unexpected prediction CSV schema in {self.path}")
            return list(reader)

    def get(self, image_id: str, target_class_id: int) -> Dict[str, str] | None:
        row = self._index.get(str(image_id))
        if row is not None and int(row["target_class_id"]) != int(target_class_id):
            raise ValueError(
                f"prediction target mismatch for image_id={image_id!r}; frozen metadata may have changed"
            )
        return row

    def record(
        self,
        *,
        image_id: str,
        target_class_id: int,
        target_class_name: str,
        predicted_class_id: int,
        confidence: float,
    ) -> Dict[str, str]:
        if self.get(image_id, target_class_id) is not None:
            raise ValueError(f"prediction already exists for image_id={image_id!r}")
        row = {
            "image_id": str(image_id),
            **self.context,
            "target_class_id": str(int(target_class_id)),
            "target_class_name": str(target_class_name),
            "predicted_class_id": str(int(predicted_class_id)),
            "predicted_class_name": class_name_for(self.context["dataset"], int(predicted_class_id)),
            "confidence": repr(float(confidence)),
            "correct": str(int(predicted_class_id == target_class_id)),
        }
        self.rows.append(row)
        self._index[row["image_id"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.path)
        return row

    def context_rows(self) -> Iterable[Dict[str, str]]:
        return tuple(self._index.values())

    def accuracy(self) -> tuple[int, int, float]:
        rows = tuple(self.context_rows())
        correct = sum(int(row["correct"]) for row in rows)
        total = len(rows)
        return correct, total, (correct / total if total else float("nan"))
