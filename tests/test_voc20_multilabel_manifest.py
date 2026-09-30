import csv
import json
from pathlib import Path

import pytest

from preprocessing.build_voc20_multilabel_manifests import VOC_CLASSES, build_manifests


def _xml(names):
    return "<annotation>" + "".join(f"<object><name>{name}</name></object>" for name in names) + "</annotation>"


def _fixture(tmp_path: Path, unknown=False):
    root = tmp_path / "voc" / "VOCdevkit" / "VOC2007"
    (root / "Annotations").mkdir(parents=True); (root / "JPEGImages").mkdir(); (root / "ImageSets" / "Main").mkdir(parents=True)
    ids = [f"{i:06d}" for i in range(1, 42)]
    (root / "ImageSets" / "Main" / "trainval.txt").write_text("\n".join(ids) + "\n")
    for index, image_id in enumerate(ids):
        name = "unknown" if unknown and index == 1 else VOC_CLASSES[index % len(VOC_CLASSES)]
        (root / "Annotations" / f"{image_id}.xml").write_text(_xml([name]))
        (root / "JPEGImages" / f"{image_id}.jpg").write_bytes(b"jpeg")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("image_id,dataset\n000001,voc\n", encoding="utf-8")
    return root, metadata


def test_manifest_is_deterministic_complete_and_leakage_safe(tmp_path):
    root, metadata = _fixture(tmp_path)
    outputs = build_manifests(root, metadata, tmp_path / "out", seed=42)
    train = list(csv.DictReader(outputs["train"].open()))
    validation = list(csv.DictReader(outputs["validation"].open()))
    frozen = set(outputs["exclusions"].read_text().split())
    train_ids, validation_ids = {r["image_id"] for r in train}, {r["image_id"] for r in validation}
    assert not (train_ids & validation_ids)
    assert not ((train_ids | validation_ids) & frozen)
    assert len(train) + len(validation) == 40
    assert all(len(json.loads(row["targets"])) == 20 for row in train + validation)
    version = json.loads(outputs["version"].read_text())
    assert version["classes"] == VOC_CLASSES and version["seed"] == 42


def test_manifest_rejects_unknown_label(tmp_path):
    root, metadata = _fixture(tmp_path, unknown=True)
    with pytest.raises(ValueError, match="unknown VOC labels"):
        build_manifests(root, metadata, tmp_path / "out")
