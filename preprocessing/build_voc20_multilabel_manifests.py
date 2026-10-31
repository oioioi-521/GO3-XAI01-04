"""Build reproducible, leakage-safe VOC2007 multi-label train/validation manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:  # Supports both ``python preprocessing/...py`` and package imports in tests.
    from preprocessing.extract_voc_labels import VOC_CLASSES
except ModuleNotFoundError:  # pragma: no cover - direct-script path only
    from extract_voc_labels import VOC_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
VOC_ROOT = DATA_DIR / "voc" / "VOCdevkit" / "VOC2007"
FIELDS = ("image_id", "image_path", "targets")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_ids(metadata_path: Path) -> set[str]:
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        return {f"{int(row['image_id']):06d}" for row in csv.DictReader(handle) if row["dataset"] == "voc"}


def labels_for(xml_path: Path) -> list[int]:
    names = {(node.findtext("name") or "").strip() for node in ET.parse(xml_path).findall("object")}
    unknown = names.difference(VOC_CLASSES)
    if unknown:
        raise ValueError(f"unknown VOC labels in {xml_path.name}: {sorted(unknown)}")
    if not names:
        raise ValueError(f"annotation without object: {xml_path.name}")
    return [int(name in names) for name in VOC_CLASSES]


def split_rows(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Iterative per-class deterministic allocation with every class represented in both splits."""
    rng = random.Random(seed)
    remaining = {row["image_id"]: row for row in rows}
    validation: dict[str, dict[str, str]] = {}
    by_class: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        for index, value in enumerate(json.loads(row["targets"])):
            if value:
                by_class[index].append(row["image_id"])
    # Choose roughly 10% positives of each class, preserving multi-label rows selected earlier.
    for index in range(len(VOC_CLASSES)):
        candidates = sorted(image_id for image_id in by_class[index] if image_id in remaining)
        rng.shuffle(candidates)
        desired = max(1, round(len(by_class[index]) * 0.10))
        current = sum(json.loads(row["targets"])[index] for row in validation.values())
        for image_id in candidates[: max(0, desired - current)]:
            validation[image_id] = remaining.pop(image_id)
    # Fill to a global 10% deterministically; this does not remove class coverage.
    target_size = round(len(rows) * 0.10)
    candidates = sorted(remaining)
    rng.shuffle(candidates)
    for image_id in candidates[: max(0, target_size - len(validation))]:
        validation[image_id] = remaining.pop(image_id)
    return sorted(remaining.values(), key=lambda row: row["image_id"]), sorted(validation.values(), key=lambda row: row["image_id"])


def build_manifests(voc_root: Path, metadata_path: Path, output_dir: Path, seed: int = 42) -> dict[str, Path]:
    trainval = [line.strip() for line in (voc_root / "ImageSets" / "Main" / "trainval.txt").read_text().splitlines() if line.strip()]
    if len(trainval) != len(set(trainval)):
        raise ValueError("trainval.txt contains duplicate IDs")
    frozen = frozen_ids(metadata_path)
    candidates = sorted(set(trainval).difference(frozen))
    if len(frozen.intersection(trainval)) != len(frozen):
        raise ValueError("not every frozen VOC ID is present in trainval")
    rows = []
    data_dir = metadata_path.parent
    for image_id in candidates:
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        if not xml_path.is_file() or not image_path.is_file():
            raise FileNotFoundError(f"missing XML or JPEG for {image_id}")
        rows.append({"image_id": image_id, "image_path": str(image_path.relative_to(data_dir)).replace("\\", "/"), "targets": json.dumps(labels_for(xml_path), separators=(",", ":"))})
    train, validation = split_rows(rows, seed)
    for index, name in enumerate(VOC_CLASSES):
        if not any(json.loads(row["targets"])[index] for row in train) or not any(json.loads(row["targets"])[index] for row in validation):
            raise ValueError(f"class {name} is absent from a split")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"train": output_dir / "voc20_multilabel_train.csv", "validation": output_dir / "voc20_multilabel_validation.csv", "exclusions": output_dir / "voc20_multilabel_frozen_exclusions.txt", "version": output_dir / "VOC20_MULTILABEL_VERSION.json"}
    for key in ("train", "validation"):
        with outputs[key].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(train if key == "train" else validation)
    outputs["exclusions"].write_text("\n".join(sorted(frozen)) + "\n", encoding="utf-8")
    positives = Counter()
    for row in rows:
        for index, value in enumerate(json.loads(row["targets"])):
            positives[VOC_CLASSES[index]] += value
    payload = {"schema": 1, "seed": seed, "task_type": "multilabel", "output_activation": "sigmoid", "classes": VOC_CLASSES, "frozen_count": len(frozen), "candidate_count": len(rows), "train_count": len(train), "validation_count": len(validation), "positive_counts": dict(positives), "sha256": {key: sha256(path) for key, path in outputs.items() if key != "version"}, "metadata_sha256": sha256(metadata_path)}
    outputs["version"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    outputs = build_manifests(VOC_ROOT, DATA_DIR / "metadata.csv", DATA_DIR, args.seed)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
