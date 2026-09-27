from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.merge_eval_handoff import (
    DATASETS,
    METRICS,
    MODELS,
    HandoffValidationError,
    validate_and_merge,
)
from analysis.schema import PER_IMAGE_COLUMNS, UNIT_COLUMNS
from experiments.predictions import PREDICTION_COLUMNS


def _build_handoff(root: Path, *, images: int = 2) -> None:
    units = []
    for model in MODELS:
        for dataset in DATASETS:
            unit = f"{model}_{dataset}"
            result = root / "results" / "occlusion_eval" / unit
            result.mkdir(parents=True)
            config_hash = f"cfg-{unit}"
            prediction_hash = f"pred-{unit}"
            weight_hash = f"weight-{unit}"
            per_rows = []
            prediction_rows = []
            for index in range(images):
                image_id = f"{dataset}-{index}"
                values = {"efficiency_time_ms": 10.0 + 10 * index, "faithfulness_morf_auc_raw": 0.2 + 0.2 * index}
                for metric, value in values.items():
                    per_rows.append({
                        "image_id": image_id, "dataset": dataset, "model": model,
                        "method": "occlusion", "metric": metric, "value": value,
                        "time_ms": value if metric == "efficiency_time_ms" else "",
                    })
                prediction_rows.append({
                    **dict.fromkeys(PREDICTION_COLUMNS, ""),
                    "image_id": image_id, "dataset": dataset, "split": "eval",
                    "model": model, "weights": "default", "target_class_id": "0",
                    "target_class_name": "target", "predicted_class_id": str(index),
                    "predicted_class_name": "prediction", "confidence": "0.8",
                    "correct": str(int(index == 0)), "checkpoint_sha256": weight_hash,
                    "config_hash": prediction_hash,
                })
            per_image = pd.DataFrame(per_rows, columns=PER_IMAGE_COLUMNS)
            per_image.to_csv(result / "per_image.csv", index=False)
            summary = (
                per_image.assign(value=pd.to_numeric(per_image["value"]))
                .groupby(["method", "model", "dataset", "metric"], as_index=False)["value"]
                .agg(mean="mean", std="std", n="count")
            )
            summary["config_hash"] = config_hash
            summary[UNIT_COLUMNS].to_csv(result / "units.csv", index=False)
            pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).to_csv(
                result / "predictions.csv", index=False
            )
            raw = per_image[per_image["metric"] == "faithfulness_morf_auc_raw"]["value"].astype(float)
            timing = per_image[per_image["metric"] == "efficiency_time_ms"]["value"].astype(float)
            units.append({
                "unit": unit, "result_directory": f"results/occlusion_eval/{unit}",
                "config_hash": config_hash, "prediction_context_hash": prediction_hash,
                "processed": images, "skipped": 0, "failed": 0,
                "raw_mean": raw.mean(), "raw_std": raw.std(),
                "attribution_ms_mean": timing.mean(), "weight": {"sha256": weight_hash},
            })
    manifest = {"schema_version": 1, "units": units}
    docs = root / "docs"
    docs.mkdir()
    (docs / "B_OCCLUSION_EVAL_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_validate_and_merge_six_units_and_top1_groups(tmp_path):
    _build_handoff(tmp_path)
    tables = validate_and_merge(tmp_path, expected_images=2)
    assert len(tables.per_image) == 24
    assert len(tables.units) == 12
    assert len(tables.predictions) == 12
    assert set(tables.by_top1_correct["prediction_group"]) == {
        "top1_correct", "top1_incorrect"
    }
    assert (tables.prediction_summary["top1_target_hit_rate"] == 0.5).all()
    assert len(tables.common_correct_cohort) == 4
    assert tables.common_correct_cohort["correct_all_models"].sum() == 2


def test_validate_and_merge_rejects_config_hash_drift(tmp_path):
    _build_handoff(tmp_path)
    path = tmp_path / "results/occlusion_eval/resnet50_imagenet/units.csv"
    frame = pd.read_csv(path)
    frame["config_hash"] = "wrong"
    frame.to_csv(path, index=False)
    with pytest.raises(HandoffValidationError, match="config hash"):
        validate_and_merge(tmp_path, expected_images=2)


def test_validate_and_merge_rejects_unit_identity_drift(tmp_path):
    _build_handoff(tmp_path)
    path = tmp_path / "results/occlusion_eval/resnet50_imagenet/units.csv"
    frame = pd.read_csv(path)
    frame["model"] = "vgg16"
    frame.to_csv(path, index=False)
    with pytest.raises(HandoffValidationError, match="summary identity"):
        validate_and_merge(tmp_path, expected_images=2)


def test_validate_and_merge_rejects_manifest_path_escape(tmp_path):
    _build_handoff(tmp_path)
    path = tmp_path / "docs/B_OCCLUSION_EVAL_MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["units"][0]["result_directory"] = "../outside"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HandoffValidationError, match="escapes"):
        validate_and_merge(tmp_path, expected_images=2)
