"""Validate and merge a six-unit formal-eval handoff.

The handoff keeps each model/dataset unit in an isolated result directory.
This module validates those directories against the tracked brief manifest,
then produces deterministic merged tables for downstream analysis.

Example:
    python -m analysis.merge_eval_handoff \
        --root results/incoming/B_OCCLUSION_EVAL_LIGHT \
        --output-dir results/analysis/occlusion_eval_acceptance
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.predictions import PREDICTION_COLUMNS

from .schema import read_csv, validate_per_image, validate_units


MODELS = ("resnet50", "densenet121", "vgg16")
DATASETS = ("imagenet", "voc")
METRICS = ("efficiency_time_ms", "faithfulness_morf_auc_raw")


class HandoffValidationError(ValueError):
    """Raised when a handoff cannot safely enter formal analysis."""


@dataclass(frozen=True)
class HandoffTables:
    per_image: pd.DataFrame
    units: pd.DataFrame
    predictions: pd.DataFrame
    by_top1_correct: pd.DataFrame
    prediction_summary: pd.DataFrame
    common_correct_cohort: pd.DataFrame


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HandoffValidationError(message)


def _resolve_under(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / Path(str(relative))).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise HandoffValidationError(
            f"manifest path escapes handoff root: {relative}"
        ) from error
    return candidate


def _close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12))


def validate_and_merge(
    root: str | Path,
    *,
    expected_images: int = 460,
    manifest_path: str = "docs/B_OCCLUSION_EVAL_MANIFEST.json",
) -> HandoffTables:
    """Validate one formal handoff and return merged analysis tables."""

    handoff_root = Path(root).resolve()
    _require(handoff_root.is_dir(), f"handoff root does not exist: {handoff_root}")
    manifest_file = _resolve_under(handoff_root, manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1, "unsupported handoff schema")

    entries = manifest.get("units")
    _require(isinstance(entries, list), "manifest.units must be a list")
    expected_units = {
        f"{model}_{dataset}" for model in MODELS for dataset in DATASETS
    }
    indexed = {str(entry.get("unit")): entry for entry in entries}
    _require(
        len(indexed) == len(entries) and set(indexed) == expected_units,
        "manifest must contain exactly the six model/dataset units",
    )

    per_image_frames: list[pd.DataFrame] = []
    unit_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for model in MODELS:
        for dataset in DATASETS:
            unit_name = f"{model}_{dataset}"
            entry = indexed[unit_name]
            _require(
                int(entry.get("processed", -1)) == expected_images,
                f"processed count mismatch: {unit_name}",
            )
            _require(
                int(entry.get("skipped", -1)) == 0,
                f"skipped rows present: {unit_name}",
            )
            _require(
                int(entry.get("failed", -1)) == 0,
                f"failed rows present: {unit_name}",
            )

            result_dir = _resolve_under(handoff_root, str(entry["result_directory"]))
            per_image = read_csv(result_dir / "per_image.csv", "per_image")
            units = read_csv(result_dir / "units.csv", "units")
            predictions = read_csv(result_dir / "predictions.csv", "predictions")

            validate_per_image(per_image)
            validate_units(units)
            _require(
                list(predictions.columns) == PREDICTION_COLUMNS,
                f"prediction schema mismatch: {unit_name}",
            )
            _require(
                len(predictions) == expected_images,
                f"prediction count mismatch: {unit_name}",
            )
            _require(
                not predictions["image_id"].astype(str).duplicated().any(),
                f"duplicate predictions: {unit_name}",
            )
            _require(
                set(predictions["split"].astype(str)) == {"eval"},
                f"non-eval predictions present: {unit_name}",
            )
            _require(
                set(predictions["dataset"].astype(str)) == {dataset}
                and set(predictions["model"].astype(str)) == {model},
                f"prediction identity mismatch: {unit_name}",
            )
            correct = pd.to_numeric(predictions["correct"], errors="coerce")
            _require(
                correct.notna().all() and set(correct.astype(int)) <= {0, 1},
                f"invalid correct flag: {unit_name}",
            )

            weight = entry.get("weight") or {}
            _require(
                set(predictions["checkpoint_sha256"].astype(str))
                == {str(weight.get("sha256"))},
                f"checkpoint identity mismatch: {unit_name}",
            )
            _require(
                set(predictions["config_hash"].astype(str))
                == {str(entry.get("prediction_context_hash"))},
                f"prediction context mismatch: {unit_name}",
            )

            _require(
                len(per_image) == expected_images * len(METRICS),
                f"metric row count mismatch: {unit_name}",
            )
            _require(
                set(per_image["dataset"].astype(str)) == {dataset}
                and set(per_image["model"].astype(str)) == {model}
                and set(per_image["method"].astype(str)) == {"occlusion"},
                f"per-image identity mismatch: {unit_name}",
            )
            _require(
                set(per_image["metric"].astype(str)) == set(METRICS),
                f"metric set mismatch: {unit_name}",
            )
            prediction_ids = set(predictions["image_id"].astype(str))
            metric_ids = set(per_image["image_id"].astype(str))
            _require(
                prediction_ids == metric_ids and len(metric_ids) == expected_images,
                f"image set mismatch: {unit_name}",
            )

            _require(
                len(units) == len(METRICS),
                f"unit summary row count mismatch: {unit_name}",
            )
            _require(
                set(units["metric"].astype(str)) == set(METRICS),
                f"unit metric mismatch: {unit_name}",
            )
            _require(
                set(units["dataset"].astype(str)) == {dataset}
                and set(units["model"].astype(str)) == {model}
                and set(units["method"].astype(str)) == {"occlusion"},
                f"unit summary identity mismatch: {unit_name}",
            )
            _require(
                set(units["config_hash"].astype(str)) == {str(entry.get("config_hash"))},
                f"unit config hash mismatch: {unit_name}",
            )

            numeric = per_image.assign(
                value=pd.to_numeric(per_image["value"], errors="raise")
            )
            recomputed = numeric.groupby("metric")["value"].agg(["mean", "std", "count"])
            for row in units.itertuples(index=False):
                current = recomputed.loc[str(row.metric)]
                _require(
                    int(row.n) == expected_images == int(current["count"]),
                    f"summary n mismatch: {unit_name}/{row.metric}",
                )
                _require(
                    _close(float(row.mean), float(current["mean"])),
                    f"summary mean mismatch: {unit_name}/{row.metric}",
                )
                _require(
                    _close(float(row.std), float(current["std"])),
                    f"summary std mismatch: {unit_name}/{row.metric}",
                )

            raw = recomputed.loc["faithfulness_morf_auc_raw"]
            timing = recomputed.loc["efficiency_time_ms"]
            _require(
                _close(float(entry["raw_mean"]), float(raw["mean"])),
                f"manifest raw mean mismatch: {unit_name}",
            )
            _require(
                _close(float(entry["raw_std"]), float(raw["std"])),
                f"manifest raw std mismatch: {unit_name}",
            )
            _require(
                _close(float(entry["attribution_ms_mean"]), float(timing["mean"])),
                f"manifest timing mismatch: {unit_name}",
            )

            per_image_frames.append(per_image)
            unit_frames.append(units)
            prediction_frames.append(predictions)

    per_image_all = pd.concat(per_image_frames, ignore_index=True)
    units_all = pd.concat(unit_frames, ignore_index=True)
    predictions_all = pd.concat(prediction_frames, ignore_index=True)
    validate_per_image(per_image_all)
    validate_units(units_all)

    joined = per_image_all.merge(
        predictions_all[["image_id", "dataset", "model", "correct"]],
        on=["image_id", "dataset", "model"],
        how="left",
        validate="many_to_one",
    )
    _require(joined["correct"].notna().all(), "metric rows are missing predictions")
    joined["prediction_group"] = np.where(
        pd.to_numeric(joined["correct"]).astype(int).eq(1),
        "top1_correct",
        "top1_incorrect",
    )
    joined["value"] = pd.to_numeric(joined["value"], errors="raise")
    by_correct = (
        joined.groupby(
            ["method", "model", "dataset", "metric", "prediction_group"],
            as_index=False,
            sort=True,
        )["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index(drop=True)
    )
    by_correct["std"] = by_correct["std"].fillna(0.0)

    prediction_summary = (
        predictions_all.assign(
            correct=pd.to_numeric(predictions_all["correct"], errors="raise").astype(int)
        )
        .groupby(["model", "dataset"], as_index=False, sort=True)["correct"]
        .agg(top1_correct="sum", n="count", top1_target_hit_rate="mean")
    )

    common_correct_cohort = (
        predictions_all.assign(
            correct=pd.to_numeric(predictions_all["correct"], errors="raise").astype(int)
        )
        .pivot(index=["dataset", "image_id"], columns="model", values="correct")
        .reset_index()
    )
    _require(
        set(common_correct_cohort.columns) == {"dataset", "image_id", *MODELS},
        "common-correct cohort is missing one or more model columns",
    )
    _require(
        common_correct_cohort[list(MODELS)].notna().all().all(),
        "common-correct cohort has incomplete model predictions",
    )
    common_correct_cohort["correct_all_models"] = (
        common_correct_cohort[list(MODELS)].astype(int).eq(1).all(axis=1).astype(int)
    )
    common_correct_cohort = common_correct_cohort[
        ["dataset", "image_id", *MODELS, "correct_all_models"]
    ]

    return HandoffTables(
        per_image=per_image_all,
        units=units_all,
        predictions=predictions_all,
        by_top1_correct=by_correct,
        prediction_summary=prediction_summary,
        common_correct_cohort=common_correct_cohort,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, default=460)
    args = parser.parse_args()

    tables = validate_and_merge(args.root, expected_images=args.expected_images)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_image.csv": tables.per_image,
        "units.csv": tables.units,
        "predictions.csv": tables.predictions,
        "by_top1_correct.csv": tables.by_top1_correct,
        "prediction_summary.csv": tables.prediction_summary,
        "common_correct_cohort.csv": tables.common_correct_cohort,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    print(
        f"validated units={len(tables.units) // len(METRICS)}, "
        f"images={len(tables.predictions)}, metric_rows={len(tables.per_image)}"
    )
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
