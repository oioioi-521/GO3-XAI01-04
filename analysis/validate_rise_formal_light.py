"""Validate C's received RISE formal light ZIP without executing its contents."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = "C_RISE_EVAL_LIGHT_20261009_214005/"
MODELS = ("resnet50", "densenet121", "vgg16")
DATASETS = ("imagenet", "voc")
METRICS = ("efficiency_time_ms", "faithfulness_morf_auc_raw")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(archive: ZipFile, path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(ROOT + path)), dtype={"image_id": str})


def close(a: float, b: float) -> bool:
    return bool(np.isclose(a, b, rtol=1e-10, atol=1e-12))


def validate(path: Path) -> dict:
    with ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) and all(
            name.startswith(ROOT) and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts and "\\" not in name
            for name in names), "unsafe or duplicate ZIP paths")
        manifest = json.loads(archive.read(ROOT + "file_manifest.json"))
        entries = manifest["files"]
        listed = [ROOT + entry["path"] for entry in entries]
        require(len(listed) == len(set(listed)) and set(listed) == set(names) - {ROOT + "file_manifest.json"},
                "manifest file set mismatch")
        for entry in entries:
            contents = archive.read(ROOT + entry["path"])
            require(len(contents) == entry["bytes"]
                    and hashlib.sha256(contents).hexdigest() == entry["sha256"],
                    f"file integrity: {entry['path']}")

        metadata = read_csv(archive, "data/metadata.csv")
        predictions = read_csv(archive, "results/predictions.csv")
        per_image = read_csv(archive, "results/per_image.csv")
        units = read_csv(archive, "results/units.csv")
        require(len(predictions) == 2760 and len(per_image) == 5520 and len(units) == 12,
                "total row counts")
        require(not predictions.duplicated(["dataset", "model", "image_id"]).any()
                and not per_image.duplicated(["dataset", "model", "image_id", "metric"]).any()
                and not units.duplicated(["dataset", "model", "metric"]).any(), "duplicate keys")
        require(set(predictions.split) == {"eval"} and set(predictions.model) == set(MODELS)
                and set(predictions.dataset) == set(DATASETS) and set(per_image.method) == {"rise"}
                and set(units.method) == {"rise"}, "mixed identity or split")
        require(np.isfinite(per_image.value).all(), "nonfinite metric")

        logs = [json.loads(line) for line in archive.read(ROOT + "results/run_log.jsonl").splitlines()]
        output = []
        for model in MODELS:
            for dataset in DATASETS:
                unit = f"{model}_{dataset}"
                frozen = metadata.loc[(metadata.dataset == dataset) & (metadata.split == "eval")]
                require(len(frozen) == 460 and frozen.image_id.is_unique, f"frozen metadata: {unit}")
                frozen_ids = set(frozen.image_id)
                pred = predictions.loc[(predictions.model == model) & (predictions.dataset == dataset)]
                values = per_image.loc[(per_image.model == model) & (per_image.dataset == dataset)]
                summary = units.loc[(units.model == model) & (units.dataset == dataset)]
                require(len(pred) == 460 and set(pred.image_id) == frozen_ids, f"predictions: {unit}")
                require(len(values) == 920 and set(values.metric) == set(METRICS), f"metric rows: {unit}")
                require(len(summary) == 2 and set(summary.metric) == set(METRICS), f"summary rows: {unit}")
                require(set(pred.checkpoint_sha256.dropna()) and len(set(pred.checkpoint_sha256.dropna())) == 1,
                        f"checkpoint identities: {unit}")
                targets = dict(zip(frozen.image_id, frozen.class_id.astype(int)))
                require(all(int(row.target_class_id) == targets[row.image_id] for row in pred.itertuples(index=False)),
                        f"metadata targets: {unit}")
                config = {row.metric: str(row.config_hash) for row in summary.itertuples(index=False)}
                require(len(set(config.values())) == 1, f"summary config hash: {unit}")
                state = json.loads(archive.read(ROOT + f"results/run_state/rise_{unit}.json"))
                report = json.loads(archive.read(ROOT + f"validation/{unit}.json"))
                require(state["config_hash"] == config[METRICS[0]] == report["config_hash"],
                        f"config binding: {unit}")
                require(report["status"] == "passed" and report["predictions"] == 460
                        and report["metric_rows"] == 920, f"validation report: {unit}")
                unit_logs = [row for row in logs if row.get("model") == model and row.get("dataset") == dataset]
                require(len(unit_logs) >= 2 and unit_logs[0]["processed"] == 460
                        and unit_logs[-1]["processed"] == 0 and unit_logs[-1]["skipped"] == 460
                        and all(row["config_hash"] == config[METRICS[0]] and row["split"] == "eval"
                                for row in unit_logs), f"resume evidence: {unit}")
                for metric in METRICS:
                    subset = values.loc[values.metric == metric]
                    row = summary.loc[summary.metric == metric].iloc[0]
                    reported = report["metrics"][metric]
                    require(len(subset) == 460 and set(subset.image_id) == frozen_ids and row.n == 460,
                            f"metric coverage: {unit}/{metric}")
                    require(close(subset.value.mean(), row["mean"])
                            and close(subset.value.std(ddof=1), row["std"])
                            and close(row["mean"], reported["mean"])
                            and close(row["std"], reported["std"])
                            and reported["count"] == 460, f"metric aggregation: {unit}/{metric}")
                correct = pred.set_index("image_id").correct.astype(int)
                faith = values.loc[values.metric == METRICS[1]].set_index("image_id").value
                for is_correct in (0, 1):
                    cohort = faith.loc[correct[correct == is_correct].index]
                    reported = report["faithfulness_groups"][str(is_correct)]
                    require(len(cohort) == reported["n"] and close(cohort.mean(), reported["mean"])
                            and close(cohort.std(ddof=1), reported["std"]),
                            f"correctness cohort: {unit}/{is_correct}")
                output.append({"unit": unit, "images": 460, "correct": int(correct.sum()),
                               "faithfulness_raw_mean": float(summary.loc[summary.metric == METRICS[1], "mean"].iloc[0]),
                               "efficiency_time_ms_mean": float(summary.loc[summary.metric == METRICS[0], "mean"].iloc[0]),
                               "resume": "processed=0 skipped=460"})
        return {"zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "manifest_files": len(entries), "units": output,
                "limit": "light ZIP omits raw NPY/PNG maps and model weights; their original content cannot be independently verified here"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.zip), indent=2))


if __name__ == "__main__":
    main()
