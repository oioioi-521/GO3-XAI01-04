"""Build the 2026-10-14 six-method handoff from locally audited DCU archives.

No inference is run. CSV measurements are preserved; cohort summaries are
recomputed from individual images with sample standard deviations (ddof=1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import aggregate_per_image
from .schema import PER_IMAGE_KEY, UNIT_KEY, validate_per_image, validate_units


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> pd.DataFrame:
    # In particular, VOC IDs and 64-bit noise seeds must never pass through floats.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/team-handoff-20261014"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    require(not output.exists(), f"Refusing to overwrite existing handoff: {output}")
    five = repo / "results/formal-final-evidence"
    lime = repo / "results/lime-vectorized-formal-complete-20261014"
    roots = {"five": five, "lime": lime}
    methods = {"gradcam", "ig", "occlusion", "rise", "kernelshap", "lime"}
    models = {"resnet50", "densenet121", "vgg16"}
    metrics = {"efficiency_time_ms", "faithfulness_morf_auc_raw", "stability_spearman", "stability_valid_rate"}
    sources: dict[str, str] = {}

    def checked(path: Path, expected: str | None = None) -> pd.DataFrame:
        digest = sha256(path)
        require(expected is None or digest == expected, f"Source hash mismatch: {path}")
        sources[str(path.relative_to(repo))] = digest
        return read(path)

    tables: dict[str, list[pd.DataFrame]] = {k: [] for k in ("per_image", "units", "stability_trace")}
    for label, root in roots.items():
        complete = json.loads((root / "run/COMPLETE.json").read_text())
        require(complete["status"] == "verified", f"{label} was not verified")
        for name in tables:
            filename = name + ".csv"
            tables[name].append(checked(root / "run/merged" / filename, complete["outputs"][filename]))
    data = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    per, units, trace = (data[k] for k in ("per_image", "units", "stability_trace"))
    validate_per_image(per)
    validate_units(units)
    require((len(per), len(units), len(trace)) == (66240, 144, 82800), "Unexpected merged row counts")
    require(set(per.method) == methods and set(per.model) == models and set(per.metric) == metrics, "Wrong formal matrix")
    require(per.groupby(UNIT_KEY).size().eq(460).all(), "Each unit/metric must have 460 images")
    recomputed = aggregate_per_image(per, units)
    comparison = units.merge(recomputed, on=UNIT_KEY, suffixes=("_source", "_recomputed"), validate="one_to_one")
    for field in ("mean", "std", "n"):
        require(np.allclose(comparison[field + "_source"].astype(float), comparison[field + "_recomputed"].astype(float), rtol=1e-10, atol=1e-10), f"Summary mismatch: {field}")

    predictions = []
    for unit in sorted((five / "run/formal").iterdir()):
        if not unit.is_dir():
            continue
        completion = json.loads((unit / "completion.json").read_text())
        require(completion["status"] == "verified" and completion["resume_verified"], f"Incomplete unit: {unit.name}")
        prediction = checked(unit / "predictions.csv", completion["csv_sha256"]["predictions.csv"])
        prediction.insert(0, "method", unit.name.split("-")[0])
        predictions.append(prediction)
    lime_complete = json.loads((lime / "run/COMPLETE.json").read_text())
    prediction = checked(lime / "run/merged/predictions.csv", lime_complete["outputs"]["predictions.csv"])
    prediction.insert(0, "method", "lime")
    predictions.append(prediction)
    method_predictions = pd.concat(predictions, ignore_index=True)
    key = ["image_id", "dataset", "model"]
    require(len(method_predictions) == 16560 and not method_predictions.duplicated(key + ["method"]).any(), "Prediction keys incomplete or duplicated")
    require(method_predictions.groupby(key).method.nunique().eq(6).all(), "Missing method prediction")
    context_columns = [c for c in method_predictions.columns if c not in ("method", "config_hash")]
    identity_columns = [c for c in context_columns if c != "confidence"]
    require(method_predictions[identity_columns].drop_duplicates().shape[0] == 2760, "Prediction context differs between methods")
    confidence_groups = method_predictions.assign(confidence=method_predictions.confidence.map(float)).groupby(key).confidence
    confidence_delta = float((confidence_groups.max() - confidence_groups.min()).max())
    require(confidence_delta <= 1e-12, "Prediction confidence differs beyond CSV rounding")
    require(method_predictions.assign(confidence=method_predictions.confidence.astype(np.float32)).groupby(key).confidence.nunique().eq(1).all(), "Float32 prediction confidence differs")
    # The LIME merge round-tripped CSV floats. Preserve all spellings in the
    # method table; use the IG record for the shared prediction context.
    prediction_context = method_predictions.loc[method_predictions.method == "ig", context_columns].sort_values(key)
    metadata = checked(five / "release/data/metadata.csv")
    require(sha256(five / "release/data/metadata.csv") == sha256(lime / "release/data/metadata.csv"), "Frozen metadata differs")
    expected_ids = metadata[metadata.split == "eval"]
    require(len(expected_ids) == 920 and not expected_ids.duplicated(["image_id", "dataset"]).any(), "Frozen eval metadata mismatch")
    for (dataset, model), group in prediction_context.groupby(["dataset", "model"]):
        expected = expected_ids[expected_ids.dataset == dataset].set_index("image_id")
        require(set(group.image_id) == set(expected.index), f"Wrong eval IDs: {dataset}/{model}")
        target = group.set_index("image_id").target_class_id.astype(int).sort_index()
        require(target.equals(expected.class_id.astype(float).astype(int).sort_index().rename("target_class_id")), "Metadata target mismatch")
    require(set(prediction_context.correct) <= {"0", "1"}, "Nonbinary correctness flag")
    require((prediction_context.correct.astype(int) == (prediction_context.target_class_id == prediction_context.predicted_class_id).astype(int)).all(), "Prediction correctness mismatch")

    cohorts = prediction_context[key + ["correct"]].rename(columns={"correct": "model_correct"}).copy()
    cohorts["model_correct"] = cohorts.model_correct.astype(int)
    common = cohorts.groupby(["image_id", "dataset"]).model_correct.agg(["sum", "count"])
    require(common["count"].eq(3).all(), "Missing model in common-correct cohort")
    common["common_correct"] = common["sum"].eq(3).astype(int)
    cohorts = cohorts.merge(common[["common_correct"]], left_on=["image_id", "dataset"], right_index=True, validate="many_to_one")
    cohorts = cohorts.sort_values(["dataset", "model", "image_id"])
    counts = cohorts.groupby(["dataset", "model"], as_index=False).agg(eval_images=("image_id", "count"), model_correct=("model_correct", "sum"), common_correct=("common_correct", "sum"))
    tagged = per.merge(cohorts, on=key, validate="many_to_one")
    require(len(tagged) == len(per), "Missing cohort joins")

    trace_key = ["image_id", "dataset", "model", "method"]
    require(not trace.duplicated(trace_key + ["repeat"]).any(), "Duplicate stability repeat")
    require(trace.groupby(trace_key)["repeat"].agg(lambda x: set(x) == {"0", "1", "2", "3", "4"}).all(), "Incomplete stability repeats")
    require(trace.groupby(key + ["repeat"]).seed.nunique().eq(1).all(), "Noise seeds not paired across methods")
    require(trace.sigma.astype(float).eq(.005).all(), "Unexpected RGB sigma")
    require(trace.protocol_version.eq("rgb-gaussian-spearman-v1").all(), "Unexpected stability protocol")
    traced = trace.merge(prediction_context[key + ["target_class_id"]], on=key, validate="many_to_one")
    require(len(traced) == len(trace) and traced.target.eq(traced.target_class_id).all(), "Stability target mismatch")
    trace_numeric = trace.assign(score=trace.score.astype(float), valid=trace.status.eq("valid").astype(float))
    require(np.isfinite(trace_numeric.score).all(), "Nonfinite stability score")
    stability = trace_numeric.groupby(trace_key).agg(stability_spearman=("score", "mean"), stability_valid_rate=("valid", "mean"))
    for metric in ("stability_spearman", "stability_valid_rate"):
        expected = per[per.metric == metric].set_index(trace_key).value.astype(float).sort_index()
        actual = stability[metric].sort_index()
        require(expected.index.equals(actual.index) and np.allclose(expected, actual, rtol=1e-10, atol=1e-10), f"Trace aggregate mismatch: {metric}")
    require(trace.status.ne("valid").sum() == 28, "Unexpected invalid repeat count")

    output.mkdir(parents=True)
    (output / "data").mkdir()
    for name, frame in data.items():
        frame.to_csv(output / "data" / (name + ".csv"), index=False)
    method_predictions.to_csv(output / "data/method_predictions.csv", index=False)
    prediction_context.to_csv(output / "data/predictions.csv", index=False)
    cohorts.to_csv(output / "data/cohorts.csv", index=False)
    counts.to_csv(output / "data/cohort_counts.csv", index=False)
    subset_counts = {}
    for cohort in ("model_correct", "common_correct"):
        selected = tagged.loc[tagged[cohort] == 1, per.columns]
        selected.to_csv(output / "data" / f"per_image_{cohort}.csv", index=False)
        aggregate_per_image(selected, units).to_csv(output / "data" / f"units_{cohort}.csv", index=False)
        subset_counts[cohort] = len(selected)
    overview = units.pivot(index=["method", "model", "dataset"], columns="metric", values="mean").reset_index()
    for metric in metrics:
        overview[metric] = overview[metric].astype(float)
    overview["n"] = 460
    overview["run_id"] = np.where(overview.method == "lime", "lime-vectorized-8000-20261014", "full-five-20261013")
    overview.to_csv(output / "data/unit_overview.csv", index=False)

    def copy(source: Path, relative: str) -> None:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    for label, root in roots.items():
        shutil.copytree(root / "release", output / "source" / label / "release")
        for name in ("COMPLETE.json", "manifest.json", "plan.json"):
            if (root / "run" / name).exists():
                copy(root / "run" / name, f"evidence/{label}/{name}")
        for unit in sorted((root / "run/formal").iterdir()):
            if not unit.is_dir():
                continue
            for name in ("unit.yaml", "protocol.yaml", "completion.json"):
                copy(unit / name, f"configs/{label}/{unit.name}/{name}")
            shutil.copytree(unit / "configs", output / "configs" / label / unit.name / "configs")
            shutil.copytree(unit / "stability_configs", output / "configs" / label / unit.name / "stability_configs")
    shutil.copytree(lime / "overlay", output / "source/lime/overlay")
    copy(five / "run/final_audit.json", "evidence/five/final_audit.json")
    copy(lime / "independent_audit.json", "evidence/lime/independent_audit.json")
    copy(lime / "server_asset_recheck.json", "evidence/lime/server_asset_recheck.json")
    for unit in (lime / "run/units").iterdir():
        copy(unit / "completion.json", f"evidence/lime/logical_units/{unit.name}.json")
    for name in ("DCU_FULL_RESULTS_20261013.md", "LIME_DCU_RESULTS_20261013.md", "LIME_DCU_BUDGET_REVIEW_20261013.md", "STABILITY_PROTOCOL.md"):
        copy(repo / "docs" / name, "docs/" + name)
    for name in ("dcu_full_results_20261013.json", "lime_dcu_full_results_20261014.json", "lime_dcu_high_budget_audit_20261014.json", "lime_optimization_20261014.json"):
        copy(repo / "analysis" / name, "evidence/" + name)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(["git", "archive", "--format=zip", f"--output={output / 'source/repository.zip'}", "HEAD"], cwd=repo, check=True)
    copy(Path(__file__), "source/prepare_dcu_team_handoff.py")
    audit = {
        "status": "verified",
        "repository_commit": commit,
        "rows": {name: len(frame) for name, frame in data.items()},
        "method_predictions": len(method_predictions), "prediction_contexts": len(prediction_context),
        "prediction_identity_exact_across_six_methods": True,
        "confidence_float32_equal": True, "confidence_max_abs_difference": confidence_delta,
        "shared_prediction_confidence_reference_method": "ig",
        "cohort_counts": counts.to_dict("records"), "subset_per_image_rows": subset_counts,
        "noise_seed_pairs_verified": True, "trace_status_counts": trace.status.value_counts().to_dict(),
        "unit_mean_std_recomputed": True, "trace_means_and_valid_rates_recomputed": True,
        "source_csv_sha256": sources,
        "scope": "Handoff CSV identities, source hashes, paired predictions, cohorts and aggregation checked; prior complete NPY audits retained, no new GPU run.",
    }
    (output / "evidence/handoff_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (output / "evidence/workbook_input.json").write_text(json.dumps({"overview": overview.to_dict("records"), "cohorts": counts.to_dict("records")}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
