"""Independently read the DCU six-method handoff as data, never execute it."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

ROOT = "team-handoff-20261014/"
EXPECTED_SHA = "14b6ea625f41e6c7c0587fd0073fc70e2064df0260107dca042335c6fc2303dc"
METHODS = ("ig", "gradcam", "occlusion", "rise", "kernelshap", "lime")
MODELS = ("resnet50", "densenet121", "vgg16")
DATASETS = ("imagenet", "voc")
METRICS = ("faithfulness_morf_auc_raw", "stability_spearman", "efficiency_time_ms")
ALL_METRICS = METRICS + ("stability_valid_rate",)
KEY = ["image_id", "dataset", "model", "method"]
UNIT = ["dataset", "model", "method", "metric"]
PRED_KEY = ["image_id", "dataset", "model"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def object_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode())[:12]


def check_config_snapshots(z, trace, units, predictions):
    """Bind trace hashes to inert YAML snapshots, including all LIME shards."""
    snapshots = [n for n in z.namelist() if n.startswith(ROOT + "configs/")
                 and "/stability_configs/" in n and n.endswith(".yaml")]
    require(len(snapshots) == 90, "30 five-method units + 60 LIME shards required")
    seen = set()
    semantic_protocol = None
    for name in snapshots:
        execution = yaml.safe_load(z.read(name))
        # The runner adds these two locator fields AFTER computing identity.
        identity = {k: v for k, v in execution.items()
                    if k not in ("base_config_path", "stability_protocol_path")}
        digest = object_hash(identity)
        require(PurePosixPath(name).name == digest + "_stability_unit.yaml", "stability config hash")
        base = execution["base_config"]
        base_hash = object_hash(base)
        directory = name.split("/stability_configs/")[0]
        saved_base = yaml.safe_load(z.read(directory + "/configs/" + base_hash + "_unit.yaml"))
        require(base == saved_base and execution["base_config_hash"] == base_hash, "base config binding")
        method, dataset, model = base["method"], base["dataset"]["name"], base["model"]["name"]
        selected = trace[trace.method.eq(method) & trace.dataset.eq(dataset)
                         & trace.model.eq(model) & trace.config_hash.eq(digest)]
        ids = set(selected.image_id)
        configured_ids = base["dataset"].get("image_ids")
        require(len(ids) == (46 if method == "lime" else 460)
                and (configured_ids is None or ids == set(configured_ids)), "shard image IDs")
        seen.add(digest)
        context = execution["prediction_context"]
        predicted = predictions[predictions.dataset.eq(dataset) & predictions.model.eq(model)
                                & predictions.method.eq(method)]
        require(predicted.checkpoint_sha256.eq(context["checkpoint_sha256"]).all()
                and predicted.config_hash.eq(context["config_hash"]).all(), "prediction context binding")
        semantic = {k: v for k, v in execution["stability"].items() if k not in ("runtime", "output")}
        if semantic_protocol is None:
            semantic_protocol = semantic
        require(semantic == semantic_protocol, "inconsistent stability semantics")
        require(base["dataset"]["split"] == "eval"
                and base["metrics"]["deletion_fractions"] == [i / 20 for i in range(21)], "eval/MoRF grid")
        require(base["model"]["output_activation"] == ("softmax" if dataset == "imagenet" else "sigmoid"), "activation")
        completion = json.loads(z.read(directory + "/completion.json"))
        require(completion["status"] == "verified" and completion["images"] == len(ids)
                and completion["trace_rows"] == len(ids) * 5, "completion evidence counts")
        if method == "lime":
            require(base["attribution"]["num_samples"] == 8000
                    and base["attribution"]["batch_size"] == 32
                    and base["runtime"]["implementation"] == "VectorizedLIME", "LIME budget/implementation")
        else:
            rows = units[units.method.eq(method) & units.dataset.eq(dataset) & units.model.eq(model)]
            require(rows.loc[rows.metric.str.startswith("stability_"), "config_hash"].eq(digest).all()
                    and rows.loc[~rows.metric.str.startswith("stability_"), "config_hash"].eq(base_hash).all(), "unit config hashes")
    require(seen == set(trace.config_hash), "unbound trace configuration")
    return len(snapshots), object_hash(semantic_protocol)


def check_aggregates(frame, summary):
    numeric = frame.copy()
    numeric["value"] = pd.to_numeric(numeric["value"], errors="raise")
    require(not numeric.duplicated(KEY + ["metric"]).any(), "duplicate per-image key")
    require(not summary.duplicated(UNIT).any(), "duplicate summary key")
    require(np.isfinite(numeric.value).all(), "non-finite metric")
    calculated = numeric.groupby(UNIT)["value"].agg(["mean", "std", "count"]).sort_index()
    supplied = summary.set_index(UNIT).sort_index()
    require(calculated.index.equals(supplied.index), "summary coverage")
    for field in ("mean", "std"):
        require(np.allclose(calculated[field], pd.to_numeric(supplied[field]),
                            rtol=1e-10, atol=1e-12), f"summary {field} mismatch")
    require(np.array_equal(calculated["count"], pd.to_numeric(supplied.n)), "summary n mismatch")


def paired_seed(dataset, image_id, repeat):
    payload = f"{dataset.lower()}\0{image_id}\0{repeat}".encode()
    return str(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1))


def validate(path: Path, project: Path, expected_sha=EXPECTED_SHA):
    raw_sha = sha256(path.read_bytes())
    require(expected_sha is None or raw_sha == expected_sha, "handoff ZIP SHA mismatch")
    with ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), "duplicate ZIP entries")
        require(all(n.startswith(ROOT) and ".." not in PurePosixPath(n).parts
                    and "\\" not in n and ":" not in n for n in names), "unsafe ZIP names")
        require(sum(i.file_size for i in z.infolist()) < 100_000_000
                and all(i.file_size < 50_000_000 for i in z.infolist()), "oversized handoff")
        require(z.testzip() is None, "ZIP CRC failure")
        manifest = json.loads(z.read(ROOT + "manifest.json"))
        for name, spec in manifest["files"].items():
            contents = z.read(ROOT + name)
            require(len(contents) == spec["bytes"] and sha256(contents) == spec["sha256"],
                    f"manifest hash/size: {name}")
        sums = {}
        for line in z.read(ROOT + "SHA256SUMS").decode().splitlines():
            digest, name = line.split(None, 1)
            name = name.lstrip(" *")
            require(name not in sums, "duplicate checksum entry")
            sums[name] = digest
            require(sha256(z.read(ROOT + name)) == digest, f"SHA256SUMS: {name}")
        expected_names = {ROOT + n for n in manifest["files"]} | {ROOT + "manifest.json", ROOT + "SHA256SUMS"}
        require(set(names) == expected_names, "unlisted/missing ZIP payload")

        def read(name):
            return pd.read_csv(io.BytesIO(z.read(ROOT + "data/" + name)), dtype=str,
                               keep_default_na=False)

        per, units, trace = read("per_image.csv"), read("units.csv"), read("stability_trace.csv")
        predictions, cohorts = read("predictions.csv"), read("cohorts.csv")
        require((len(per), len(units), len(trace)) == (66240, 144, 82800), "formal counts")
        expected_units = set(itertools.product(DATASETS, MODELS, METHODS))
        require(set(map(tuple, per[["dataset", "model", "method"]].drop_duplicates().values))
                == expected_units and set(per.metric) == set(ALL_METRICS), "36-unit coverage")
        require(per.groupby(UNIT).size().eq(460).all(), "per-metric image count")
        per["value"] = pd.to_numeric(per["value"], errors="raise")
        check_aggregates(per, units)
        for metric in ("faithfulness_morf_auc_raw", "stability_valid_rate"):
            require(per.loc[per.metric.eq(metric), "value"].between(0, 1).all(), "metric range")
        require(per.loc[per.metric.eq("stability_spearman"), "value"].between(-1, 1).all()
                and per.loc[per.metric.eq("efficiency_time_ms"), "value"].ge(0).all(), "score/time range")
        require(not trace.duplicated(KEY + ["repeat"]).any()
                and trace.groupby(KEY).size().eq(5).all(), "trace uniqueness/count")
        require(trace.groupby(KEY)["repeat"].agg(set).map(lambda v: v == set("01234")).all(), "repeat IDs")
        require(trace.protocol_version.eq("rgb-gaussian-spearman-v1").all()
                and pd.to_numeric(trace.sigma).eq(.005).all(), "protocol/sigma")
        seeds = trace[["dataset", "image_id", "repeat", "seed"]].drop_duplicates()
        require(not seeds.duplicated(["dataset", "image_id", "repeat"]).any(), "unpaired seeds")
        require(all(r.seed == paired_seed(r.dataset, r.image_id, int(r.repeat))
                    for r in seeds.itertuples()), "seed derivation mismatch")
        trace["score"] = pd.to_numeric(trace.score, errors="raise")
        trace["valid"] = trace.status.eq("valid").astype(float)
        allowed = {"valid", "both_constant", "candidate_constant", "reference_constant"}
        require(set(trace.status) <= allowed, "unaccounted trace statuses")
        require(np.isfinite(trace.score).all() and trace.score.between(-1, 1).all(), "trace score range")
        bad = trace.valid.eq(0)
        require(trace.loc[bad, "score"].eq(0).all() and trace.loc[bad, "spearman"].eq("").all(), "degenerate handling")
        require(np.allclose(trace.loc[~bad, "score"], pd.to_numeric(trace.loc[~bad, "spearman"]),
                            rtol=1e-12, atol=1e-14), "valid score/Spearman")
        for metric, column in [("stability_spearman", "score"), ("stability_valid_rate", "valid")]:
            calc = trace.groupby(KEY)[column].mean().sort_index()
            given = per[per.metric.eq(metric)].set_index(KEY).value.sort_index()
            require(calc.index.equals(given.index) and np.allclose(calc, given, rtol=1e-10, atol=1e-12),
                    f"trace-to-image {metric}")
        require(len(predictions) == len(cohorts) == 2760 and predictions.split.eq("eval").all(), "prediction/cohort counts")
        require(not predictions.duplicated(PRED_KEY).any() and not cohorts.duplicated(PRED_KEY).any(), "prediction/cohort duplicate")
        meta_bytes = z.read(ROOT + "source/five/release/data/metadata.csv")
        version = json.loads(z.read(ROOT + "source/five/release/data/DATA_VERSION.json"))
        require(sha256(meta_bytes) == sha256((project / "data/metadata.csv").read_bytes())
                == version["sha256"]["metadata.csv"], "frozen metadata SHA")
        metadata = pd.read_csv(io.BytesIO(meta_bytes), dtype=str, keep_default_na=False)
        frozen_ids = {d: set(metadata.loc[metadata.dataset.eq(d) & metadata.split.eq("eval"), "image_id"])
                      for d in DATASETS}
        for (dataset, model, method, metric), group in per.groupby(UNIT):
            require(set(group.image_id) == frozen_ids[dataset], "frozen eval sample coverage")
        target_table = predictions.merge(metadata[["image_id", "dataset", "class_id", "split"]],
            on=["image_id", "dataset"], validate="many_to_one")
        require(len(target_table) == len(predictions) and target_table.split_y.eq("eval").all()
                and np.array_equal(pd.to_numeric(target_table.target_class_id), pd.to_numeric(target_table.class_id)), "metadata targets")
        target_check = trace.merge(predictions[PRED_KEY + ["target_class_id"]], on=PRED_KEY, validate="many_to_one")
        require(len(target_check) == len(trace) and np.array_equal(pd.to_numeric(target_check.target), pd.to_numeric(target_check.target_class_id)), "trace targets")
        correct = predictions.target_class_id.eq(predictions.predicted_class_id).astype(int)
        require(np.array_equal(correct, pd.to_numeric(predictions.correct)), "correct flag")
        expected = predictions[PRED_KEY].assign(model_correct=correct)
        common = expected.groupby(["image_id", "dataset"]).model_correct.min()
        expected = expected.join(common.rename("common_correct"), on=["image_id", "dataset"]).set_index(PRED_KEY).sort_index()
        supplied = cohorts.set_index(PRED_KEY).sort_index()
        require(expected.index.equals(supplied.index), "cohort identity")
        frames = {"all": per}
        for flag in ("model_correct", "common_correct"):
            require(np.array_equal(expected[flag], pd.to_numeric(supplied[flag])), f"cohort {flag}")
            subset = read(f"per_image_{flag}.csv")
            subset["value"] = pd.to_numeric(subset.value, errors="raise")
            chosen = per.merge(cohorts[PRED_KEY + [flag]], on=PRED_KEY, validate="many_to_one")
            chosen = chosen[chosen[flag].eq("1")].set_index(KEY + ["metric"]).value.sort_index()
            actual = subset.set_index(KEY + ["metric"]).value.sort_index()
            require(chosen.index.equals(actual.index)
                    and np.allclose(chosen, actual, rtol=1e-12, atol=1e-14), f"subset values: {flag}")
            check_aggregates(subset, read(f"units_{flag}.csv"))
            frames[flag] = subset
        method_predictions = read("method_predictions.csv")
        require(len(method_predictions) == 16560 and not method_predictions.duplicated(KEY).any(), "method prediction counts")
        compare = method_predictions.merge(predictions, on=PRED_KEY, suffixes=("_method", "_shared"), validate="many_to_one")
        require(len(compare) == len(method_predictions) and set(method_predictions.method) == set(METHODS), "method prediction coverage")
        for name in ("target_class_id", "predicted_class_id", "checkpoint_sha256", "weights"):
            require(compare[name + "_method"].eq(compare[name + "_shared"]).all(), f"method prediction {name}")
        config_count, semantic_hash = check_config_snapshots(z, trace, units, method_predictions)
        frozen_protocol = yaml.safe_load((project / "configs/stability_protocol_v1.yaml").read_text(encoding="utf-8"))
        require(semantic_hash == object_hash({k: v for k, v in frozen_protocol.items()
                                             if k not in ("runtime", "output")}), "frozen protocol semantics")
        count_table = read("cohort_counts.csv")
        for row in count_table.itertuples():
            selected = cohorts[cohorts.dataset.eq(row.dataset) & cohorts.model.eq(row.model)]
            require(len(selected) == int(row.eval_images)
                    and int(pd.to_numeric(selected.model_correct).sum()) == int(row.model_correct)
                    and int(pd.to_numeric(selected.common_correct).sum()) == int(row.common_correct), "cohort counts")
        receipt = dict(zip_sha256=raw_sha, source_repository_commit=manifest["repository_commit"],
            manifest_files_checked=len(manifest["files"]), checksum_entries_checked=len(sums),
            rows={"per_image": len(per), "units": len(units), "trace": len(trace),
                  "model_correct": len(frames["model_correct"]), "common_correct": len(frames["common_correct"])},
            units=36, paired_seed_entries=len(seeds), invalid_trace_count=int(bad.sum()),
            config_snapshots_checked=config_count, semantic_protocol_hash=semantic_hash,
            invalid_by_method=trace.loc[bad].groupby("method").size().to_dict(),
            cohort_counts=count_table.to_dict("records"),
            checks="ZIP CRC; all file hashes; frozen metadata/sample IDs; targets; paired seeds; config snapshots/shards; prediction/weight identities; trace aggregates; ddof=1 means/std/n; cohorts",
            limits="No delivered code executed, GPU recomputation, or proof of generation-time provenance. Noise maps are not archived.")
        classes = metadata[metadata.split.eq("eval")].set_index(["dataset", "image_id"]).class_id
        return frames, receipt, classes
