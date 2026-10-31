"""Recheck B's M5 evidence and select existing observations; never load a model."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yaml

from analysis.validate_dcu_team_handoff import validate, EXPECTED_SHA

ROOT = Path(__file__).resolve().parents[1]
NEW_SHA = "a1ae2e245e3a96fd2790c4d61a3bbb8c61c5f25bbc5ee2d06f10436f19c2894a"
E_HEAD = "6b4c4ec980fd3ea354a46851fa052877f21255ee"
METHODS = ["ig", "gradcam", "occlusion", "rise", "kernelshap", "lime"]
METRICS = ["faithfulness_morf_auc_raw", "stability_spearman",
           "stability_valid_rate", "efficiency_time_ms"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(ref: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)


def read_csv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)


def select_cases(per: pd.DataFrame, predictions: pd.DataFrame,
                 trace: pd.DataFrame, cohorts: pd.DataFrame):
    """E's three VOC/ResNet candidates plus a VOC hit-group degeneration example."""
    cohort = cohorts[(cohorts.dataset == "voc") & (cohorts.model == "resnet50")
                     & (cohorts.model_correct == "1")]
    ids = set(cohort.image_id)
    rows = per[(per.dataset == "voc") & (per.model == "resnet50") & per.image_id.isin(ids)]
    values = rows.assign(value=pd.to_numeric(rows.value)).pivot(
        index="image_id", columns=["metric", "method"], values="value")
    if values.isna().any().any() or len(rows) != len(ids) * 24:
        raise ValueError("incomplete six-method candidate matrix")
    ranks = pd.concat({m: values[m].rank(axis=1, method="average",
                         ascending=m in (METRICS[0], METRICS[3])) for m in METRICS}, axis=1)
    pred = predictions[(predictions.dataset == "voc") & (predictions.model == "resnet50")]
    score = pd.DataFrame({
        "central_distance": (ranks - ranks.median()).abs().sum(axis=1),
        "rank_discordance": ((ranks[METRICS[0]] - ranks[METRICS[1]]) ** 2).sum(axis=1),
        "confidence": pd.to_numeric(pred.set_index("image_id").confidence),
    }).loc[values.index]
    score["numeric_id"] = score.index.astype(int)
    rules = [
        ("central", ["central_distance", "confidence", "numeric_id"], [True, False, True]),
        ("tradeoff", ["rank_discordance", "numeric_id"], [False, True]),
        ("low_confidence", ["confidence", "numeric_id"], [True, True]),
    ]
    chosen, used = [], set()
    for label, columns, ascending in rules:
        order = score.sort_values(columns, ascending=ascending)
        image_id = next(i for i in order.index if i not in used)
        used.add(image_id)
        chosen.append(dict(role=label, dataset="voc", model="resnet50", image_id=image_id,
                           **score.loc[image_id].to_dict()))
    invalid = trace[(trace.dataset == "voc") & (trace.method == "gradcam")
                    & trace.status.isin(["both_constant", "candidate_constant", "reference_constant"])]
    bad = invalid.groupby(["dataset", "model", "image_id"]).size().rename("invalid_repeats").reset_index()
    bad = bad.merge(cohorts, on=["dataset", "model", "image_id"], validate="one_to_one")
    bad = bad[bad.model_correct == "1"].assign(numeric_id=lambda x: x.image_id.astype(int))
    if not bad.empty:
        row = bad.sort_values(["invalid_repeats", "model", "numeric_id"], ascending=[False, True, True]).iloc[0]
        chosen.append(dict(role="constant_gradcam", dataset=row.dataset, model=row.model,
                           image_id=row.image_id, invalid_repeats=int(row.invalid_repeats)))
    return chosen, score.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--analysis-handoff", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Existing data directory; read only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frames, receipt, _ = validate(args.handoff, ROOT)
    checkpoint_manifest = json.loads((ROOT / "docs/VOC20_CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8"))
    training_hash_checks = []
    for spec in checkpoint_manifest["shared_data"].values():
        data = (ROOT / spec["path"]).read_bytes()
        crlf = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        assert sha(crlf) == spec["sha256"], spec["path"]
        training_hash_checks.append(dict(path=spec["path"], expected=spec["sha256"],
            checkout_raw_sha256=sha(data), raw_match=sha(data) == spec["sha256"],
            crlf_representation_matches=True))
    train = read_csv((ROOT / "data/voc20_multilabel_train.csv").read_bytes())
    val = read_csv((ROOT / "data/voc20_multilabel_validation.csv").read_bytes())
    meta = read_csv((ROOT / "data/metadata.csv").read_bytes())
    train_ids, val_ids = set(train.image_id.astype(int)), set(val.image_id.astype(int))
    frozen_ids = set(meta.loc[meta.dataset == "voc", "image_id"].astype(int))
    assert len(train) == 4037 and len(val) == 474 and len(frozen_ids) == 500
    assert not (train_ids & val_ids or train_ids & frozen_ids or val_ids & frozen_ids)
    oldroot, newroot = "team-handoff-20261014/", "dcu_analysis_handoff_20261026/"
    assert sha(args.analysis_handoff.read_bytes()) == NEW_SHA
    with ZipFile(args.handoff) as old, ZipFile(args.analysis_handoff) as new:
        assert new.testzip() is None
        manifest = json.loads(new.read(newroot + "manifest.json"))
        for path, spec in manifest["files"].items():
            data = new.read(newroot + path)
            assert len(data) == spec["bytes"] and sha(data) == spec["sha256"], path
        sums = new.read(newroot + "SHA256SUMS").decode().splitlines()
        for line in sums:
            digest, path = line.split(None, 1)
            assert sha(new.read(newroot + path.lstrip("*"))) == digest, path
        assert sha(new.read(newroot + "archives/dcu_team_handoff_20261014.zip")) == EXPECTED_SHA
        original_members = [n for n in old.namelist() if not n.endswith("/")]
        for name in original_members:
            assert old.read(name) == new.read(newroot + "analysis_input/" + name[len(oldroot):]), name
        catalog_bytes = new.read(newroot + "indexes/full_package_asset_catalog.csv")
        catalog = read_csv(catalog_bytes)
        key = ["dataset", "model", "method", "image_id"]
        assert len(catalog) == 16560 and not catalog.duplicated(key).any()
        per = read_csv(old.read(oldroot + "data/per_image.csv"))
        predictions = read_csv(old.read(oldroot + "data/predictions.csv"))
        method_predictions = read_csv(old.read(oldroot + "data/method_predictions.csv"))
        cohorts = read_csv(old.read(oldroot + "data/cohorts.csv"))
        trace = read_csv(old.read(oldroot + "data/stability_trace.csv"))
        joined = catalog.merge(method_predictions, on=key, suffixes=("", "_pred"), validate="one_to_one")
        assert len(joined) == 16560
        for left, right in [("target", "target_class_id"), ("checkpoint_sha256", "checkpoint_sha256_pred"),
                            ("predicted_class_id", "predicted_class_id_pred")]:
            assert joined[left].equals(joined[right]), (left, right)
        flags = catalog.merge(cohorts, on=["dataset", "model", "image_id"], suffixes=("", "_cohort"), validate="many_to_one")
        for flag in ("model_correct", "common_correct"):
            assert flags[flag].equals(flags[flag + "_cohort"])
        for metric in METRICS:
            given = per[per.metric == metric].set_index(key).value.astype(float).sort_index()
            actual = catalog.set_index(key)[metric].astype(float).sort_index()
            assert given.index.equals(actual.index) and np.allclose(given, actual, rtol=1e-12, atol=1e-14)
        for path, digest in catalog[["config_path", "config_sha256"]].drop_duplicates().itertuples(index=False):
            assert sha(new.read(newroot + path)) == digest
        budgets = dict(ig=dict(n_steps=50, internal_batch_size=10, approximation_method="gausslegendre"),
            occlusion=dict(window_height=32, window_width=32, stride_height=16, stride_width=16, perturbations_per_eval=16),
            rise=dict(num_masks=4000, mask_size=7, mask_probability=.5, batch_size=32, seed=42),
            kernelshap=dict(n_samples=2048, perturbations_per_eval=16, feature_grid_size=7, seed=42),
            lime=dict(num_samples=8000, batch_size=32, seed=42, segmentation_seed=42,
                      kernel_width=.25, ridge_alpha=1., feature_selection="none"))
        configs = [n for n in old.namelist() if n.startswith(oldroot + "configs/")
                   and "/stability_configs/" not in n and n.endswith("_unit.yaml")]
        assert len(configs) == 90
        for path in configs:
            config = yaml.safe_load(old.read(path))
            assert config["runtime"]["warmup_runs"] == 1
            assert config["model"]["num_classes"] == (1000 if config["dataset"]["name"] == "imagenet" else 20)
            if config["method"] == "gradcam":
                expected_layer = dict(resnet50="layer4.2", densenet121="features.norm5", vgg16="features.28")
                assert config["attribution"]["target_layer"] == expected_layer[config["model"]["name"]]
            else:
                assert all(config["attribution"][k] == v for k, v in budgets[config["method"]].items())
        selected, scores = select_cases(per, predictions, trace, cohorts)
        scores.to_csv(args.output_dir / "selection_scores.csv", index=False, lineterminator="\n")
        chosen = pd.DataFrame(selected)
        candidate = catalog.merge(chosen[["dataset", "model", "image_id", "role"]],
                                  on=["dataset", "model", "image_id"], validate="many_to_one")
        assert len(candidate) == 6 * len(selected)
        ebase = "deliverables/e_m5/demo/"
        e = json.loads(git_bytes(E_HEAD, ebase + "cases.json").decode("utf-8-sig"))
        visual_receipts = []
        for row in candidate.to_dict("records"):
            identity = next((c for c in e["cases"] if c["image_id"] == row["image_id"]
                             and c["model"] == row["model"] and c["dataset"] == row["dataset"]), None)
            checked = False
            if identity:
                method = next(m for m in identity["methods"] if m["method"] == row["method"])
                assert method["map_npy_sha256"] == row["npy_sha256"]
                assert method["map_png_sha256"] == row["png_sha256"]
                assert method["map_provenance_sha256"] == row["provenance_sha256"]
                png = git_bytes(E_HEAD, ebase + method["map_png"])
                sidecar = git_bytes(E_HEAD, ebase + method["map_provenance_sidecar"])
                assert sha(png) == row["png_sha256"] and sha(sidecar) == row["provenance_sha256"]
                from PIL import Image
                with Image.open(io.BytesIO(png)) as image:
                    assert image.size == (224, 224) and image.mode == "L"
                provenance = json.loads(sidecar)
                assert provenance["map_sha256"] == row["npy_sha256"]
                for field in ("dataset", "model", "method", "image_id", "split", "checkpoint_sha256", "base_config_hash", "prediction_context_hash"):
                    assert str(provenance[field]) == row[field], field
                checked = True
            image_checked = False
            if args.image_root:
                path = args.image_root / row["image_path"].removeprefix("dataset/")
                if path.is_file():
                    assert sha(path.read_bytes()) == row["image_sha256"]
                    image_checked = True
            visual_receipts.append(dict(dataset=row["dataset"], model=row["model"], method=row["method"],
                image_id=row["image_id"], png_and_sidecar_bytes_verified=checked,
                original_bytes_verified=image_checked, npy_bytes_verified=False))
        candidate["npy_status"] = "indexed_hash_only_bytes_not_received"
        candidate["png_status"] = ["E_git_preview_verified" if r["png_and_sidecar_bytes_verified"]
                                   else "not_received" for r in visual_receipts]
        candidate.to_csv(args.output_dir / "case_candidates.csv", index=False, lineterminator="\n")
        source_rows = []
        sources = [n for n in old.namelist() if n.startswith(oldroot + "source/five/release/")
                   and n.endswith(tuple(["/" + p for p in ["factory.py", "dataset.py", "run_unit.py",
                       "run_stability.py", "metrics.py", "stability.py", "integrated_gradients.py",
                       "grad_cam.py", "occlusion.py", "rise.py", "kernelshap.py", "gradient_common.py"]]))]
        sources += [oldroot + "source/lime/overlay/" + p for p in
                    ["lime_vectorized.py", "run_lime_vectorized_full.py", "overlay-manifest.json"]]
        for name in sources:
            data = old.read(name)
            source_rows.append(dict(member=name[len(oldroot):], sha256=sha(data), bytes=len(data)))
        pd.DataFrame(source_rows).to_csv(args.output_dir / "frozen_source_inventory.csv", index=False, lineterminator="\n")
        stats = json.loads((ROOT / "analysis/dcu_20261015/statistics.json").read_text(encoding="utf-8"))
        exceptions = [r for r in stats["method_pairs"] if r["dataset"] == "imagenet"
                      and r["model"] == "densenet121" and r["a"] == "ig" and r["b"] == "kernelshap"
                      and r["metric"] == METRICS[0]]
        for row in exceptions:
            values = frames[row["cohort"]]
            values = values[(values.dataset == "imagenet") & (values.model == "densenet121") & (values.metric == METRICS[0])]
            pivot = values.pivot(index="image_id", columns="method", values="value")
            advantage = pivot.kernelshap - pivot.ig
            assert len(advantage) == row["n"]
            assert np.isclose(advantage.mean(), row["mean_advantage_a"], atol=1e-14)
            assert int((advantage.round(12) > 0).sum()) == row["wins_a"]
        pd.DataFrame(exceptions).to_csv(args.output_dir / "ig_kernelshap_exception.csv", index=False, lineterminator="\n")
        evidence = dict(formal_input=receipt, supplemental_input=dict(sha256=NEW_SHA,
            manifest_files_checked=len(manifest["files"]), checksum_entries_checked=len(sums),
            original_members_identical=len(original_members), scope=manifest["scope"],
            asset_catalog_sha256=sha(catalog_bytes), catalog_rows_crosschecked=len(catalog)),
            selection=dict(status="B_reproducible_proposal_pending_E_confirmation", candidates=selected,
                           candidate_pool=387),
            previews=visual_receipts, e_head=subprocess.check_output(["git", "rev-parse", E_HEAD], cwd=ROOT, text=True).strip(),
            voc_isolation=dict(train_count=len(train), validation_count=len(val), frozen_count=len(frozen_ids),
                pairwise_disjoint=True, training_manifest_checks=training_hash_checks,
                checkpoint_binary_rehash_performed=False),
            formal_budget_snapshots_checked=len(configs),
            boundaries="No model, GPU, attribution, bootstrap, permutation or noise-map regeneration. NPY hashes are catalog claims, not local byte verification.")
        (args.output_dir / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(dict(selected=selected, candidate_rows=len(candidate), previews_verified=sum(
            r["png_and_sidecar_bytes_verified"] for r in visual_receipts)), ensure_ascii=False))


if __name__ == "__main__":
    main()
