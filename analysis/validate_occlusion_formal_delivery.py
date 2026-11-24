"""Independently validate B's received six-unit formal Occlusion stability ZIP.

This reads the archive in place, never executes included code, and compares its
base metrics and reference maps with D's previously accepted base handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd


UNITS = ("resnet50_imagenet", "resnet50_voc", "densenet121_imagenet",
         "densenet121_voc", "vgg16_imagenet", "vgg16_voc")
BASE_METRICS = {"efficiency_time_ms", "faithfulness_morf_auc_raw"}
STABILITY_METRICS = {"stability_spearman", "stability_valid_rate"}
PROTOCOL_SHA256 = "324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(archive: ZipFile, path: str) -> dict:
    return json.loads(archive.read(path))


def read_csv(archive: ZipFile, path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(path)), dtype={"image_id": str})


def seed(dataset: str, image_id: str, repeat: int) -> int:
    payload = f"{dataset.lower()}\0{image_id}\0{repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def close(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12))


def validate(zip_path: Path, base_light: Path, base_float32: Path) -> dict:
    require(base_light.is_dir() and base_float32.is_dir(), "accepted base handoff roots missing")
    with ZipFile(zip_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "duplicate ZIP paths")
        require(all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
                    and "\\" not in name for name in names), "unsafe ZIP path")
        file_manifest = read_json(archive, "file_manifest.json")
        require(file_manifest.get("kind") == "formal_occlusion_stability_light_handoff", "wrong package kind")
        entries = file_manifest["files"]
        listed = [item["path"] for item in entries]
        require(len(listed) == len(set(listed)) and set(listed) == set(names) - {"file_manifest.json"},
                "package file set differs from manifest")
        for item in entries:
            contents = archive.read(item["path"])
            require(len(contents) == item["bytes"] and hashlib.sha256(contents).hexdigest() == item["sha256"],
                    f"package hash mismatch: {item['path']}")

        manifest = read_json(archive, "docs/B_OCCLUSION_STABILITY_EVAL_MANIFEST.json")
        require(manifest["frozen_protocol"]["git_blob_sha256"] == PROTOCOL_SHA256, "frozen protocol mismatch")
        protocol = archive.read("configs/stability_protocol_v1.yaml").replace(b"\r\n", b"\n")
        require(hashlib.sha256(protocol).hexdigest() == PROTOCOL_SHA256, "protocol bytes mismatch")
        require(manifest["counts"]["units"] == 6, "manifest unit count mismatch")
        manifest_units = {item["unit"]: item for item in manifest["units"]}
        require(set(manifest_units) == set(UNITS), "manifest unit set mismatch")

        paired: dict[tuple[str, str, int], dict[str, int]] = {}
        results = []
        map_count = 0
        for unit in UNITS:
            model, dataset = unit.rsplit("_", 1)
            expected = manifest_units[unit]
            base = f"results/occlusion_eval/{unit}/"
            stability = f"results/occlusion_stability_eval/{unit}/"
            report = read_json(archive, stability + "validation_report.json")
            require(report["status"] == "passed" and report["split"] == "eval" and report["unit"] == unit,
                    f"failed or wrong unit report: {unit}")
            require(report["images"] == 460 and report["trace_rows"] == 2300
                    and report["stability_metric_rows"] == 920, f"report counts: {unit}")
            require(report["config_hash"] == expected["stability_config_hash"]
                    and report["base_config_hash"] == expected["base_config_hash"]
                    and report["checkpoint_sha256"] == expected["checkpoint_sha256"], f"hash binding: {unit}")
            require(report["status_counts"] == {"valid": 2300} and report["degenerate_repeats"] == 0,
                    f"status counts: {unit}")
            run_log = [json.loads(line) for line in archive.read(stability + "run_log.jsonl").splitlines()]
            require(len(run_log) >= 2 and run_log[0]["processed"] == 460 and run_log[0]["skipped"] == 0
                    and run_log[-1]["processed"] == 0 and run_log[-1]["skipped"] == 460,
                    f"immediate resume: {unit}")
            require(all(row["config_hash"] == expected["stability_config_hash"] and row["split"] == "eval"
                        for row in run_log), f"run-log hashes: {unit}")

            predictions = read_csv(archive, base + "predictions.csv")
            per_image = read_csv(archive, base + "per_image.csv")
            units = read_csv(archive, base + "units.csv")
            require(len(predictions) == 460 and predictions.image_id.is_unique, f"predictions: {unit}")
            ids = set(predictions.image_id)
            require(len(per_image) == 1840 and not per_image.duplicated(["image_id", "metric"]).any()
                    and set(per_image.metric) == BASE_METRICS | STABILITY_METRICS, f"per-image keys: {unit}")
            require(all(set(per_image.loc[per_image.metric == metric, "image_id"]) == ids
                        and len(per_image.loc[per_image.metric == metric]) == 460
                        for metric in BASE_METRICS | STABILITY_METRICS), f"metric image sets: {unit}")
            require(len(units) == 4 and set(units.metric) == BASE_METRICS | STABILITY_METRICS,
                    f"unit metrics: {unit}")
            targets = dict(zip(predictions.image_id, predictions.target_class_id.astype(int)))
            require(set(predictions.checkpoint_sha256) == {expected["checkpoint_sha256"]},
                    f"prediction checkpoint: {unit}")

            # Existing accepted base outputs must be byte-identical at the row level.
            old_dir = base_light / base
            old_image = pd.read_csv(old_dir / "per_image.csv", dtype={"image_id": str})
            old_units = pd.read_csv(old_dir / "units.csv")
            old_predictions = pd.read_csv(old_dir / "predictions.csv", dtype={"image_id": str})
            pd.testing.assert_frame_equal(predictions, old_predictions, check_dtype=False)
            old_image = old_image.sort_values(["image_id", "metric"]).reset_index(drop=True)
            new_base = per_image.loc[per_image.metric.isin(BASE_METRICS)].sort_values(
                ["image_id", "metric"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(new_base, old_image, check_dtype=False)
            old_units = old_units.sort_values("metric").reset_index(drop=True)
            new_base_units = units.loc[units.metric.isin(BASE_METRICS)].sort_values("metric").reset_index(drop=True)
            pd.testing.assert_frame_equal(new_base_units, old_units, check_dtype=False)

            trace_names = [name for name in names if name.startswith(stability + "stability_trace_")
                           and name.endswith(".csv")]
            require(len(trace_names) == 1, f"trace file count: {unit}")
            traces = read_csv(archive, trace_names[0])
            require(len(traces) == 2300 and not traces.duplicated(["image_id", "repeat"]).any(),
                    f"trace keys: {unit}")
            require(set(traces.image_id) == ids and set(traces.repeat) == set(range(5))
                    and all(len(group) == 5 for _, group in traces.groupby("image_id")),
                    f"trace coverage: {unit}")
            require(set(traces.status) == {"valid"} and set(traces.protocol_version) == {"rgb-gaussian-spearman-v1"}
                    and set(traces.sigma) == {0.005} and set(traces.config_hash) == {expected["stability_config_hash"]},
                    f"trace protocol: {unit}")
            require(set(traces.dataset) == {dataset} and set(traces.model) == {model}
                    and set(traces.method) == {"occlusion"}, f"trace identity: {unit}")
            require(np.isfinite(traces[["score", "spearman", "attribution_time_ms"]]).all().all()
                    and traces.score.between(-1, 1).all()
                    and np.allclose(traces.score, traces.spearman, rtol=0, atol=1e-12),
                    f"trace scores: {unit}")
            for row in traces.itertuples(index=False):
                require(int(row.seed) == seed(dataset, row.image_id, int(row.repeat))
                        and int(row.target) == targets[row.image_id], f"paired seed or target: {unit}/{row.image_id}")
                paired.setdefault((dataset, row.image_id, int(row.repeat)), {})[model] = int(row.seed)

            indexed = per_image.set_index(["image_id", "metric"])["value"]
            for image_id, group in traces.groupby("image_id"):
                require(close(indexed.loc[(image_id, "stability_spearman")], group.score.mean())
                        and close(indexed.loc[(image_id, "stability_valid_rate")], 1.0),
                        f"per-image aggregate: {unit}/{image_id}")
            for row in units.itertuples(index=False):
                values = per_image.loc[per_image.metric == row.metric, "value"]
                require(row.n == 460 and close(row.mean, values.mean()) and close(row.std, values.std(ddof=1)),
                        f"unit aggregate: {unit}/{row.metric}")
                require(row.config_hash == (expected["stability_config_hash"] if row.metric in STABILITY_METRICS
                                            else expected["base_config_hash"]), f"unit config hash: {unit}/{row.metric}")

            sidecars = [name for name in names if name.startswith(base + "maps_float/")
                        and name.endswith(".npy.provenance.json")]
            require(len(sidecars) == 460, f"map provenance count: {unit}")
            for sidecar in sidecars:
                provenance = read_json(archive, sidecar)
                image_id = PurePosixPath(sidecar).name.removesuffix(".npy.provenance.json")
                map_file = base_float32 / sidecar.removesuffix(".provenance.json")
                require(map_file.is_file() and image_id in ids, f"missing accepted reference map: {unit}/{image_id}")
                require(provenance["binding"] == "verified_recomputation"
                        and provenance["map_sha256"] == hashlib.sha256(map_file.read_bytes()).hexdigest()
                        and provenance["base_config_hash"] == expected["base_config_hash"]
                        and provenance["checkpoint_sha256"] == expected["checkpoint_sha256"]
                        and provenance["dataset"] == dataset and provenance["model"] == model
                        and provenance["method"] == "occlusion" and provenance["split"] == "eval"
                        and provenance["image_id"] == image_id, f"map provenance: {unit}/{image_id}")
            map_count += len(sidecars)
            mean = float(per_image.loc[per_image.metric == "stability_spearman", "value"].mean())
            std = float(per_image.loc[per_image.metric == "stability_spearman", "value"].std(ddof=1))
            require(close(mean, expected["spearman_mean"]) and close(std, expected["spearman_sample_std"]),
                    f"handoff summary: {unit}")
            results.append({"unit": unit, "images": 460, "trace_rows": 2300,
                            "spearman_mean": mean, "spearman_sample_std": std,
                            "valid_rate": 1.0, "resume": "processed=0 skipped=460"})

        require(all(len(models) == 3 and len(set(models.values())) == 1 for models in paired.values()),
                "cross-model seed pairing")
        require(len(paired) == 4600 and map_count == 2760, "total pairing or maps count")
        return {"zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                "manifest_files": len(entries), "paired_dataset_image_repeat_keys": len(paired),
                "verified_reference_maps": map_count, "units": results,
                "provenance_limit": "verified_recomputation does not prove original generation-time weight identity"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--base-light", type=Path, default=Path("results/incoming/B_OCCLUSION_EVAL_LIGHT"))
    parser.add_argument("--base-float32", type=Path, default=Path("results/incoming/B_OCCLUSION_EVAL_FLOAT32"))
    args = parser.parse_args()
    print(json.dumps(validate(args.zip, args.base_light, args.base_float32), indent=2))


if __name__ == "__main__":
    main()
