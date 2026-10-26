"""Run isolated one-image gates, then resume and validate A's formal stability units."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.map_provenance import sha256_file  # noqa: E402
from experiments.run_stability import STABILITY_METRICS, _load_protocol, _unit_mask  # noqa: E402
from experiments.run_unit import UNIT_COLUMNS, _config_hash, _load_config, _resolve  # noqa: E402
from experiments.stability import stable_seed  # noqa: E402
from preprocessing.dataset import DATA_DIR, get_split  # noqa: E402

UNITS = tuple(f"{method}_{model}_{dataset}" for dataset in ("imagenet", "voc")
              for model in ("resnet50", "densenet121", "vgg16")
              for method in ("ig", "gradcam"))
PROTOCOL_SHA256 = "324315f1ecf62a807ec91e77b2cba1ea53593e9a7ad98a2a90cce09a9e32dcdf"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"image_id": str, "config_hash": str})


def identity(config: dict) -> dict[str, str]:
    return {"dataset": config["dataset"]["name"], "model": config["model"]["name"],
            "method": config["method"]}


def base_rows(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    keys = [column for column in ("dataset", "model", "method", "image_id", "metric")
            if column in frame.columns]
    return frame[~frame["metric"].isin(STABILITY_METRICS)].sort_values(keys).reset_index(drop=True)


def check_base_unchanged(before: dict[Path, pd.DataFrame]) -> None:
    for path, frame in before.items():
        pd.testing.assert_frame_equal(frame, base_rows(path), check_dtype=False,
                                      check_exact=False, rtol=1e-12, atol=1e-12)


def preflight(config: dict) -> tuple[dict[str, int], dict[str, str]]:
    unit = identity(config)
    if config["dataset"]["split"] != "eval" or config["runtime"]["max_images"] is not None:
        raise ValueError("A formal configs must use unrestricted eval")
    meta = get_split(unit["dataset"], "eval")
    expected = dict(zip(meta["image_id"].astype(str), meta["class_id"].astype(int)))
    if len(expected) != 460 or len(meta) != 460:
        raise ValueError("frozen eval must contain exactly 460 unique IDs")
    for relative in meta["image_path"]:
        path = Path(str(relative).replace("\\", "/"))
        path = path if path.is_absolute() else Path(DATA_DIR) / path
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen image: {path}")
    checkpoint = config["model"].get("checkpoint")
    if checkpoint:
        manifest = json.loads((ROOT / "docs/VOC20_CHECKPOINT_MANIFEST.json").read_text())
        record = next(item for item in manifest["checkpoints"] if item["model"] == unit["model"])
        path = _resolve(checkpoint)
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"checkpoint does not match frozen manifest: {path}")
    output = config["output"]
    rows = read_csv(_resolve(output["per_image_csv"]))
    rows = rows[_unit_mask(rows, unit) & rows["metric"].isin(config["metrics"]["names"])]
    for metric in config["metrics"]["names"]:
        selected = rows[rows["metric"].eq(metric)]
        if len(selected) != 460 or set(selected["image_id"]) != set(expected):
            raise ValueError(f"incomplete base metric: {unit}/{metric}")
        if not np.isfinite(pd.to_numeric(selected["value"])).all():
            raise ValueError(f"non-finite base metric: {unit}/{metric}")
    summaries = read_csv(_resolve(output["units_csv"]))
    summaries = summaries[_unit_mask(summaries, unit)
                          & summaries["metric"].isin(config["metrics"]["names"])]
    if (len(summaries) != len(config["metrics"]["names"])
            or set(summaries["metric"]) != set(config["metrics"]["names"])
            or set(summaries["config_hash"]) != {_config_hash(config)}
            or not summaries["n"].eq(460).all()):
        raise ValueError(f"base summary/config identity mismatch: {unit}")
    name = "_".join(unit[key] for key in ("method", "model", "dataset"))
    maps = _resolve(output["float_maps_dir"]) / name
    hashes = {}
    for image_id in expected:
        path = maps / f"{image_id}.npy"
        values = np.load(path, allow_pickle=False)
        if values.shape != (224, 224) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"invalid reference map: {path}")
        hashes[str(path)] = sha256_file(path)
    return expected, hashes


def prepare_gate(config: dict, protocol: dict, name: str, first_id: str) -> tuple[Path, Path]:
    gate = ROOT / "results/a_stability_gates" / name
    source_map = _resolve(config["output"]["float_maps_dir"]) / name / f"{first_id}.npy"
    binding = {"source_base_hash": _config_hash(config), "image_id": first_id,
               "source_map_sha256": sha256_file(source_map), "protocol_sha256": PROTOCOL_SHA256}
    if gate.exists():
        if json.loads((gate / "input_manifest.json").read_text()) != binding:
            raise ValueError(f"gate inputs changed; archive the old gate first: {gate}")
        return gate / "base.yaml", gate / "protocol.yaml"
    gate.mkdir(parents=True)
    cloned = json.loads(json.dumps(config))
    cloned["output"].update(per_image_csv=str(gate / "per_image.csv"),
                            units_csv=str(gate / "units.csv"),
                            float_maps_dir=str(gate / "maps_float"))
    map_copy = gate / "maps_float" / name / source_map.name
    map_copy.parent.mkdir(parents=True)
    shutil.copy2(source_map, map_copy)  # Never bind a gate sidecar to the original map.
    rows = read_csv(_resolve(config["output"]["per_image_csv"]))
    unit = identity(config)
    rows = rows[_unit_mask(rows, unit) & rows["image_id"].eq(first_id)
                & rows["metric"].isin(config["metrics"]["names"])]
    rows.to_csv(gate / "per_image.csv", index=False)
    summaries = [{**unit, "metric": row.metric, "mean": row.value, "std": 0.0,
                  "n": 1, "config_hash": _config_hash(cloned)} for row in rows.itertuples()]
    pd.DataFrame(summaries)[UNIT_COLUMNS].to_csv(gate / "units.csv", index=False)
    gate_protocol = json.loads(json.dumps(protocol))
    gate_protocol["output"] = {"trace_csv": str(gate / "trace.csv"),
                               "state_dir": str(gate / "state"),
                               "config_dir": str(gate / "configs"),
                               "run_log": str(gate / "run_log.jsonl")}
    for path, content in ((gate / "base.yaml", cloned), (gate / "protocol.yaml", gate_protocol)):
        path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    (gate / "input_manifest.json").write_text(json.dumps(binding, indent=2) + "\n")
    return gate / "base.yaml", gate / "protocol.yaml"


def validate_stability(config: dict, protocol: dict, expected: dict[str, int]) -> dict:
    unit = identity(config)
    rows = read_csv(_resolve(config["output"]["per_image_csv"]))
    rows = rows[_unit_mask(rows, unit) & rows["metric"].isin(STABILITY_METRICS)]
    if len(rows) != len(expected) * 2 or rows.duplicated(["image_id", "metric"]).any():
        raise ValueError("missing or duplicate stability rows")
    if any(set(rows[rows["metric"].eq(metric)]["image_id"]) != set(expected)
           for metric in STABILITY_METRICS):
        raise ValueError("stability ID coverage mismatch")
    summaries = read_csv(_resolve(config["output"]["units_csv"]))
    summaries = summaries[_unit_mask(summaries, unit) & summaries["metric"].isin(STABILITY_METRICS)]
    if (len(summaries) != 2 or summaries["config_hash"].nunique() != 1
            or set(summaries["metric"]) != set(STABILITY_METRICS)):
        raise ValueError("stability summary identity mismatch")
    digest = summaries.iloc[0]["config_hash"]
    template = _resolve(protocol["output"]["trace_csv"])
    name = "_".join(unit[key] for key in ("method", "model", "dataset"))
    trace = read_csv(template.with_name(f"{template.stem}_{name}_eval_{digest}{template.suffix}"))
    if len(trace) != len(expected) * 5 or trace.duplicated(["image_id", "repeat"]).any():
        raise ValueError("missing or duplicate trace repeats")
    if (set(trace["image_id"]) != set(expected) or set(trace["config_hash"]) != {digest}
            or not _unit_mask(trace, unit).all()
            or set(trace["protocol_version"]) != {protocol["protocol_version"]}):
        raise ValueError("trace identity mismatch")
    for image_id, target in expected.items():
        repeats = trace[trace["image_id"].eq(image_id)]
        if set(repeats["repeat"]) != set(range(5)) or not repeats["target"].eq(target).all():
            raise ValueError("trace repeat/target mismatch")
        for row in repeats.itertuples():
            if row.seed != stable_seed(unit["dataset"], image_id, row.repeat):
                raise ValueError("trace seed mismatch")
            if row.status == "valid":
                if not np.isfinite(row.spearman) or not np.isclose(row.score, row.spearman):
                    raise ValueError("valid Spearman/score mismatch")
            elif row.status not in {"nonfinite", "both_constant", "reference_constant",
                                    "candidate_constant", "undefined_spearman"} or row.score != 0:
                raise ValueError("invalid degenerate score/status")
        actual = rows[rows["image_id"].eq(image_id)].set_index("metric")["value"]
        if (not np.isfinite(repeats["score"]).all() or not repeats["score"].between(-1, 1).all()
                or not np.isclose(actual["stability_spearman"], repeats["score"].mean())
                or not np.isclose(actual["stability_valid_rate"], repeats["status"].eq("valid").mean())):
            raise ValueError("per-image aggregate mismatch")
    for row in summaries.itertuples():
        values = rows[rows["metric"].eq(row.metric)]["value"]
        if (row.n != len(expected) or not np.isclose(row.mean, values.mean())
                or not np.isclose(row.std, values.std(ddof=1) if len(values) > 1 else 0.0)):
            raise ValueError("unit aggregate mismatch")
    logs = [json.loads(line) for line in _resolve(protocol["output"]["run_log"]).read_text().splitlines()]
    relevant = [row for row in logs if all(row.get(key) == value for key, value in unit.items())
                and row.get("config_hash") == digest]
    if not relevant or relevant[-1]["processed"] != 0 or relevant[-1]["skipped"] != len(expected):
        raise ValueError("immediate resume did not skip the complete unit")
    return {**unit, "images": len(expected), "trace_rows": len(trace), "config_hash": digest,
            "resume_processed": 0, "resume_skipped": len(expected),
            "degenerate_repeats": int(trace["status"].ne("valid").sum())}


def invoke(config_path: Path, protocol_path: Path, limit: int | None = None) -> None:
    command = [sys.executable, "-u", str(ROOT / "experiments/run_stability.py"),
               "--config", str(config_path), "--protocol", str(protocol_path)]
    if limit is not None:
        command += ["--max-images", str(limit)]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", action="append", choices=UNITS, help="default: all 12 units")
    parser.add_argument("--preflight", action="store_true", help="check inputs without running attribution")
    args = parser.parse_args()
    protocol_path = ROOT / "configs/stability_protocol_v1.yaml"
    if sha256_file(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("frozen protocol YAML changed")
    protocol = _load_protocol(protocol_path)
    if not args.preflight and not torch.cuda.is_available():
        raise RuntimeError("the formal A batch requires a CUDA GPU; use --preflight to check inputs")
    names = args.unit or UNITS
    inputs = {}
    for name in names:
        config = _load_config(ROOT / "configs" / f"{name}.yaml")
        expected, hashes = preflight(config)
        inputs[name] = (config, expected, hashes)
        print(f"preflight OK: {name}, images={len(expected)}, maps={len(hashes)}", flush=True)
    if args.preflight:
        return
    output_paths = {_resolve(config["output"][key]) for config, _, _ in inputs.values()
                    for key in ("per_image_csv", "units_csv")}
    before = {path: base_rows(path) for path in output_paths}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    delivery = ROOT / "results/a_stability_batches" / stamp
    delivery.mkdir(parents=True)
    for index, path in enumerate(sorted(output_paths)):
        shutil.copy2(path, delivery / f"original_{index}_{path.name}")
    reports = []
    for name, (config, expected, hashes) in inputs.items():
        first_id = next(iter(expected))
        gate_config, gate_protocol = prepare_gate(config, protocol, name, first_id)
        invoke(gate_config, gate_protocol, 1)
        invoke(gate_config, gate_protocol, 1)
        gate_report = validate_stability(_load_config(gate_config), _load_protocol(gate_protocol),
                                         {first_id: expected[first_id]})
        check_base_unchanged(before)
        invoke(ROOT / "configs" / f"{name}.yaml", protocol_path)
        invoke(ROOT / "configs" / f"{name}.yaml", protocol_path)
        formal_report = validate_stability(config, protocol, expected)
        check_base_unchanged(before)
        if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
            raise ValueError("an original reference NPY was modified")
        reports.append({"unit": name, "gate": gate_report, "formal": formal_report})
        (delivery / "validation_report.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(f"validated: {name}, 460 images, 2300 repeats, resume skipped=460", flush=True)
    print(f"completed {len(reports)} units; receipt: {delivery / 'validation_report.json'}")


if __name__ == "__main__":
    main()
