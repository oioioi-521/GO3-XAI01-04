from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments import run_stability, run_unit
from preprocessing.dataset import MEAN, STD


class TinyClassifier(torch.nn.Module):
    def forward(self, inputs):
        score = inputs[:, 0].mean(dim=(1, 2)) - inputs[:, 1].mean(dim=(1, 2))
        return torch.stack((score, -score), dim=1)


def _normalized_pattern() -> torch.Tensor:
    pixels = torch.zeros(3, 4, 4)
    pixels[0] = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 15
    pixels[1] = pixels[0].flip(1)
    pixels[2] = 0.5
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (pixels - mean) / std


class SyntheticDataset:
    def __init__(self, dataset, split="debug", limit=None):
        assert (dataset, split) == ("synthetic", "debug")
        self.samples = [{
            "image": _normalized_pattern(),
            "image_id": "synthetic-0001",
            "target": 0,
            "class_name": "positive",
            "image_path": "synthetic.png",
        }]
        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


class SpatialAttributor:
    def attribute(self, image, target, baseline=None):
        assert target == 0
        spatial = image[0, 0].detach().cpu()
        span = spatial.max() - spatial.min()
        return ((spatial - spatial.min()) / span).to(torch.float32)


def _base_config(tmp_path: Path) -> dict:
    results = tmp_path / "results"
    return {
        "method": "occlusion",
        "dataset": {"name": "synthetic", "split": "debug"},
        "model": {
            "name": "tiny",
            "weights": "none",
            "num_classes": 2,
            "output_activation": "softmax",
            "checkpoint": None,
        },
        "attribution": {
            "window_height": 2,
            "window_width": 2,
            "stride_height": 2,
            "stride_width": 2,
        },
        "metrics": {"names": ["efficiency_time_ms"]},
        "runtime": {"device": "cpu", "max_images": 1, "resume": True},
        "output": {
            "per_image_csv": str(results / "per_image.csv"),
            "units_csv": str(results / "units.csv"),
            "predictions_csv": str(results / "predictions.csv"),
            "config_dir": str(results / "configs"),
            "run_log": str(results / "run_log.jsonl"),
            "save_maps": True,
            "maps_dir": str(results / "maps"),
            "save_float_maps": True,
            "float_maps_dir": str(results / "maps_float"),
        },
    }


def _protocol(tmp_path: Path) -> dict:
    results = tmp_path / "results"
    return {
        "protocol_version": "rgb-gaussian-spearman-v1",
        "noise_distribution": "gaussian",
        "noise_domain": "rgb_0_1",
        "sigma": 0.005,
        "clip": [0.0, 1.0],
        "repeats": 5,
        "target": "metadata",
        "similarity": "spearman",
        "direction": "max",
        "top_fraction": 0.1,
        "seed_fields": ["dataset", "image_id", "repeat"],
        "degenerate_score": 0.0,
        "runtime": {"device": "cpu", "warmup_runs": 1},
        "output": {
            "trace_csv": str(results / "stability_trace.csv"),
            "state_dir": str(results / "stability_run_state"),
            "config_dir": str(results / "configs"),
            "run_log": str(results / "stability_run_log.jsonl"),
        },
    }


def _write_inputs(tmp_path: Path) -> tuple[dict, Path, Path]:
    config = _base_config(tmp_path)
    protocol = _protocol(tmp_path)
    config_path = tmp_path / "base.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")

    map_dir = (
        Path(config["output"]["float_maps_dir"])
        / "occlusion_tiny_synthetic"
    )
    map_dir.mkdir(parents=True)
    reference = SpatialAttributor().attribute(
        SyntheticDataset("synthetic").samples[0]["image"].unsqueeze(0), target=0
    )
    np.save(map_dir / "synthetic-0001.npy", reference.numpy(), allow_pickle=False)

    per_image = Path(config["output"]["per_image_csv"])
    per_image.parent.mkdir(parents=True, exist_ok=True)
    with per_image.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_stability.PER_IMAGE_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "image_id": "synthetic-0001",
            "dataset": "synthetic",
            "model": "tiny",
            "method": "occlusion",
            "metric": "efficiency_time_ms",
            "value": 1.25,
            "time_ms": 1.25,
        })
    units = Path(config["output"]["units_csv"])
    with units.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_stability.UNIT_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "method": "occlusion",
            "model": "tiny",
            "dataset": "synthetic",
            "metric": "efficiency_time_ms",
            "mean": 1.25,
            "std": 0.0,
            "n": 1,
            "config_hash": "original-base",
        })
    return config, config_path, protocol_path


def _patch_runtime(monkeypatch) -> None:
    monkeypatch.setattr(
        run_stability, "load_model", lambda **kwargs: TinyClassifier().eval()
    )
    monkeypatch.setattr(run_stability, "MetadataDataset", SyntheticDataset)
    monkeypatch.setattr(
        run_stability,
        "_build_attributor",
        lambda method, model, attribution: SpatialAttributor(),
    )


def _trace_path(tmp_path: Path, split: str = "debug") -> Path:
    paths = list((tmp_path / "results").glob(f"stability_trace_*_{split}_*.csv"))
    assert len(paths) == 1
    return paths[0]


def test_formal_stability_appends_metrics_trace_and_resumes(tmp_path, monkeypatch):
    config, config_path, protocol_path = _write_inputs(tmp_path)
    _patch_runtime(monkeypatch)

    run_stability.run(config_path, protocol_path)
    run_stability.run(config_path, protocol_path)

    with Path(config["output"]["per_image_csv"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert {row["metric"] for row in rows} == {
        "efficiency_time_ms",
        "stability_spearman",
        "stability_valid_rate",
    }
    scores = {row["metric"]: float(row["value"]) for row in rows}
    assert -1.0 <= scores["stability_spearman"] <= 1.0
    assert scores["stability_valid_rate"] == 1.0

    trace_path = _trace_path(tmp_path)
    with trace_path.open(newline="", encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    assert len(trace) == 5
    assert {int(row["repeat"]) for row in trace} == set(range(5))
    assert {row["status"] for row in trace} == {"valid"}
    assert len({row["config_hash"] for row in trace}) == 1
    assert {row["protocol_version"] for row in trace} == {
        "rgb-gaussian-spearman-v1"
    }

    with Path(config["output"]["units_csv"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        units = list(csv.DictReader(handle))
    assert {row["metric"] for row in units} == {
        "efficiency_time_ms",
        "stability_spearman",
        "stability_valid_rate",
    }
    assert next(row for row in units if row["metric"] == "efficiency_time_ms")[
        "config_hash"
    ] == "original-base"

    logs = [
        json.loads(line)
        for line in Path(_protocol(tmp_path)["output"]["run_log"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["processed"], row["skipped"]) for row in logs] == [(1, 0), (0, 1)]
    assert all(row["repeats"] == 5 for row in logs)


def test_resume_replaces_partial_trace_without_touching_base_metrics(
    tmp_path, monkeypatch
):
    config, config_path, protocol_path = _write_inputs(tmp_path)
    _patch_runtime(monkeypatch)
    run_stability.run(config_path, protocol_path)

    per_image_path = Path(config["output"]["per_image_csv"])
    with per_image_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_stability.PER_IMAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(
            row for row in rows if row["metric"] != "stability_valid_rate"
        )

    trace_path = _trace_path(tmp_path)
    with trace_path.open(newline="", encoding="utf-8") as handle:
        trace = list(csv.DictReader(handle))
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_stability.TRACE_COLUMNS)
        writer.writeheader()
        writer.writerows(trace[:-1])

    run_stability.run(config_path, protocol_path)
    with per_image_path.open(newline="", encoding="utf-8") as handle:
        rerun_rows = list(csv.DictReader(handle))
    with trace_path.open(newline="", encoding="utf-8") as handle:
        rerun_trace = list(csv.DictReader(handle))
    assert len(rerun_rows) == 3
    assert len(rerun_trace) == 5
    assert sum(row["metric"] == "efficiency_time_ms" for row in rerun_rows) == 1


def test_protocol_is_strict_and_original_float_map_is_required(tmp_path, monkeypatch):
    config, config_path, protocol_path = _write_inputs(tmp_path)
    _patch_runtime(monkeypatch)
    protocol = _protocol(tmp_path)
    protocol["sigma"] = 0.01
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="sigma"):
        run_stability.run(config_path, protocol_path)

    protocol_path.write_text(yaml.safe_dump(_protocol(tmp_path)), encoding="utf-8")
    map_path = (
        Path(config["output"]["float_maps_dir"])
        / "occlusion_tiny_synthetic"
        / "synthetic-0001.npy"
    )
    map_path.unlink()
    with pytest.raises(FileNotFoundError, match="missing original float32"):
        run_stability.run(config_path, protocol_path)


def test_reference_map_rejects_another_config_or_changed_npy(tmp_path, monkeypatch):
    config, config_path, protocol_path = _write_inputs(tmp_path)
    _patch_runtime(monkeypatch)
    run_stability.run(config_path, protocol_path)

    map_path = (
        Path(config["output"]["float_maps_dir"])
        / "occlusion_tiny_synthetic"
        / "synthetic-0001.npy"
    )
    sidecar = map_path.with_suffix(".npy.provenance.json")
    assert json.loads(sidecar.read_text(encoding="utf-8"))["binding"] == "verified_recomputation"
    trace_before = _trace_path(tmp_path).read_bytes()

    changed = _base_config(tmp_path)
    changed["attribution"]["stride_width"] = 1
    config_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="reference map provenance mismatch"):
        run_stability.run(config_path, protocol_path)
    assert _trace_path(tmp_path).read_bytes() == trace_before

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    np.save(map_path, np.load(map_path, allow_pickle=False).copy()[:, ::-1], allow_pickle=False)
    with pytest.raises(ValueError, match="reference map provenance mismatch"):
        run_stability.run(config_path, protocol_path)


def test_reference_map_rejects_changed_checkpoint_bytes(tmp_path, monkeypatch):
    config, config_path, protocol_path = _write_inputs(tmp_path)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"first-checkpoint")
    config["model"]["checkpoint"] = str(checkpoint)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _patch_runtime(monkeypatch)

    run_stability.run(config_path, protocol_path)
    trace_before = _trace_path(tmp_path).read_bytes()
    checkpoint.write_bytes(b"changed-checkpoint")

    with pytest.raises(ValueError, match="reference map provenance mismatch"):
        run_stability.run(config_path, protocol_path)
    assert _trace_path(tmp_path).read_bytes() == trace_before


class SplitSyntheticDataset(SyntheticDataset):
    def __init__(self, dataset, split="debug", limit=None):
        assert split in {"debug", "eval"}
        super().__init__(dataset, split="debug", limit=limit)


def test_debug_eval_debug_preserves_independent_trace_and_state(tmp_path, monkeypatch):
    debug_dir = tmp_path / "debug"
    eval_dir = tmp_path / "eval"
    debug_dir.mkdir()
    eval_dir.mkdir()
    debug_config, debug_path, debug_protocol_path = _write_inputs(debug_dir)
    eval_config, eval_path, eval_protocol_path = _write_inputs(eval_dir)
    eval_config["dataset"]["split"] = "eval"
    eval_path.write_text(yaml.safe_dump(eval_config), encoding="utf-8")

    shared = tmp_path / "shared"
    protocol = _protocol(debug_dir)
    protocol["output"] = {
        "trace_csv": str(shared / "stability_trace.csv"),
        "state_dir": str(shared / "state"),
        "config_dir": str(shared / "configs"),
        "run_log": str(shared / "run_log.jsonl"),
    }
    debug_protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    eval_protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(run_stability, "MetadataDataset", SplitSyntheticDataset)

    run_stability.run(debug_path, debug_protocol_path)
    run_stability.run(eval_path, eval_protocol_path)
    run_stability.run(debug_path, debug_protocol_path)

    trace_paths = list(shared.glob("stability_trace_*.csv"))
    assert len(trace_paths) == 2
    assert {"_debug_", "_eval_"} == {
        "_debug_" if "_debug_" in path.name else "_eval_" for path in trace_paths
    }
    assert all(len(list(csv.DictReader(path.open(encoding="utf-8")))) == 5 for path in trace_paths)
    states = list((shared / "state").glob("stability_*.json"))
    assert len(states) == 2
    logs = [json.loads(line) for line in (shared / "run_log.jsonl").read_text().splitlines()]
    assert [(row["processed"], row["skipped"]) for row in logs] == [(1, 0), (1, 0), (0, 1)]

    # Reusing one base long-table across splits is rejected before cleanup.
    eval_config["output"]["per_image_csv"] = debug_config["output"]["per_image_csv"]
    eval_config["output"]["units_csv"] = debug_config["output"]["units_csv"]
    eval_path.write_text(yaml.safe_dump(eval_config), encoding="utf-8")
    with pytest.raises(ValueError, match="separate per-image and unit CSVs"):
        run_stability.run(eval_path, eval_protocol_path)


def test_base_resume_preserves_stability_and_force_fails_closed(tmp_path, monkeypatch):
    config = _base_config(tmp_path)
    config_path = tmp_path / "base.yaml"
    protocol_path = tmp_path / "protocol.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    protocol_path.write_text(yaml.safe_dump(_protocol(tmp_path)), encoding="utf-8")
    _patch_runtime(monkeypatch)
    monkeypatch.setattr(run_unit, "load_model", lambda **kwargs: TinyClassifier().eval())
    monkeypatch.setattr(run_unit, "MetadataDataset", SyntheticDataset)
    monkeypatch.setattr(
        run_unit, "_build_attributor", lambda method, model, attribution: SpatialAttributor()
    )

    run_unit.run(config_path)
    map_path = (
        Path(config["output"]["float_maps_dir"])
        / "occlusion_tiny_synthetic"
        / "synthetic-0001.npy"
    )
    assert json.loads(map_path.with_suffix(".npy.provenance.json").read_text())["binding"] == "base_runner"
    run_stability.run(config_path, protocol_path)
    units_path = Path(config["output"]["units_csv"])
    per_image_path = Path(config["output"]["per_image_csv"])
    with units_path.open(encoding="utf-8", newline="") as handle:
        before = {row["metric"]: row for row in csv.DictReader(handle)}

    run_unit.run(config_path)
    with units_path.open(encoding="utf-8", newline="") as handle:
        after = {row["metric"]: row for row in csv.DictReader(handle)}
    assert after["stability_spearman"] == before["stability_spearman"]
    assert after["stability_valid_rate"] == before["stability_valid_rate"]
    assert after["efficiency_time_ms"]["config_hash"] == run_unit._config_hash(config)

    per_image_before = per_image_path.read_bytes()
    units_before = units_path.read_bytes()
    with pytest.raises(ValueError, match="invalidate existing stability rows"):
        run_unit.run(config_path, force=True)
    assert per_image_path.read_bytes() == per_image_before
    assert units_path.read_bytes() == units_before
