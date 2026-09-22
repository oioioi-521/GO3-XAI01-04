from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments import run_stability
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

    trace_path = Path(_protocol(tmp_path)["output"]["trace_csv"])
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

    trace_path = Path(_protocol(tmp_path)["output"]["trace_csv"])
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
