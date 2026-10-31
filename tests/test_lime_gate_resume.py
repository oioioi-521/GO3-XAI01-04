"""Synthetic CPU regression for the real plan -> gate -> resume CLI path."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from experiments import lime_budget_gate as gate
from experiments.lime_gate_integrity import output_lock, tensor_sha


def setup_gate(tmp_path, monkeypatch):
    project = tmp_path / "project"
    data = project / "data"
    data.mkdir(parents=True)
    source = data / "fixture.bin"
    source.write_bytes(b"synthetic source bytes, not an experiment image")
    rows = [dict(image_id=f"fixture-{i}", image_path="fixture.bin", class_id=1,
                 dataset="imagenet", split="debug") for i in range(40)]
    with (data / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (data / "DATA_VERSION.json").write_text(json.dumps({"sha256": {
        "metadata.csv": hashlib.sha256((data / "metadata.csv").read_bytes()).hexdigest()}}))
    config = yaml.safe_load(Path("configs/lime_resnet50_imagenet.yaml").read_text())
    config["runtime"]["device"] = "cpu"
    config_path = project / "fixture.yaml"
    config_path.write_text(yaml.safe_dump(config))
    state = SimpleNamespace(calls=0, loads=0, delta=0., bad_diagnostic=False, change_source=False)

    class Dataset:
        def __init__(self, *a, **kw):
            self.rows = pd.DataFrame(rows)

        def __len__(self):
            return 40

        def __getitem__(self, index):
            return dict(image=torch.zeros(3, 2, 2, dtype=torch.float32) + state.delta,
                        image_id=rows[index]["image_id"], target=1)

    class Attributor:
        def __init__(self, params):
            self.last_diagnostics = dict(num_samples=params["num_samples"], seed=params["seed"],
                                         n_superpixels=2, weighted_r2=-.5)

        def attribute(self, image, target, baseline):
            state.calls += 1
            if state.bad_diagnostic:
                self.last_diagnostics["num_samples"] = 999
            if state.change_source and state.calls == 7:
                source.write_bytes(b"changed during attribution")
            return torch.tensor([[0., .25], [.75, 1.]], dtype=torch.float32)

    def load(**kw):
        state.loads += 1
        return torch.nn.Identity().eval()

    monkeypatch.setattr(gate, "ROOT", project)
    monkeypatch.setattr(gate, "MetadataDataset", Dataset)
    monkeypatch.setattr(gate, "build_prediction_context", lambda **kw: dict(
        dataset="imagenet", split="debug", model="resnet50", output_activation="softmax",
        checkpoint_sha256="a" * 64))
    monkeypatch.setattr(gate, "load_model", load)
    monkeypatch.setattr(gate, "_build_attributor", lambda method, model, params: Attributor(params))
    monkeypatch.setattr(gate.importlib.metadata, "version", lambda name: "synthetic-test")
    output = tmp_path / "gate"
    monkeypatch.setattr("sys.argv", ["gate", "--config", str(config_path), "--output", str(output),
                                    "--budgets", "8", "16", "--seeds", "42", "43", "44", "--single-image-ms", "1"])
    return SimpleNamespace(project=project, source=source, config=config, output=output, state=state)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    fixture = setup_gate(tmp_path, monkeypatch)
    gate.main()
    assert fixture.state.calls == 246
    return fixture


def snapshot(output):
    return {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_complete_resume_does_not_load_model_or_call_attributor(completed, monkeypatch):
    before = snapshot(completed.output)
    monkeypatch.setattr(gate, "load_model", lambda **kw: pytest.fail("complete resume loaded model"))
    gate.main()
    assert completed.state.calls == 246 and completed.state.loads == 1
    after = snapshot(completed.output)
    assert all(after[p] == v for p, v in before.items() if p != "summary.json")
    summary = json.loads(after["summary.json"])
    assert summary["n_images"] == 40 and "pending" in summary["budget_status"]
    assert (summary["processed"], summary["skipped"], summary["warmup_calls"]) == (0, 40, 0)
    assert (summary["run_count"], summary["budget_comparison_count"], summary["seed_pair_count"]) == (240, 120, 240)
    assert len(pd.read_csv(completed.output / "timings.csv")) == 240


@pytest.mark.parametrize("change", ["missing_runs", "duplicate_runs", "target", "comparison", "variance",
    "seed_pairs", "nan_time", "negative_time", "nan_r2", "budget", "map_hash", "map_bytes",
    "path", "dtype", "range", "source_hash", "tensor_hash", "duplicate_json"])
def test_corrupt_resume_rejected_before_any_model_or_receipt_write(completed, change, monkeypatch):
    path = next((completed.output / "records").glob("*.json"))
    record = json.loads(path.read_text())
    run = record["runs"][0]
    if change == "missing_runs":
        record["runs"] = []
    elif change == "duplicate_runs":
        record["runs"][1] = run.copy()
    elif change == "target":
        record["target"] = 0
    elif change == "comparison":
        record["budget_comparisons"][0]["spearman"] = .1
    elif change == "variance":
        record["seed_variance"][0]["pixel_variance_mean"] = .7
    elif change == "seed_pairs":
        record["seed_variance"][0]["seed_pairs"].pop()
    elif change == "nan_time":
        run["time_ms"] = float("nan")
    elif change == "negative_time":
        run["time_ms"] = -1
    elif change == "nan_r2":
        run["weighted_r2"] = float("nan")
    elif change == "budget":
        run["num_samples"] = 999
    elif change == "map_hash":
        run["map_sha256"] = "0" * 64
    elif change in ("map_bytes", "dtype", "range"):
        target = completed.output / run["map_path"]
        arr = np.load(target)
        np.save(target, (1-arr) if change == "map_bytes" else arr.astype("float64") if change == "dtype" else arr+1)
        run["map_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    elif change == "path":
        run["map_path"] = "maps_float/../../secret.npy"
    elif change == "source_hash":
        record["source_image_sha256"] = "0" * 64
    elif change == "tensor_hash":
        record["preprocessed_image_sha256"] = "0" * 64
    if change == "duplicate_json":
        path.write_text('{"image_id":"a","image_id":"b"}')
    else:
        path.write_text(json.dumps(record))
    before = snapshot(completed.output)
    monkeypatch.setattr(gate, "load_model", lambda **kw: pytest.fail("bad record loaded model"))
    with pytest.raises((ValueError, KeyError)):
        gate.main()
    assert snapshot(completed.output) == before


@pytest.mark.parametrize("change", ["source", "tensor", "machine", "code", "plan_hash", "legacy", "unknown_record"])
def test_identity_changes_rejected_without_mixing_or_upgrading(completed, change, monkeypatch):
    if change == "source":
        completed.source.write_bytes(b"replacement bytes")
    elif change == "tensor":
        completed.state.delta = .25
    elif change in ("machine", "code"):
        original = gate.execution_identity
        def changed(config):
            result = original(config)
            result["machine_sha256" if change == "machine" else "code_sha256"] = "changed"
            return result
        monkeypatch.setattr(gate, "execution_identity", changed)
    elif change in ("legacy", "plan_hash"):
        p = completed.output / "plan.json"
        data = json.loads(p.read_text())
        data.pop("gate_schema_version") if change == "legacy" else data.update(config_hash="bad")
        p.write_text(json.dumps(data))
    elif change == "unknown_record":
        (completed.output / "records/foreign.json").write_text("{}")
    before = snapshot(completed.output)
    with pytest.raises((ValueError, KeyError)):
        gate.main()
    assert snapshot(completed.output) == before


def test_partial_resume_computes_only_missing_image(completed):
    # Deliberately remove ONE disposable synthetic fixture record, not user data.
    next((completed.output / "records").glob("*.json")).unlink()
    gate.main()
    assert completed.state.calls == 246 + 6 + 6
    assert len(pd.read_csv(completed.output / "timings.csv")) == 240


@pytest.mark.parametrize("change", ["bad_diagnostic", "change_source"])
def test_new_corrupt_computation_never_publishes_summary(tmp_path, monkeypatch, change):
    fixture = setup_gate(tmp_path, monkeypatch)
    setattr(fixture.state, change, True)
    with pytest.raises(ValueError):
        gate.main()
    assert not (fixture.output / "summary.json").exists()


def test_tensor_hash_binds_shape_values_and_dtype():
    a = torch.zeros(3, 2, 2)
    assert tensor_sha(a) != tensor_sha(a + .1)
    assert tensor_sha(a) != tensor_sha(a.reshape(3, 1, 4))
    with pytest.raises(ValueError):
        tensor_sha(a.double())


def test_output_lock_is_exclusive_and_released_after_error(tmp_path):
    with pytest.raises(RuntimeError):
        with output_lock(tmp_path):
            with pytest.raises(ValueError, match="already locked"):
                with output_lock(tmp_path):
                    pytest.fail("duplicate run obtained gate lock")
            raise RuntimeError("synthetic interruption")
    with output_lock(tmp_path):
        pass  # No stale PID/lock-file deletion required after an exception.
