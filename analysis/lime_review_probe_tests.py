"""D's manual review probes; run against the isolated E source, not D's runner.

Set LIME_REVIEW_SOURCE to E's archived source and PYTHONPATH to its isolated
dependencies, then run pytest on this file from E's source directory.
Synthetic fixtures only; no GPU/data gate.
The strict xfails document observed defects, not successful acceptance checks.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

source_root = Path(os.environ["LIME_REVIEW_SOURCE"]).resolve()
if not (source_root / "experiments/lime_budget_gate.py").is_file():
    raise RuntimeError("LIME_REVIEW_SOURCE must identify the isolated E checkout")
sys.path.insert(0, str(source_root))
from experiments import lime_budget_gate as gate  # noqa: E402
if gate.ROOT.resolve() != source_root:
    raise RuntimeError("Python imported a different experiments package; use a fresh process")


@pytest.fixture
def completed_gate(tmp_path, monkeypatch):
    config = gate._load_config(Path("configs/lime_resnet50_imagenet.yaml"))
    config["runtime"]["device"] = "cpu"

    class Dataset:
        rows = pd.DataFrame({"image_id": [f"fixture-{i}" for i in range(40)]})

        def __getitem__(self, index):
            return dict(image=torch.zeros(3, 2, 2), target=1,
                        image_id=f"fixture-{index}")

    report = dict(config_hash="d-review-synthetic-only", budgets=[8, 16],
                  seeds=[42, 43, 44], missing_assets=[])
    monkeypatch.setattr(gate, "plan", lambda *a: (config, Dataset(), copy.deepcopy(report)))
    monkeypatch.setattr(gate, "load_model", lambda **kw: torch.nn.Identity().eval())
    calls = []

    class Attributor:
        last_diagnostics = dict(n_superpixels=2, weighted_r2=0.5)

        def attribute(self, image, target, baseline):
            calls.append(target)
            return torch.tensor([[0., .25], [.75, 1.]], dtype=torch.float32)

    monkeypatch.setattr(gate, "_build_attributor", lambda *a: Attributor())
    output = tmp_path / "gate"
    monkeypatch.setattr("sys.argv", ["lime_budget_gate", "--config",
        "configs/lime_resnet50_imagenet.yaml", "--output", str(output),
        "--single-image-ms", "1"])
    gate.main()
    assert len(calls) == 246
    assert len(pd.read_csv(output / "timings.csv")) == 240
    return output, calls


def test_complete_gate_resume_skips_all_timed_maps(completed_gate):
    output, calls = completed_gate
    gate.main()
    assert len(calls) == 252  # Six warm-ups only, no repeated timed maps.
    assert len(pd.read_csv(output / "timings.csv")) == 240
    assert len(pd.read_csv(output / "comparisons.csv")) == 120
    assert json.loads((output / "summary.json").read_text())["n_images"] == 40


@pytest.mark.xfail(strict=True, reason="64d2ffd accepts a truncated completed-image record")
def test_resume_must_reject_incomplete_record(completed_gate):
    output, _ = completed_gate
    path = next((output / "records").glob("*.json"))
    record = json.loads(path.read_text())
    record["runs"] = []
    record["budget_comparisons"] = []
    record["seed_variance"] = []
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        gate.main()


@pytest.mark.xfail(strict=True, reason="64d2ffd trusts saved target/statistics without revalidation")
def test_resume_must_reject_changed_target_and_comparison(completed_gate):
    output, _ = completed_gate
    path = next((output / "records").glob("*.json"))
    record = json.loads(path.read_text())
    record["target"] = 0
    record["budget_comparisons"][0]["spearman"] = -123.0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        gate.main()


@pytest.mark.xfail(strict=True, reason="64d2ffd gate identity omits source image bytes")
def test_plan_identity_must_change_when_input_image_bytes_change(tmp_path, monkeypatch):
    config = gate._load_config(Path("configs/lime_resnet50_imagenet.yaml"))
    data = tmp_path / "data"
    data.mkdir()
    metadata = data / "metadata.csv"
    metadata.write_bytes(b"D review synthetic metadata fixture only\n")
    (data / "DATA_VERSION.json").write_text(json.dumps({"sha256": {
        "metadata.csv": hashlib.sha256(metadata.read_bytes()).hexdigest()}}))
    source = data / "fixture.bin"
    source.write_bytes(b"first-source-image-fixture")

    class Dataset:
        rows = pd.DataFrame({"image_id": [f"fixture-{i}" for i in range(40)],
                             "image_path": ["fixture.bin"] * 40})

        def __len__(self):
            return 40

    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_resolve", lambda p: tmp_path / p)
    monkeypatch.setattr(gate, "MetadataDataset", lambda *a, **kw: Dataset())
    monkeypatch.setattr(gate, "build_prediction_context", lambda **kw: {
        "checkpoint_sha256": "fixed-synthetic-weights"})
    _, _, first = gate.plan(config, [8, 16], [42, 43, 44], tmp_path / "gate")
    source.write_bytes(b"replacement-source-image-fixture")
    _, _, second = gate.plan(config, [8, 16], [42, 43, 44], tmp_path / "gate")
    assert first["config_hash"] != second["config_hash"]
